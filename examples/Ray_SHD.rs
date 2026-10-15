extern crate arrayfire;
extern crate raybnn;

use arrayfire as af;
use nohash_hasher::IntMap;
use raybnn::export::dataloader_f32::{labels_to_onehot_f32, ShdDat};
use raybnn::physics::update_f32::add_neuron_option_type;

const BACK_END: af::Backend = af::Backend::CUDA;
const DEVICE: i32 = 0;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn shd_file(base: &str, split: &str, suffix: &str) -> String {
    format!("{base}/shd_{split}_dt2ms_T700_{suffix}")
}

fn shd_meta_file(base: &str, split: &str) -> String {
    format!("{base}/shd_{split}_dt2ms_T700.meta.txt")
}

fn load_shd_batch(
    dataset: &mut ShdDat,
    start: usize,
    batch_size: u64,
    input_size: u64,
    traj_size: u64,
    output_size: u64,
) -> (af::Array<f32>, af::Array<f32>, af::Array<f32>) {
    let (x_cbt, y_idx) = dataset.load_batch_cbt_f32(start, batch_size as usize);
    let x = af::Array::new(
        &x_cbt,
        af::Dim4::new(&[input_size, batch_size, traj_size, 1]),
    );
    let x_bin = af::gt(&x, &af::constant(0.0f32, x.dims()), false).cast::<f32>();

    let y_onehot = labels_to_onehot_f32(&y_idx, output_size as usize);
    let y = af::Array::new(
        &y_onehot,
        af::Dim4::new(&[output_size, batch_size, 1, 1]),
    );
    let y_seq = af::tile(&y, af::Dim4::new(&[1, 1, traj_size, 1]));

    (x_bin, y, y_seq)
}

fn select_raybnn_outputs(
    q: &af::Array<f32>,
    output_rows: &af::Array<i32>,
    batch_size: u64,
    proc_num: u64,
    q_slices: u64,
) -> af::Array<f32> {
    let batch_seq = af::Seq::new(0.0, (batch_size - 1) as f32, 1.0);
    let time_seq = af::Seq::new((proc_num - 1) as f32, (q_slices - 1) as f32, 1.0);

    let mut idx = af::Indexer::default();
    idx.set_index(output_rows, 0, None);
    idx.set_index(&batch_seq, 1, None);
    idx.set_index(&time_seq, 2, None);
    af::index_gen(q, idx)
}

fn accuracy_from_sequence(yhat_seq: &af::Array<f32>, y_onehot: &af::Array<f32>) -> (u64, u64) {
    let yhat_rate = af::mean(yhat_seq, 2);
    let (_, pred_idx) = af::imax(&yhat_rate, 0);
    let (_, true_idx) = af::imax(y_onehot, 0);

    let mut pred = vec![0u32; pred_idx.elements()];
    let mut truth = vec![0u32; true_idx.elements()];
    pred_idx.host(&mut pred);
    true_idx.host(&mut truth);

    let correct = pred
        .iter()
        .zip(truth.iter())
        .filter(|(p, t)| p == t)
        .count() as u64;

    (correct, pred.len() as u64)
}

fn append_predictions(
    yhat_seq: &af::Array<f32>,
    y_onehot: &af::Array<f32>,
    all_true: &mut Vec<f32>,
    all_pred: &mut Vec<f32>,
) {
    let yhat_rate = af::mean(yhat_seq, 2);
    let (_, pred_idx) = af::imax(&yhat_rate, 0);
    let (_, true_idx) = af::imax(y_onehot, 0);

    let mut pred = vec![0u32; pred_idx.elements()];
    let mut truth = vec![0u32; true_idx.elements()];
    pred_idx.host(&mut pred);
    true_idx.host(&mut truth);

    for (t, p) in truth.iter().zip(pred.iter()) {
        all_true.push(*t as f32);
        all_pred.push(*p as f32);
    }
}

fn main() {
    af::set_backend(BACK_END);
    af::set_device(DEVICE);
    af::set_seed(10);

    af::sync(DEVICE);
    af::device_gc();

    let shd_dir = std::env::var("SHD_DAT_DIR")
        .unwrap_or_else(|_| "/home/yilin/Downloads/TwoPatterns/shd_dat".to_string());

    let train_meta = shd_meta_file(&shd_dir, "train");
    let train_x = shd_file(&shd_dir, "train", "X_u8.dat");
    let train_y = shd_file(&shd_dir, "train", "y_i64.dat");
    let test_meta = shd_meta_file(&shd_dir, "test");
    let test_x = shd_file(&shd_dir, "test", "X_u8.dat");
    let test_y = shd_file(&shd_dir, "test", "y_i64.dat");

    let mut shd_train = ShdDat::open(&train_meta, &train_x, &train_y);
    let mut shd_test = ShdDat::open(&test_meta, &test_x, &test_y);

    let input_size = shd_train.meta.c as u64;
    let max_input_size = input_size;
    let traj_size = shd_train.meta.t as u64;
    let output_size = 20u64;
    let max_output_size = output_size;

    assert_eq!(shd_test.meta.c as u64, input_size);
    assert_eq!(shd_test.meta.t as u64, traj_size);

    let batch_size = env_u64("SHD_BATCH_SIZE", 250);
    let hidden_size = env_u64("SHD_HIDDEN_SIZE", 1200);
    let conn_num = env_u64("SHD_CONN_NUM", 50);
    let max_epoch = env_u64("SHD_EPOCHS", 1200);
    let proc_num = env_u64("SHD_PROC_NUM", 25);
    let learning_rate_scale = env_f32("SHD_LR_SCALE", 0.002);

    let train_steps = (shd_train.meta.n as u64 / batch_size).max(1);
    let test_steps = (shd_test.meta.n as u64 / batch_size).max(1);
    let max_neuron_size = input_size + hidden_size + output_size;
    let dir_path = "/tmp/".to_string();

    println!(
        "RayBNN SHD train={} test={} input={} traj={} classes={} batch={} hidden={} epochs={}",
        shd_train.meta.n,
        shd_test.meta.n,
        input_size,
        traj_size,
        output_size,
        batch_size,
        hidden_size,
        max_epoch
    );

    let mut arch_search = raybnn::interface::automatic_f32::create_start_archtecture(
        input_size,
        max_input_size,
        output_size,
        max_output_size,
        max_neuron_size,
        batch_size,
        traj_size,
        &dir_path,
    );

    arch_search.neural_network.netdata.proc_num = proc_num;
    arch_search.neural_network.netdata.input_size = input_size;
    arch_search.neural_network.netdata.neuron_std = 0.001;
    arch_search.neural_network.netdata.con_rad =
        (arch_search.neural_network.netdata.sphere_rad / proc_num as f32) * 2.0;

    arch_search.neural_network.neuron_pos = af::rows(
        &arch_search.neural_network.neuron_pos,
        input_size as i64,
        (arch_search.neural_network.neuron_pos.dims()[0] - 1) as i64,
    );

    let input_neurons = raybnn::physics::initial_f32::create_spaced_input_neuron_on_sphere_1D(
        arch_search.neural_network.netdata.sphere_rad + 0.2,
        input_size,
    );
    arch_search.neural_network.neuron_pos =
        af::join(0, &input_neurons, &arch_search.neural_network.neuron_pos);
    arch_search.neural_network.netdata.active_size =
        arch_search.neural_network.neuron_pos.dims()[0];

    raybnn::physics::initial_f32::assign_neuron_idx_with_buffer(
        max_input_size,
        max_output_size,
        &arch_search.neural_network.netdata,
        &arch_search.neural_network.neuron_pos,
        &mut arch_search.neural_network.neuron_idx,
    );

    let current_hidden =
        arch_search.neural_network.neuron_idx.dims()[0] - input_size - output_size;
    if hidden_size > current_hidden {
        let add_neuron_options = add_neuron_option_type {
            new_active_size: hidden_size - current_hidden,
            init_connection_num: conn_num,
            input_neuron_con_rad: 60.0 * arch_search.neural_network.netdata.neuron_rad,
            hidden_neuron_con_rad: 110.0 * arch_search.neural_network.netdata.neuron_rad,
            output_neuron_con_rad: 60.0 * arch_search.neural_network.netdata.neuron_rad,
        };

        raybnn::physics::update_f32::add_neuron_to_existing3(
            &add_neuron_options,
            &mut arch_search,
        );
    }

    raybnn::physics::initial_f32::assign_neuron_idx_with_buffer(
        max_input_size,
        max_output_size,
        &arch_search.neural_network.netdata,
        &arch_search.neural_network.neuron_pos,
        &mut arch_search.neural_network.neuron_idx,
    );

    arch_search.neural_network.netdata.active_size =
        arch_search.neural_network.neuron_idx.dims()[0];

    let w_row_idx_coo =
        raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);
    arch_search.neural_network.WRowIdxCSR = raybnn::graph::large_sparse_i32::COO_to_CSR(
        &w_row_idx_coo,
        arch_search.neural_network.netdata.neuron_size,
    );

    let traj_steps = traj_size + proc_num - 1;
    let state_dims = af::Dim4::new(&[
        arch_search.neural_network.netdata.neuron_size,
        batch_size,
        traj_steps,
        1,
    ]);
    let mut z = af::constant::<f32>(0.0, state_dims);
    let mut q = af::constant::<f32>(0.0, state_dims);

    let x_dims = af::Dim4::new(&[input_size, batch_size, traj_steps, 1]);
    let mut x_state = af::constant::<f32>(0.0, x_dims);

    let active_size = arch_search.neural_network.neuron_idx.dims()[0];
    let output_rows = af::rows(
        &arch_search.neural_network.neuron_idx,
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );
    let q_slices = q.dims()[2];

    let total_param_size = arch_search.neural_network.network_params.dims()[0];
    let opt_dims = af::Dim4::new(&[total_param_size, 1, 1, 1]);
    let mut mt = af::constant::<f32>(0.0, opt_dims);
    let mut vt = af::constant::<f32>(0.0, opt_dims);
    let mut grad = af::constant::<f32>(0.0, opt_dims);

    let mut wseqs = [af::Seq::<i32>::default()];
    let mut hseqs = [af::Seq::<i32>::default()];
    let mut aseqs = [af::Seq::<i32>::default()];
    let mut bseqs = [af::Seq::<i32>::default()];
    let mut cseqs = [af::Seq::<i32>::default()];
    let mut dseqs = [af::Seq::<i32>::default()];
    let mut eseqs = [af::Seq::<i32>::default()];

    let mut idxsel_out: IntMap<i64, af::Array<i32>> = IntMap::default();
    let mut valsel_out: IntMap<i64, af::Array<i32>> = IntMap::default();
    let mut cvec_out: IntMap<i64, af::Array<i32>> = IntMap::default();
    let mut dxsel_out: IntMap<i64, af::Array<i32>> = IntMap::default();
    let mut nrows_out: IntMap<i64, u64> = IntMap::default();
    let mut sparseval_out: IntMap<i64, af::Array<i32>> = IntMap::default();
    let mut sparsecol_out: IntMap<i64, af::Array<i32>> = IntMap::default();
    let mut sparserow_out: IntMap<i64, af::Array<i32>> = IntMap::default();

    let mut hidxsel_out: IntMap<i64, af::Array<i32>> = IntMap::default();
    let mut aidxsel_out: IntMap<i64, af::Array<i32>> = IntMap::default();
    let mut bidxsel_out: IntMap<i64, af::Array<i32>> = IntMap::default();
    let mut cidxsel_out: IntMap<i64, af::Array<i32>> = IntMap::default();
    let mut didxsel_out: IntMap<i64, af::Array<i32>> = IntMap::default();
    let mut eidxsel_out: IntMap<i64, af::Array<i32>> = IntMap::default();
    let mut combidxsel_out: IntMap<i64, af::Array<i32>> = IntMap::default();

    let mut daseqs_out: IntMap<i64, [af::Seq<i32>; 2]> = IntMap::default();
    let mut dbseqs_out: IntMap<i64, [af::Seq<i32>; 2]> = IntMap::default();
    let mut dcseqs_out: IntMap<i64, [af::Seq<i32>; 2]> = IntMap::default();
    let mut ddseqs_out: IntMap<i64, [af::Seq<i32>; 2]> = IntMap::default();
    let mut deseqs_out: IntMap<i64, [af::Seq<i32>; 2]> = IntMap::default();

    let w_values_len = arch_search.neural_network.WColIdx.dims()[0];
    let h_values_len = (total_param_size - w_values_len) / 6;

    raybnn::graph::path_f32::find_path_backward_group2(
        &arch_search.neural_network.netdata,
        traj_steps,
        traj_size,
        &w_row_idx_coo,
        &arch_search.neural_network.WColIdx,
        &arch_search.neural_network.neuron_idx,
        w_values_len,
        h_values_len,
        h_values_len,
        h_values_len,
        h_values_len,
        h_values_len,
        h_values_len,
        &mut idxsel_out,
        &mut valsel_out,
        &mut cvec_out,
        &mut dxsel_out,
        &mut nrows_out,
        &mut sparseval_out,
        &mut sparserow_out,
        &mut sparsecol_out,
        &mut hidxsel_out,
        &mut aidxsel_out,
        &mut bidxsel_out,
        &mut cidxsel_out,
        &mut didxsel_out,
        &mut eidxsel_out,
        &mut combidxsel_out,
        &mut daseqs_out,
        &mut dbseqs_out,
        &mut dcseqs_out,
        &mut ddseqs_out,
        &mut deseqs_out,
        &mut wseqs,
        &mut hseqs,
        &mut aseqs,
        &mut bseqs,
        &mut cseqs,
        &mut dseqs,
        &mut eseqs,
    );

    let start_time = std::time::Instant::now();
    let mut loss_val = 0.0f32;

    for epoch in 0..max_epoch {
        let batch_i = (epoch % train_steps) as usize;
        let start = batch_i * batch_size as usize;
        let (x_batch, y_onehot, y_seq) = load_shd_batch(
            &mut shd_train,
            start,
            batch_size,
            input_size,
            traj_size,
            output_size,
        );
        af::set_slices(&mut x_state, &x_batch, 0, (traj_size - 1) as i64);

        raybnn::neural::network_f32::state_space_forward_batch(
            &arch_search.neural_network.netdata,
            &x_state,
            &arch_search.neural_network.WRowIdxCSR,
            &arch_search.neural_network.WColIdx,
            &wseqs,
            &hseqs,
            &aseqs,
            &bseqs,
            &cseqs,
            &dseqs,
            &eseqs,
            &arch_search.neural_network.network_params,
            &mut z,
            &mut q,
        );

        let yhat = select_raybnn_outputs(&q, &output_rows, batch_size, proc_num, q_slices);
        loss_val = raybnn::optimal::loss_f32::RMSE(&yhat, &y_seq);

        raybnn::neural::network_f32::state_space_backward_group2(
            &arch_search.neural_network.netdata,
            &x_state,
            &arch_search.neural_network.network_params,
            &z,
            &q,
            &y_seq,
            raybnn::optimal::loss_f32::MSE_grad,
            &arch_search.neural_network.neuron_idx,
            &idxsel_out,
            &valsel_out,
            &cvec_out,
            &dxsel_out,
            &nrows_out,
            &sparseval_out,
            &sparserow_out,
            &sparsecol_out,
            &hidxsel_out,
            &aidxsel_out,
            &bidxsel_out,
            &cidxsel_out,
            &didxsel_out,
            &eidxsel_out,
            &combidxsel_out,
            &daseqs_out,
            &dbseqs_out,
            &dcseqs_out,
            &ddseqs_out,
            &deseqs_out,
            &mut grad,
        );

        raybnn::optimal::gd_f32::adam(0.9, 0.999, &mut grad, &mut mt, &mut vt);
        let alpha = (learning_rate_scale / (loss_val + 1e-10)).min(0.007);
        arch_search.neural_network.network_params =
            arch_search.neural_network.network_params.clone() + alpha * -grad.clone();

        if epoch % 10 == 0 {
            let (correct, total) = accuracy_from_sequence(&yhat, &y_onehot);
            println!(
                "epoch={} rmse={:.6} train_acc={:.2}% elapsed={:.1}s",
                epoch,
                loss_val,
                100.0 * correct as f32 / total as f32,
                start_time.elapsed().as_secs_f32()
            );
        }
    }

    af::sync(DEVICE);
    println!("final_train_rmse={:.6}", loss_val);

    let mut test_correct = 0u64;
    let mut test_total = 0u64;
    let mut all_true: Vec<f32> = Vec::new();
    let mut all_pred: Vec<f32> = Vec::new();

    for batch_i in 0..test_steps {
        let start = batch_i as usize * batch_size as usize;
        let (x_batch, y_onehot, _y_seq) = load_shd_batch(
            &mut shd_test,
            start,
            batch_size,
            input_size,
            traj_size,
            output_size,
        );
        af::set_slices(&mut x_state, &x_batch, 0, (traj_size - 1) as i64);

        raybnn::neural::network_f32::state_space_forward_batch(
            &arch_search.neural_network.netdata,
            &x_state,
            &arch_search.neural_network.WRowIdxCSR,
            &arch_search.neural_network.WColIdx,
            &wseqs,
            &hseqs,
            &aseqs,
            &bseqs,
            &cseqs,
            &dseqs,
            &eseqs,
            &arch_search.neural_network.network_params,
            &mut z,
            &mut q,
        );

        let yhat = select_raybnn_outputs(&q, &output_rows, batch_size, proc_num, q_slices);
        let (correct, total) = accuracy_from_sequence(&yhat, &y_onehot);
        test_correct += correct;
        test_total += total;
        append_predictions(&yhat, &y_onehot, &mut all_true, &mut all_pred);
    }

    println!(
        "test_acc={:.2}% ({}/{})",
        100.0 * test_correct as f32 / test_total as f32,
        test_correct,
        test_total
    );

    let nsamples = all_pred.len() as u64;
    let true_arr = af::Array::new(&all_true, af::Dim4::new(&[nsamples, 1, 1, 1]));
    let pred_arr = af::Array::new(&all_pred, af::Dim4::new(&[nsamples, 1, 1, 1]));
    let pred_true_arr = af::join(1, &true_arr, &pred_arr);
    raybnn::export::dataloader_f32::write_arr_to_csv(
        "target_vs_prediction_shd_raybnn.csv",
        &pred_true_arr,
    );
}
