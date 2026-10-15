extern crate arrayfire;
extern crate raybnn;
use std::collections::HashMap;
use nohash_hasher;
use rand::Rng;                          
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;
use raybnn::interface::regularization_f32::rcs_regularization_f32 as rcs;
use arrayfire::{af_print, print};
use arrayfire::{Array, Seq, index, index_gen, assign_gen, lookup, matmul, sum, tile, constant, join};
use std::fs::File;
use std::io::{BufWriter, Write};
use rayon::prelude::*;
use raybnn::{physics::update_f32::add_neuron_option_type, optimal::loss_f32::{MSE, MSE_grad}};
use rand::SeedableRng;
use arrayfire as af;
use nohash_hasher::IntMap;
use std::io::{Read, Seek, SeekFrom};

#[derive(Clone, Copy, Debug)]
pub struct ShdMeta { pub n: usize, pub c: usize, pub t: usize }


// read data 
pub fn read_meta(meta_path: &str) -> ShdMeta {
    let s = std::fs::read_to_string(meta_path).expect("read meta failed");
    let p: Vec<_> = s.split_whitespace().collect();
    assert!(p.len() >= 3, "meta must be: N C T");
    ShdMeta { n: p[0].parse().unwrap(), c: p[1].parse().unwrap(), t: p[2].parse().unwrap() }
}

pub struct ShdDat {
    pub meta: ShdMeta,
    xf: File,
    yf: File,
    tmp: Vec<u8>, // one sample: C*T bytes
}
// for reading ShdDataset 
impl ShdDat {
    pub fn open(meta_path: &str, x_path: &str, y_path: &str) -> Self {
        let meta = read_meta(meta_path);
        let xf = File::open(x_path).expect("open X dat failed");
        let yf = File::open(y_path).expect("open y dat failed");
        let tmp = vec![0u8; meta.c * meta.t];
        Self { meta, xf, yf, tmp }
    }

    fn read_label(&mut self, n: usize) -> i64 {
        let off = (n as u64) * 8u64;
        self.yf.seek(SeekFrom::Start(off)).expect("seek y failed");
        let mut buf = [0u8; 8];
        self.yf.read_exact(&mut buf).expect("read y failed");
        i64::from_le_bytes(buf)
    }

    fn read_sample_u8(&mut self, n: usize) {
        let ct = self.meta.c * self.meta.t;
        let off = (n as u64) * (ct as u64);
        self.xf.seek(SeekFrom::Start(off)).expect("seek X failed");
        self.xf.read_exact(&mut self.tmp).expect("read X failed");
    }

    /// Return X batch in [C,B,T] order as f32, plus label indices (B,)
    pub fn load_batch_cbt_f32(&mut self, start: usize, b: usize) -> (Vec<f32>, Vec<i64>) {
        let c = self.meta.c;
        let t = self.meta.t;
        assert!(start + b <= self.meta.n);

        let mut x_batch = vec![0f32; c * b * t]; // [C,B,T]
        let mut y_batch = vec![0i64; b];

        for bi in 0..b {
            let n = start + bi;
            self.read_sample_u8(n);
            y_batch[bi] = self.read_label(n);

            for cc in 0..c {
                let src_base = cc * t;
                for tt in 0..t {
                    let src_idx = src_base + tt;                 // tmp[cc,tt]
                    let dst_idx = cc + c * bi + (c * b) * tt;    // [C,B,T]
                    x_batch[dst_idx] = self.tmp[src_idx] as f32;
                }
            }
        }
        (x_batch, y_batch)
    }
}

// transform labels to one hot version
pub fn labels_to_onehot_f32(y: &[i64], num_classes: usize) -> Vec<f32> {
    let b = y.len();
    let mut out = vec![0f32; num_classes * b]; // [out,B]
    for (bi, &lab) in y.iter().enumerate() {
        let k = lab as usize;
        assert!(k < num_classes);
        out[k + num_classes * bi] = 1.0;
    }
    out
}

fn time_slice_stats(tag: &str, a: &af::Array<f32>, t0: i64, t1: i64) {
    use arrayfire as af;
    let dims = a.dims();
    let out = dims[0] as i64;
    let b   = dims[1] as i64;

    let rows  = af::Seq::new(0.0, (out-1) as f64, 1.0);
    let batch = af::Seq::new(0.0, (b-1) as f64, 1.0);
    let time  = af::Seq::new(t0 as f64, t1 as f64, 1.0);

    let mut idx = af::Indexer::default();
    idx.set_index(&rows, 0, None);
    idx.set_index(&batch, 1, None);
    idx.set_index(&time, 2, None);

    let s = af::index_gen(a, idx);
    norm_stats_f32(tag, &s);
}


fn norm_stats_f32(tag: &str, a: &af::Array<f32>) {
    use arrayfire as af;
    let (mn, _) = af::min_all(a);
    let (mx, _) = af::max_all(a);
    let (mean, _) = af::mean_all(a);

    let abs_a = af::abs(a);
    let l1 = af::sum_all(&abs_a).0;
    let abs_aa: Array<f32> = abs_a.clone();
        let abs_aaa: Array<f32> = abs_a.clone();

    let l2 = af::sum_all(&(abs_a * abs_aa)).0.sqrt();

    // nnz (approx): count(abs(a) > 0)
    let nz_mask = af::gt(&abs_aaa, &af::constant(0.0f32, a.dims()), false);
    let nnz = af::sum_all(&nz_mask.cast::<f32>()).0 as u64;

    println!(
        "[{}] min={:.6e} max={:.6e} mean={:.6e} L1={:.6e} L2={:.6e} nnz={} dims={:?}",
        tag, mn as f32, mx as f32, mean as f32, l1 as f32, l2 as f32, nnz, a.dims()
    );
}
// Compute confusion matrix and accuracy for one batch.
// target_1hot: one-hot ground-truth labels.
// pred_rate : predictions 
// returns confusion matrix and accuracy
fn batch_confusion_and_accuracy(
    pred_rate: &af::Array<f32>,   // [out, B, 1, 1]
    target_1hot: &af::Array<f32>, // [out, B, 1, 1]
) -> (Vec<u32>, u64, u64) {
    // Argmax over outputs (dim=0) → predicted/true class indices (u32)
    let (_pv, pred_idx)  = af::imax(pred_rate, 0);
    let (_tv, true_idx)  = af::imax(target_1hot, 0);

    let mut p = vec![0u32; pred_idx.elements()];
    let mut t = vec![0u32; true_idx.elements()];
    pred_idx.host(&mut p);
    true_idx.host(&mut t);

    let c = pred_rate.dims()[0] as usize;  // num classes
    let mut cm = vec![0u32; c * c];
    let mut correct: u64 = 0;
    let total:   u64 = p.len() as u64;

    for k in 0..p.len() {
        let ti = t[k] as usize; // true
        let pi = p[k] as usize; // pred
        cm[ti * c + pi] += 1;
        if ti == pi { correct += 1; }
    }
    (cm, correct, total)
}


fn stats_f32(tag: &str, a: &arrayfire::Array<f32>) {
    use arrayfire as af;

    let dims = a.dims();
    let (mn, _) = af::min_all(a);
    let (mx, _) = af::max_all(a);
    let (mean, _) = af::mean_all(a);

    // NaN count: isnan() if available; fallback: a != a
    let nan_mask = af::neq(a, a, false); // true where NaN
    let nan_count = af::sum_all(&nan_mask).0 as u64;

    println!(
        "[{}] dims={:?} min={:.6} max={:.6} mean={:.6} nan={}",
        tag, dims, mn as f32, mx as f32, mean as f32, nan_count
    );
}






    fn main() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(10);

	//Set CUDA and GPU Device 0
	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);

	arrayfire::set_seed(10  as u64);


	
        // Architecture knobs (starter)
let conn_vec: Vec<u64>        = vec![50];     // ok to keep
let neuron_size_vec: Vec<u64> = vec![1200]; // ok to keep, can try 200–600 later
	let dir_path = "/tmp/".to_string() ;

// Input dims
let arg_input_vec: Vec<u64> = vec![700];


let batch_size: u64 = 250;      // batch size for each mini batch
let traj_size: u64  = 700;      // SHD T
let T_repeat: u64   = 1;        //SHD already has time bins, so 1
let time_steps_total: u64 = traj_size * T_repeat; // this is the length of the input time series dataset transformed into spikes
let max_input_size: u64 = 700;
let mut input_size: u64 = 700;

// Output ma 
let max_output_size: u64 = 20;
let output_size: u64 = 20;

// Capacity buffer
    let max_neuron_size = input_size + neuron_size_vec[0] + output_size; // 

// Time-series packing

// Multi-proc propagation
let mut proc_num: u64 = 25;        // keep as in figure-4 code

// Dataset sizes
let train_size: u64 = 8156;
let test_size: u64  = 2264;
	let temp_dims: arrayfire::Dim4 = arrayfire::Dim4::new(&[1,1,1,1]);

	//Create Start Neural Network
	println!("Creating Network");

	let mut arch_search = raybnn::interface::automatic_f32::create_start_archtecture(

		input_size,
		max_input_size,

		output_size,
		max_output_size,

		max_neuron_size,


		batch_size,
		traj_size,

	
		&dir_path
	);


	arch_search.neural_network.netdata.proc_num = proc_num;
	
let base = "/home/yilin/Downloads/TwoPatterns/shd_dat";

let train_meta = format!("{}/shd_train_dt2ms_T700.meta.txt", base);
let train_x    = format!("{}/shd_train_dt2ms_T700_X_u8.dat", base);
let train_y    = format!("{}/shd_train_dt2ms_T700_y_i64.dat", base);
let mut shd_train = ShdDat::open(&train_meta, &train_x, &train_y);

// test
let test_meta = format!("{}/shd_test_dt2ms_T700.meta.txt", base);
let test_x    = format!("{}/shd_test_dt2ms_T700_X_u8.dat", base);
let test_y    = format!("{}/shd_test_dt2ms_T700_y_i64.dat", base);
let mut shd_test = ShdDat::open(&test_meta, &test_x, &test_y);

assert_eq!(shd_train.meta.c, 700);
assert_eq!(shd_train.meta.t, 700);
assert_eq!(shd_test.meta.c, 700);
assert_eq!(shd_test.meta.t, 700);

let steps_per_epoch:  u64 = (shd_train.meta.n as u64 / batch_size).max(1);
let steps_per_epoch2: u64 = (shd_test.meta.n as u64  / batch_size).max(1);
let train_size: u64 = shd_train.meta.n as u64;
let test_size:  u64 = shd_test.meta.n as u64;
	//Load dataset
	println!("Load Dataset");
    	arrayfire::sync(DEVICE);

	    let mut prev_target_neuron_size = 0;


	



	for ii in 0..arg_input_vec.len()
	{

let mut acc100_streak: u32 = 0;



		arch_search.neural_network.neuron_pos = arrayfire::rows(&arch_search.neural_network.neuron_pos, input_size as i64, (arch_search.neural_network.neuron_pos.dims()[0] - 1) as i64 );

		input_size = arg_input_vec[ii].clone();
		arch_search.neural_network.netdata.input_size =  input_size;

		let mut WRowIdxCOO = raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);

		arch_search.neural_network.netdata.neuron_std = 0.001;
		arch_search.neural_network.netdata.con_rad = (arch_search.neural_network.netdata.sphere_rad/(proc_num as f32))*2.0;
		let input_neurons = raybnn::physics::initial_f32::create_spaced_input_neuron_on_sphere_1D(
			arch_search.neural_network.netdata.sphere_rad+0.2,
			input_size
		);
		arch_search.neural_network.neuron_pos = arrayfire::join(0, &input_neurons, &arch_search.neural_network.neuron_pos);
		arch_search.neural_network.netdata.active_size = arch_search.neural_network.neuron_pos.dims()[0];


		raybnn::physics::initial_f32::assign_neuron_idx_with_buffer(
			max_input_size,
			max_output_size,
			&arch_search.neural_network.netdata,
			&arch_search.neural_network.neuron_pos,
			&mut arch_search.neural_network.neuron_idx,
		);
		if (prev_target_neuron_size !=  neuron_size_vec[ii])
		{

			let new_neuron_num =  neuron_size_vec[ii] - (arch_search.neural_network.neuron_idx.dims()[0] - input_size - output_size);
			println!("Iteration {}, new_neuron_num: {}", ii, new_neuron_num);

			let d_pos = arch_search.neural_network.neuron_pos.dims();
			let d_idx = arch_search.neural_network.neuron_idx.dims();
			println!("neuron_pos dims[0] (active) = {}", d_pos[0]);
			println!("neuron_idx dims[0] (active) = {}", d_idx[0]);
			println!("current hidden = {}", d_idx[0] - input_size - output_size);

			prev_target_neuron_size = neuron_size_vec[ii];

			let add_neuron_options: add_neuron_option_type = add_neuron_option_type {
				new_active_size: new_neuron_num,
				init_connection_num: conn_vec[ii as usize].clone(),
				input_neuron_con_rad: 60.0*arch_search.neural_network.netdata.neuron_rad,
				hidden_neuron_con_rad: 110.0*arch_search.neural_network.netdata.neuron_rad,
				output_neuron_con_rad: 60.0*arch_search.neural_network.netdata.neuron_rad,
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
	
		let mut active_size = arch_search.neural_network.neuron_idx.dims()[0];
		arch_search.neural_network.netdata.active_size = active_size;

		WRowIdxCOO = raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);

		arch_search.neural_network.WRowIdxCSR = raybnn::graph::large_sparse_i32::COO_to_CSR(&WRowIdxCOO,arch_search.neural_network.netdata.neuron_size);


		
	//Create Start Neural Network




		arrayfire::device_gc();
		arrayfire::sync(DEVICE);
		


		let T: u64 = 1; // this sets the time steps for SNN spikes 
		let time_steps_total: u64 = traj_size * T;  // total SNN steps per trajectory
		let Z_dims = arrayfire::Dim4::new(&[arch_search.neural_network.netdata.neuron_size,batch_size*1,time_steps_total,1]);
		// [arch_search.neural_network.netdata.neuron_size,batch_size,1,1]
		let mut Z = arrayfire::constant::<f32>(0.0,Z_dims);  // membrane potential
		let mut Q = arrayfire::constant::<f32>(0.0,Z_dims);  // spikes



		let mut alpha0: f32 = 0.002;

		let mut active_size = arch_search.neural_network.neuron_idx.dims()[0];
		let idxsel = arrayfire::rows(&arch_search.neural_network.neuron_idx, (active_size-output_size)  as i64, (active_size-1)  as i64);
		let Qslices: u64 = Q.dims()[2];

		let mut WRowIdxCOO = raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);


		let total_param_size = arch_search.neural_network.network_params.dims()[0];
		
		let mt_dims = arrayfire::Dim4::new(&[total_param_size,1,1,1]);
		let mut mt = arrayfire::constant::<f32>(0.0,mt_dims);
		let mut vt = arrayfire::constant::<f32>(0.0,mt_dims);
		let mut grad = arrayfire::constant::<f32>(0.0,mt_dims);


		let mut Wseqs: [arrayfire::Seq<i32>; 1] = [arrayfire::Seq::<i32>::default()];
		let mut Hseqs: [arrayfire::Seq<i32>; 1] = [arrayfire::Seq::<i32>::default()];
		


		let X_dims = arrayfire::Dim4::new(&[input_size,batch_size*1,time_steps_total,1]);
		let mut X = arrayfire::constant::<f32>(0.0,X_dims);

		let mut idxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
		let mut valsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();

		let mut cvec_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
		let mut dXsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();

		let mut nrows_out: nohash_hasher::IntMap<i64, u64> = nohash_hasher::IntMap::default();
		let mut sparseval_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
		let mut sparsecol_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
		let mut sparserow_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();

		let mut Hidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
		

		let WValuesdims0 =  arch_search.neural_network.WColIdx.dims()[0];

		let network_paramsdims0 =  arch_search.neural_network.network_params.dims()[0];
	
		let Hdims0 =  (network_paramsdims0 -  WValuesdims0)/6; 

		// find the path for the neurons 
		raybnn::neural::network_snn_f32::find_path_backward_snn(
			&arch_search.neural_network.netdata,
			time_steps_total,
			1,
			&WRowIdxCOO,
			&arch_search.neural_network.WColIdx,
			&arch_search.neural_network.neuron_idx,
			
				WValuesdims0,
				
			Hdims0,
			&mut Hidxsel_out,
			&mut idxsel_out,
			&mut valsel_out,

			&mut cvec_out,
			&mut dXsel_out,

			&mut nrows_out,
			&mut sparseval_out,
			&mut sparserow_out,
			&mut sparsecol_out,
			
			
			&mut Wseqs,
			&mut Hseqs,

		);





		// trajectories in X: [input, batch, traj_size, 1]
let train_X_dims = arrayfire::Dim4::new(&[input_size, batch_size, traj_size, 1]);
let mut train_X = arrayfire::constant::<f32>(0.0, train_X_dims);
let Y_dims: arrayfire::Dim4 = arrayfire::Dim4::new(&[output_size, batch_size, 1, 1]);
let mut Y: Array<f32> = arrayfire::constant::<f32>(0.0, Y_dims);

let mut batch_idx = 0;






		let mut max_epoch = 1200;


		let mut loss_val = 100000.0;
		let mut loss_val2 = 100000.0;
		let mut prev_loss = 100000.0;

		arrayfire::device_gc();
		arrayfire::sync(DEVICE);

		let start_time = std::time::Instant::now();
		let out_cols = arrayfire::Dim4::new(&[1, 3, 1, 1]); // columns: pred,label,rate
		let mut pred_true_rate = af::constant::<f32>(0.0, out_cols);   


let bsz = batch_size as usize;
		for i in 0..max_epoch
		{
            
    let bi = (i as u64 % steps_per_epoch) as usize;
        let start = bi * bsz;
            let (x_cbt, y_idx) = shd_train.load_batch_cbt_f32(start, bsz);
let X_in = af::Array::new(
        &x_cbt,
        af::Dim4::new(&[input_size, batch_size, traj_size, 1])
    );
let X_bin = af::gt(&X_in, &af::constant(0.0f32, X_in.dims()), false).cast::<f32>();
    // 3) One-hot labels: [out, B, 1, 1]
    let y_onehot = labels_to_onehot_f32(&y_idx, output_size as usize);
    let Y_scalar = af::Array::new(
        &y_onehot,
        af::Dim4::new(&[output_size, batch_size, 1, 1])
    );

    // (optional sanity print once)
    if i == 0 {
        norm_stats_f32("[SHD] X_in", &X_in);
        norm_stats_f32("[SHD] Y_scalar", &Y_scalar);
    }
	if i == 0 {
    let miny = *y_idx.iter().min().unwrap();
    let maxy = *y_idx.iter().max().unwrap();
    println!("[labels] min={} max={}", miny, maxy);
}
af::set_seed((1234) as u64); 
let in_dim    = train_X.dims()[0];
let batch_dim = train_X.dims()[1];
let it_dim    = train_X.dims()[2];
let t_dim     = T as u64;


	



println!("train_X {:.6}", af::sum_all(&X_in).0 / X_in.elements() as f32);

let y_scalar_dims = af::Dim4::new(&[output_size, batch_size, 1, 1]);
let Y_seq = af::tile(&Y_scalar, af::Dim4::new(&[1, 1, traj_size, 1]));
    if i == 0 {
        let mean = af::mean_all(&X_in).0;
        let maxv = af::max_all(&X_in).0;
        println!("[SHD sanity] X mean={:.6} max={:.3}", mean, maxv);
    }

let (_tv, true_idx) = af::imax(&Y_scalar, 0);


			raybnn::neural::network_snn_f32::state_space_forward_batch_snn(
				&arch_search.neural_network.netdata,
				&X_bin,
			
				&arch_search.neural_network.WRowIdxCSR,
				&arch_search.neural_network.WColIdx,
			
				&Wseqs,
				&Hseqs,
				&arch_search.neural_network.network_params,
				time_steps_total,
				&mut Z,
				&mut Q,

			);

			// select output neurons
let neuron_size = Q.dims()[0] as i64;  // 
let out_size    = output_size as i64;  
let out_start   = neuron_size - out_size;
let out_end     = neuron_size - 1;

let rows  = arrayfire::Seq::new(out_start as f32, out_end as f32, 1.0);
let batch = arrayfire::Seq::new(0.0, (Q.dims()[1]-1) as f32, 1.0);
let time  = arrayfire::Seq::new((0) as f32, (Q.dims()[2]-1) as f32, 1.0);

let mut idxrs = arrayfire::Indexer::default();
idxrs.set_index(&rows,  0, None);
idxrs.set_index(&batch, 1, None);
idxrs.set_index(&time,  2, None);

let Yhat: arrayfire::Array<f32> = arrayfire::index_gen(&Q, idxrs);
let TT: f32 = Yhat.dims()[2] as f32;
let spike_counts = af::sum(&Yhat, 2);             // sum of spikes over time
let pred_rate    = spike_counts / TT;              // firing rate = spikes per time step

let target_rate = af::max(&Y, 2);   
// af_print!("{}",Y);
// break;;
// true class index per sample: [1, B, 1, 1] as u32
let (_tv, true_idx) = af::imax(&pred_rate, 0);
let pr_mean = arrayfire::mean(&pred_rate, 1);
af_print!("pred_rate mean per class", pr_mean);


loss_val = raybnn::optimal::snnloss_f32::mse_spike_rate_seq(&Yhat, &Y_seq,T);
println!("loss_val {}, i {}",loss_val,i);
	let (cm_train_vec, correct_train, total_train) = batch_confusion_and_accuracy(&pred_rate, &Y_scalar);
	let acc_train = correct_train as f32 / (total_train as f32);

	// // Pretty-print confusion matrix
	let c = output_size as u64;
	let cm_train_arr = af::Array::new(&cm_train_vec, af::Dim4::new(&[c, c, 1, 1]));
	
// debug prints
// 	if (i%20 == 0){
// af_print!("Train Confusion (rows=true, cols=pred)", cm_train_arr);
// 	println!("Epoch {}: loss = {:.6}, train acc = {:.2}%", i, loss_val, acc_train * 100.0);
// 	let (total_spikes, _) = af::sum_all::<f32>(&Q);
// let total_entries = Q.elements() as f32;
// let firing_rate = total_spikes / total_entries;
// println!("Global firing rate = {}", firing_rate);
// println!("Global sparsity = {}", 1.0 - firing_rate);
// let num_edges = arch_search.neural_network.WColIdx.dims()[0];
// println!("Number of nonzero edges = {}", num_edges);

// 	}

if (i >= 250){
if (acc_train >= 0.99 ) {
    acc100_streak += 1;
} else {
    acc100_streak = 0;
}

if acc100_streak >= 2 {
    println!("Early stop: train accuracy hit 100% for 2 consecutive epochs at epoch {}", i);
    break;
}
}

if i == 0
			{
				loss_val2 = loss_val;
			}
			else
			{
				loss_val2 = 0.99*loss_val2 + 0.01*loss_val;
			}

			raybnn::neural::network_snn_f32::state_space_backward_batch(
				&arch_search.neural_network.netdata,
				&X_bin,
				&arch_search.neural_network.network_params,
				&Z,
				&Q,
				&Y_scalar,
				raybnn::optimal::snnloss_f32::ce_spike_rate_global_grad,
				&arch_search.neural_network.neuron_idx,
		
				&idxsel_out,
				&valsel_out,

				&cvec_out,
				&dXsel_out,

				&nrows_out,
				&sparseval_out,
				&sparserow_out,
				&sparsecol_out,

			
			
				&Hidxsel_out,
				&mut grad,
				T
			);
    norm_stats_f32("grad", &grad);
let Wdims0 = WValuesdims0 as i64;          // number of W params
let Hdims0 = (arch_search.neural_network.netdata.neuron_size as i64); // number of H params (per your find_path)

let gW = af::rows(&grad, 0, Wdims0 - 1);
let gH = af::rows(&grad, Wdims0, Wdims0 + Hdims0 - 1);

// debugging print
// println!("||gW||2 = {:.6e}", af::norm(&gW, af::NormType::VECTOR_2, 1.0, 1.0));
// println!("||gH||2 = {:.6e}", af::norm(&gH, af::NormType::VECTOR_2, 1.0, 1.0));


			raybnn::optimal::gd_f32::adam(
				0.9
				,0.999
				,&mut grad
				,&mut mt
				,&mut vt
			);
			// ---- grad stats ----

		
			// alpha0 = (0.002/(loss_val  +  1e-10)).min(0.007);

// 1) compute grad with backward
// 2) optionally run adam(...) to transform grad
// 3) update params
arch_search.neural_network.network_params =
    arch_search.neural_network.network_params.clone() + (alpha0 * -grad.clone());

// println!("W max = {:.6e}", af::max_all(&arch_search.neural_network.network_params).0);
// println!("W L2  = {:.6e}", af::norm(&arch_search.neural_network.network_params, af::NormType::VECTOR_2, 1.0, 1.0));






			// arch_search.neural_network.network_params = arch_search.neural_network.network_params + (alpha0*-grad.clone());
// 			// if i % 50 == 0 {
// 			// 	let elapsed_time = start_time.elapsed().as_secs_f32();
// 			// 	println!("Epoch {}: Test RMSE = {}, Time elapsed = {}s", i, loss_val, elapsed_time);
// 			// }


			// if (i > 300) && ((i % 50) == 0)
			// {
			// 	if (prev_loss-loss_val).abs()/loss_val < 0.007
			// 	{
			// 		break;
			// 	}

			// 	prev_loss = loss_val;
			// }
	


		}

		
		arrayfire::sync(DEVICE);
			let Z_dims = arrayfire::Dim4::new(&[arch_search.neural_network.netdata.neuron_size,batch_size,time_steps_total,1]);
		let mut Z = arrayfire::constant::<f32>(0.0,Z_dims);
		let mut Q = arrayfire::constant::<f32>(0.0,Z_dims);



		let Qslices: u64 = Q.dims()[2];


		let X_dims = arrayfire::Dim4::new(&[input_size,batch_size,traj_size,1]);
		let mut X = arrayfire::constant::<f32>(0.0,X_dims);



		let test_X_dims = arrayfire::Dim4::new(&[input_size,batch_size,traj_size,1]);
		let mut test_X =  arrayfire::constant::<f32>(0.0,temp_dims);
		let Y_dims = arrayfire::Dim4::new(&[output_size,batch_size,1,1]);
		let mut Y =  arrayfire::constant::<f32>(0.0,temp_dims);
		let mut batch_idx = 0;
		let epoch_num = (test_size/ (batch_size) );
		let temp_Y_dims = arrayfire::Dim4::new(&[1,output_size,1,1]);
		let mut Yhat_arr = arrayfire::constant::<f32>(0.0,temp_Y_dims);
		let mut Y_arr = arrayfire::constant::<f32>(0.0,temp_Y_dims);
		let newdims = arrayfire::Dim4::new(&[output_size,batch_size,traj_size,1]);
		let c = output_size as usize;
		let mut test_cm: Vec<u32> = vec![0; c * c];
		let mut test_correct_sum: u64 = 0;
		let mut test_total_sum:   u64 = 0;
		let mut all_preds: Vec<f32>  = Vec::new(); // predicted class index
		let mut all_trues: Vec<f32>  = Vec::new(); // true class index

let bsz = batch_size as usize;
for batch_idx in 0..(steps_per_epoch2)
		{
            let batch_idx_usize: usize = batch_idx as usize;
let start: usize = batch_idx_usize * bsz;
let (x_cbt, y_idx) = shd_test.load_batch_cbt_f32(start, bsz);
      let y_onehot = labels_to_onehot_f32(&y_idx, output_size as usize);
    let Y: Array<f32> = af::Array::new(&y_onehot, af::Dim4::new(&[output_size, batch_size, 1, 1]));

let X_in_test = af::Array::new(&x_cbt, af::Dim4::new(&[input_size, batch_size, traj_size, 1]));
let X_bin_test = af::gt(&X_in_test, &af::constant(0.0f32, X_in_test.dims()), false).cast::<f32>();

			
af::set_seed((1234) as u64); // or constant 1234 for identical spikes every epoch



			raybnn::neural::network_snn_f32::state_space_forward_batch_snn(
				&arch_search.neural_network.netdata,
				&X_bin_test,
				
				&arch_search.neural_network.WRowIdxCSR,
				&arch_search.neural_network.WColIdx,
			
				&Wseqs,
				&Hseqs,
				&arch_search.neural_network.network_params,
				time_steps_total,
				&mut Z,
				&mut Q,
									// Some(&mut trace)

			);
			

			
			



			// 	//Get Yhat
			let mut idxrs = arrayfire::Indexer::default();
			let seq1 = arrayfire::Seq::new(0.0f32, (batch_size-1) as f32, 1.0);
			let seq2 = arrayfire::Seq::new((0) as f32, (Qslices-1) as f32, 1.0);
			idxrs.set_index(&idxsel, 0, None);
			idxrs.set_index(&seq1, 1, None);
			idxrs.set_index(&seq2, 2, None);
			let mut Yhat = arrayfire::index_gen(&Q, idxrs);
			let TT: f32 = Yhat.dims()[2] as f32;
			let spike_counts = af::sum(&Yhat, 2);             // sum of spikes over time
			let pred_rate    = spike_counts / TT;              // firing rate = spikes per time step

			let (cm_vec, correct, total) = batch_confusion_and_accuracy(&pred_rate, &Y);
			test_correct_sum += correct;
			test_total_sum   += total;
			
			// accumulate confusion
			for i in 0..(c*c) { test_cm[i] += cm_vec[i]; }
			let (_pv, pred_idx)  = af::imax(&pred_rate, 0);
let (_tv, true_idx)  = af::imax(&Y, 0);

let mut p_host = vec![0u32; pred_idx.elements()];
let mut t_host = vec![0u32; true_idx.elements()];
pred_idx.host(&mut p_host);
true_idx.host(&mut t_host);
let test_acc = test_correct_sum as f32 / (test_total_sum as f32);
				let cm_test_arr = af::Array::new(&test_cm, af::Dim4::new(&[output_size, output_size, 1, 1]));
				af_print!("Test Confusion (rows=true, cols=pred)", cm_test_arr);
				println!("Mean test accuracy = {:.2}%", test_acc * 100.0);

// append to big vectors
for k in 0..p_host.len() {
    all_preds.push(p_host[k] as f32); // store as f32 for ArrayFire CSV writer
    all_trues.push(t_host[k] as f32);
}
			
		}

// Convert collected predictions/targets to ArrayFire arrays
let nsamples = all_preds.len() as u64;

// shape: [nsamples, 1, 1, 1]
let pred_arr  = af::Array::new(&all_preds, af::Dim4::new(&[nsamples, 1, 1, 1]));
let true_arr  = af::Array::new(&all_trues, af::Dim4::new(&[nsamples, 1, 1, 1]));

// [nsamples, 2, 1, 1] – column 0 = true, column 1 = pred
let pred_true_arr = af::join(1, &true_arr, &pred_arr);

// Save to CSV
let save_filename = format!("target_vs_prediction{}.csv", input_size);
raybnn::export::dataloader_f32::write_arr_to_csv(
    &save_filename,
    &pred_true_arr
);
println!("saved predictions/targets to {}", save_filename);

	}
}