use crate::graph::adjacency_f32::get_global_weight_idx;
use crate::graph::adjacency_f32::get_global_weight_idx2;
use crate::graph::large_sparse_i32::remap_rows;
use crate::graph::large_sparse_i32::COO_batch_find;
use crate::graph::large_sparse_u64::COO_batch_find as COO_batch_find_u64;
use crate::graph::tree_i32::find_unique;
use crate::interface::regularization_f32::rcs_regularization_f32 as rcs;
use crate::neural::network_f32::network_metadata_type as NetworkMetadata;
use crate::neural::network_f32::network_metadata_type;
use arrayfire::Dim4;
use arrayfire::{self as af, af_print};
use arrayfire::{
    assign_gen, constant, index, index_gen, join, lookup, matmul, sum, tile, Array, Seq,
};
use nohash_hasher::IntMap;
const COO_find_limit: u64 = 1500000000;
use std::collections::BTreeSet;

pub fn find_path_backward_snn(
    netdata: &network_metadata_type,
    Xslices: u64,
    Yslices: u64,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
    neuron_idx: &arrayfire::Array<i32>, // global neuron indices
    Wdims0: u64,
    Hdims0: u64,
    Hidxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    // Output maps (only W-related)
    idxsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    valsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    cvec_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    dXsel_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    nrows_out: &mut nohash_hasher::IntMap<i64, u64>,
    sparseval_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    sparserow_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    sparsecol_out: &mut nohash_hasher::IntMap<i64, arrayfire::Array<i32>>,
    // Sequence for W (index range of W in a global parameter vector, if needed)
    Wseqs: &mut [arrayfire::Seq<i32>; 1],
    Hseqs: &mut [arrayfire::Seq<i32>; 1],
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();
    let batch_size: u64 = netdata.batch_size.clone();

    let COO_batch_size = 1 + ((COO_find_limit / WRowIdxCOO.dims()[0]) as u64);
    let total_neurons = neuron_idx.dims()[0];
    let start_idx = (total_neurons - output_size) as i64;
    let end_idx = (total_neurons - 1) as i64;
    let mut idxsel = arrayfire::rows(neuron_idx, start_idx, end_idx);
    let output_idxsel: Array<i32> = idxsel.clone(); // keep a copy of output neuron indices
    let scalar_dim = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let mut valsel = arrayfire::constant::<i32>(0, scalar_dim);
    let mut rvec: Array<i32> = arrayfire::constant::<i32>(0, scalar_dim);
    let mut cvec = arrayfire::constant::<i32>(0, scalar_dim);
    let mut dXsel = arrayfire::constant::<i32>(0, scalar_dim);
    let mut sparseval = arrayfire::constant::<i32>(0, scalar_dim);
    let mut sparsecol = arrayfire::constant::<i32>(0, scalar_dim);
    let mut sparserow = arrayfire::constant::<i32>(0, scalar_dim);
    let mut yslicidx = (Yslices - 1) as i64;
    let mut new_idxsel = arrayfire::rows(
        neuron_idx,
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );
    let Hoffset = Wdims0 as i32;
    let mut Hidxsel = idxsel.clone();

    for t in (0i64..(Xslices as i64)).rev() {
        idxsel = new_idxsel.clone();

        idxsel_out.insert(t, idxsel.clone());
        // if (t == t){
        //             println!("t={}", t);
        // af_print!("idxsel", idxsel);
        // }

        Hidxsel = Hoffset + idxsel.clone();
        Hidxsel_out.insert(t, Hidxsel.clone());

        valsel = COO_batch_find(WRowIdxCOO, &idxsel, COO_batch_size);
        valsel_out.insert(t, valsel.clone());

        rvec = arrayfire::lookup(WRowIdxCOO, &valsel, 0); // target neuron indices
        cvec = arrayfire::lookup(WColIdx, &valsel, 0); // source neuron indices
        cvec_out.insert(t, cvec.clone());

        dXsel = remap_rows(&rvec, &idxsel, neuron_size);
        dXsel_out.insert(t, dXsel.clone());

        let global_idx = get_global_weight_idx2(neuron_size, &rvec, &cvec);
        let (_, idx_order) = arrayfire::sort_index(&global_idx, 0, true);

        sparseval = arrayfire::lookup(&valsel, &idx_order, 0); //
        sparsecol = arrayfire::lookup(&rvec, &idx_order, 0); //

        let mut idx_map =
            arrayfire::constant::<i32>(0, arrayfire::Dim4::new(&[neuron_size, 1, 1, 1]));
        let count_indices =
            arrayfire::iota::<i32>(idxsel.dims(), arrayfire::Dim4::new(&[1, 1, 1, 1]));
        let idxsel_view: Array<i32> = idxsel.clone();
        let mut indexer = arrayfire::Indexer::default();
        indexer.set_index(&idxsel_view, 0, None);
        arrayfire::assign_gen(&mut idx_map, &indexer, &count_indices);
        sparsecol = arrayfire::lookup(&idx_map, &sparsecol, 0);

        sparserow = arrayfire::lookup(&cvec, &idx_order, 0);
        let ones: Array<i32> = arrayfire::constant::<i32>(1, sparserow.dims());
        let (_, counts_per_source) = arrayfire::sum_by_key(&sparserow, &ones, 0);
        let num_sources = counts_per_source.dims()[0].clone();
        nrows_out.insert(t, counts_per_source.dims()[0].clone() as u64);
        //   if (t == t){
        //             println!("t={}", t);
        // af_print!("sparserow", sparserow);

        // }
        // println!("t:{}num_sources:{}",t,num_sources);
        let mut row_ptr = arrayfire::accum(&counts_per_source, 0);
        let zero = arrayfire::constant::<i32>(0, arrayfire::Dim4::new(&[1, 1, 1, 1]));
        row_ptr = arrayfire::join(0, &zero, &row_ptr);
        sparserow = row_ptr.clone();
        sparseval_out.insert(t, sparseval.clone());
        sparserow_out.insert(t, sparserow.clone());
        sparsecol_out.insert(t, sparsecol.clone());

        // --------------------------------------------------------------------------------------------------------------------------

        let ones = af::constant::<i32>(1, sparserow.dims());
        let (_k, counts_per_source) = af::sum_by_key(&sparserow, &ones, 0);

        let num_sources = counts_per_source.dims()[0] as u64;

        let mut row_ptr = af::accum(&counts_per_source, 0);
        let zero = af::constant::<i32>(0, af::Dim4::new(&[1, 1, 1, 1]));
        row_ptr = af::join(0, &zero, &row_ptr);

        // row_ptr is [num_sources+1, 1, 1, 1]
        let rowptr_len = row_ptr.dims()[0] as i64;

        // select last element (same start/end)
        let last_seq = af::Seq::new((rowptr_len - 1) as f32, (rowptr_len - 1) as f32, 1.0);

        // NOTE: af::index expects &[Seq<_>], so wrap in a slice
        let rowptr_last_arr = af::index(&row_ptr, &[last_seq]);

        let mut last_host = vec![0i32; 1];
        rowptr_last_arr.host(&mut last_host);

        // println!("rowptr_last = {}", last_host[0]);
        // if (t == 2999 || t == 2998){
        // println!(
        //   "t={} idxsel_len={} cvec_len={} num_sources={} rowptr_len={} rowptr_last={} nnz_edges={}",
        //   t,
        //   idxsel.dims()[0],
        //   cvec.dims()[0],
        //   num_sources,
        //   rowptr_len,
        //   last_host[0],
        //   sparserow.elements()
        // );
        // }
        new_idxsel = find_unique(&cvec, neuron_size);

        // if (yslicidx > 0) {
        //     yslicidx -= 1;
        new_idxsel = arrayfire::join(0, &new_idxsel, &output_idxsel);
        // }else{
        // }
        // new_idxsel = arrayfire::join(0, &new_idxsel, &output_idxsel);
    }

    let w_end_index = Wdims0 as i32 - 1;
    *Wseqs = [af::Seq::new(0, (Wdims0 - 1) as i32, 1)];
    let Hstart: i32 = w_end_index + 1;
    let Hend = Hstart + (netdata.neuron_size as i32) - 1; // <-- use neuron_size
    *Hseqs = [af::Seq::new(Hstart, Hend, 1)];
    // println!("idxsel_out{}",idxsel_out.len());
    // println!("nrows_out{}",nrows_out.len());
    // compare_nrows_vs_rowptr(nrows_out,idxsel_out);
}

//Hard reset, the membrane potential will be set to the reset voltage after firing
// Forward pass through spiking neural network (LIF).
pub fn state_space_forward_batch_snn(
    netdata: &NetworkMetadata,       // Neural Network Metadata
    X: &af::Array<f32>,              // input matrix
    WRowIdxCSR: &af::Array<i32>,     // Row sparse matrix of the weighted adjacency matrix
    WColIdx: &af::Array<i32>,        // Column sparse matrix of the weighted adjacency matrix
    Wseqs: &[af::Seq<i32>; 1],       // Indecies of the weight parameters
    Hseqs: &[af::Seq<i32>; 1],       // indecies  for bias values in
    network_params: &af::Array<f32>, // all trainable parameters as one array
    time_steps: u64,
    Z: &mut af::Array<f32>, // output membrane potential states [neuron_size x batch_size x time_steps]
    Q: &mut af::Array<f32>, // output spike mask states [neuron_size x batch_size x time_steps]
) {
    // Parameter
    let neuron_size = netdata.neuron_size;
    let input_size = netdata.input_size;
    let batch_size = netdata.batch_size;

    // let time_steps = 50;
    // LIF neuron parameters ()
    let tau_m = 100.0_f32; // membrane time constant
    let v_th = 1_f32; // firing threshold
    let v_reset: f32 = 0.00_f32; // reset potential after spike (er)
    let dt = 1.0_f32; // length time step  ( 1ms )
                      // Prepare weight matrix and bias vector
    let W_values = af::index(network_params, Wseqs);
    let W = af::sparse::<f32>(
        neuron_size,
        neuron_size,
        &W_values,
        WRowIdxCSR,
        WColIdx,
        af::SparseFormat::CSR,
    );
    let H: arrayfire::Array<f32> = af::index(network_params, Hseqs); // bias vector
    let mut V_prev = af::constant::<f32>(0.0, af::Dim4::new(&[neuron_size, batch_size, 1, 1]));
    let mut Q_prev = af::constant::<f32>(0.0, af::Dim4::new(&[neuron_size, batch_size, 1, 1]));

    let mut input_current: arrayfire::Array<f32> =
        af::constant::<f32>(0.0, af::Dim4::new(&[neuron_size, batch_size, 1, 1]));
    // temporary value for input
    let input_seq = af::Seq::new(0.0f32, (input_size - 1) as f32, 1.0f32);
    let sdim = Dim4::new(&[neuron_size, batch_size, 1, 1]);
    let c_inv_tau = af::constant::<f32>(-1.0 / tau_m, sdim);
    let c_dt: Array<f32> = af::constant::<f32>(dt, sdim);
    let c_vth = af::constant::<f32>(v_th, sdim);
    let c_vreset = af::constant::<f32>(v_reset, sdim);
    let x_slices = X.dims()[2];
    let single_input = x_slices <= 1;

    // Loop over each time step
    for i in 0..time_steps {
        if !single_input {
            // If X has multiple slices (time-varying input), use the t-th slice
            let X_t: Array<f32> = af::slice(X, i as i64);
            // Reset input_current

            let input_i32 = input_size as i64;
            let tail = af::rows(&Q_prev, input_i32, (neuron_size as i64) - 1); // [neuron_size-input, B, 1, 1]
            Q_prev = af::join(0, &X_t, &tail);

            af::sync(0);
        } else if i == 0 {
            // If only one input slice is provided
            let X0 = if x_slices == 1 {
                af::slice(X, 0) // single slice
            } else {
                X.clone() // input is 2D
            };
            af::assign_seq(&mut Q_prev, &[input_seq, af::Seq::default()], &X0);
            af::sync(0);
        }

        // Wighted input from spikes: I_syn = W * Q_prev
        let mut I_syn: Array<f32> = af::matmul(&W, &Q_prev, af::MatProp::NONE, af::MatProp::NONE);
        I_syn = af::add(&I_syn, &H, false); // add constant

        let decay: Array<f32> = af::mul(&V_prev, &c_inv_tau, false); // decay = membrane potential times the decay rate
        let dv: Array<f32> = af::add(&decay, &I_syn, false);
        let V_prop: Array<f32> = af::add(&V_prev, &af::mul(&dv, &c_dt, false), false);

        let spike_mask = af::ge(&V_prop, &c_vth, false);
        let V_next: Array<f32> = af::select(&c_vreset, &spike_mask, &V_prop); // compare membrane potential with threshold
        let zero = af::constant::<f32>(0.0, Dim4::new(&[1, 1, 1, 1]));
        let S_t: Array<f32> = spike_mask.cast::<f32>();
        af::set_slice(Z, &V_prop, i as i64);
        af::set_slice(Q, &S_t, i as i64);

        V_prev = V_next; // membrane potential carries to next step
        Q_prev = S_t; // spikes at this step become input for next step
    }
}

//Hard reset, the membrane potential will be set to the reset voltage after firing
// Forward pass through spiking neural network (LIF).

//Soft reset, the membrane potential will decrease the threshold potential after firing

//   pub fn state_space_forward_batch_snn_soft(
//         netdata: &NetworkMetadata,             // Neural Network Metadata
//         X: &af::Array<f32>,                   // input matrix
//         WRowIdxCSR: &af::Array<i32>,          // Row sparse matrix of the weighted adjacency matrix
//         WColIdx: &af::Array<i32>,             // Column sparse matrix of the weighted adjacency matrix
//         Wseqs: &[af::Seq<i32>; 1],            // Indexes of the weight parameters
//         Hseqs: &[af::Seq<i32>; 1],            // index range for bias values in `network_params`
//         network_params: &af::Array<f32>,      // all trainable parameters as one array
//          time_steps: u64,
//         Z: &mut af::Array<f32>,               // output membrane potential states [neuron_size x batch_size x time_steps]
//         Q: &mut af::Array<f32>,               // output spike mask states [neuron_size x batch_size x time_steps]

//     ) {
//         // Parameter
//         let neuron_size = netdata.neuron_size;
//         let input_size  = netdata.input_size;
//         let batch_size  = netdata.batch_size;
//         // let time_steps: u64  = Z.dims()[2];  // number of time steps
//         let time_steps = 20;

//         // LIF neuron parameters ()
//         let tau_m    = 10.0_f32;    // membrane time constant
//         let v_th     = 0.5f32;       // firing threshold
//         let v_reset: f32  = 0.00_f32;     // reset potential after spike (er)
//         let dt: f32       = 1.0_f32;     // length time step  ( 1ms )
//         let beta   = 0.98;        // exp(-dt/tau)

//         // Prepare weight matrix and bias vector
//         let W_values: Array<f32> = af::index(network_params, Wseqs);       // weight values (length = number of nonzero connections)
//         let W: Array<f32> = af::sparse::<f32>(neuron_size, neuron_size, &W_values, WRowIdxCSR, WColIdx, af::SparseFormat::CSR);
//         let H: arrayfire::Array<f32> = af::index(network_params, Hseqs);                              // bias vector
//         let mut V_prev = af::constant::<f32>(0.0, af::Dim4::new(&[neuron_size, batch_size, 1, 1]));
//         let mut Q_prev = af::constant::<f32>(0.0, af::Dim4::new(&[neuron_size, batch_size, 1, 1]));

//         let mut input_current: arrayfire::Array<f32> = af::constant::<f32>(0.0, af::Dim4::new(&[neuron_size, batch_size, 1, 1]));
//         // temporary value for input
//     let seqs = &[arrayfire::Seq::new(0.0f32, (input_size-1) as f32, 1.0f32),  arrayfire::Seq::default()];
//         // postion of input_current
//         let sdim = Dim4::new(&[neuron_size, batch_size, 1, 1]);
//         let c_beta = af::constant::<f32>(beta, sdim);
//         let c_dt: Array<f32>      = af::constant::<f32>(dt,           sdim);
//         let c_vth     = af::constant::<f32>(v_th,         sdim);
//         let c_vreset  = af::constant::<f32>(v_reset,      sdim);

//         let x_slices = X.dims()[2];  // traj_size of x
//     let mut tempx =  arrayfire::slice(X, 0);

//         // Loop over each time step

//             for i in 0..time_steps{
//                         // let X_t = if x_slices > 1 { af::slice(X, i as i64) } else { X.clone() };

//             //  if i!= 0{
//                  if x_slices > 1
//         {
//             tempx =  arrayfire::slice(X, i);
//             arrayfire::assign_seq(&mut Q_prev, seqs, &tempx);
//             drop(tempx);
//         }
//         else
//         {
//             arrayfire::assign_seq(&mut Q_prev, seqs, &X);
//         }

//             let mut I_syn: Array<f32>  = af::matmul(&W, &Q_prev, af::MatProp::NONE, af::MatProp::NONE);
//             I_syn = af::add(&I_syn, &H, false);

//             let mut reset: Array<f32> = af::mul(&Q_prev,  &c_vth,false);

//             let decay  = af::mul(&V_prev, &c_beta, false);

//             let V_prop    = af::sub(&af::add(&I_syn, &decay, false), &reset, false);

//             let spike_mask = af::ge(&V_prop, &c_vth, false);
//             let S_t = spike_mask.cast::<f32>();
//             af::set_slice(Z, &V_prop, i as i64);
//             af::set_slice(Q, &S_t, i as i64);
//             V_prev = V_prop;   // membrane potential carries to next step
//             Q_prev = S_t;      // spikes at this step become input for next step
//             // af_print!( "Q_prev     Q_prev {:?}", Q_prev);

//         }
//     }

//     Backpropagate
//     Compute total_error by applying the loss function
//     Compute surrogate dσ/dV (here: β·σ·(1-σ) on V) and multiply by local error to get `dX = dL/dV`.
//     Accumulate bias grads
//     Compute weight grads for edges gW = (dX_rows ⊙ inputs_on_the_source_side)
//     Summed over batch, scatter-add to grad[valsel_out[i]].
//     Propagate error to previous selection via sparse CSR

const COO_FIND_LIMIT: u64 = 1_500_000_000;

pub fn state_space_backward_batch(
    netdata: &NetworkMetadata,
    X: &Array<f32>,                                                       // prediction
    network_params: &Array<f32>, //All trainable parameters in the network
    Z: &Array<f32>,              // membrane potential
    Q: &Array<f32>,              // spike
    Y: &Array<f32>,              // targetsa
    loss_grad: impl Fn(&Array<f32>, &Array<f32>, f32, f32) -> Array<f32>, // loss gradient function
    neuron_idx: &Array<i32>,     // Indexes of neurons
    idxsel_out: &nohash_hasher::IntMap<i64, Array<i32>>, // The rows (neurons) that have a path to any output
    valsel_out: &nohash_hasher::IntMap<i64, Array<i32>>, // indexes of WValues that edge targets  idxsel_out
    cvec_out: &nohash_hasher::IntMap<i64, Array<i32>>,   // the columns of WValues
    dXsel_out: &nohash_hasher::IntMap<i64, Array<i32>>,  // Indexes of the dX values
    nrows_out: &nohash_hasher::IntMap<i64, u64>, //row count of the CSR matrix used to propagate error backward
    sparseval_out: &nohash_hasher::IntMap<i64, Array<i32>>, //Indexes of the values in the sparse matrix
    sparserow_out: &nohash_hasher::IntMap<i64, Array<i32>>, //Indexes of the rows in the sparse matrix
    sparsecol_out: &nohash_hasher::IntMap<i64, Array<i32>>, //Indexes of the columns in the sparse matrix
    Hidxsel_out: &nohash_hasher::IntMap<i64, Array<i32>>,   //bias parameter indices
    grad: &mut Array<f32>,                                  //  Gradient of  trainable parameters
    T: u64,
) {
    let neuron_size = netdata.neuron_size;
    let input_size = netdata.input_size;
    let output_size = netdata.output_size;
    let proc_num = netdata.proc_num;
    let batch_size = netdata.batch_size;

    let Zslices = Z.dims()[2];
    let Xslices = X.dims()[2];
    let Qslices = Q.dims()[2];
    let Yslices = Y.dims()[2];

    *grad = af::constant::<f32>(0.0, network_params.dims());
    let zero_thr: f32 = 1e-8; // tune if needed
    let log_every: i64 = 100; // print every 100 steps for IT>1

    // global output rows in neuron_idx ordering
    let active_size = neuron_idx.dims()[0];
    let out_rows_global = af::rows(
        neuron_idx,
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );

    let batch_seq: Seq<f32> = Seq::new(0.0, (batch_size - 1) as f32, 1.0);
    let yhat_time_seq: Seq<f32> = Seq::new((1 - 1) as f32, (Qslices - 1) as f32, 1.0);

    let mut yhat_idx = af::Indexer::default();
    yhat_idx.set_index(&out_rows_global, 0, None);
    yhat_idx.set_index(&batch_seq, 1, None);
    yhat_idx.set_index(&yhat_time_seq, 2, None);

    let Yhat = af::index_gen(Q, yhat_idx); // [out, batch, Yslices, 1]

    let total_error = loss_grad(&Yhat, Y, 10.0, 1e-6); // error gradient

    // error with last label slice
    let mut yslicidx: i64 = (Yslices - 1) as i64;
    let mut error = af::slice(&total_error, yslicidx); // [out, batch, 1, 1] error^t-1

    //  seqs for input overwrite
    let input_seq_rows: Seq<f32> = Seq::new(0.0, (input_size - 1) as f32, 1.0);
    let input_seq_all: Seq<f32> = Seq::<f32>::default();
    let input_range: [Seq<f32>; 2] = [input_seq_rows, input_seq_all];

    let state_dims = Dim4::new(&[neuron_size, batch_size, 1, 1]);

    // surrogate slope
    let beta: f32 = 5.0;
    let mut derror = arrayfire::slice(&total_error, 0);

    for i in (0..Zslices as i64).rev() {
        let (sel_i, dXsel_i, cvec_i, valsel_i) = match (
            idxsel_out.get(&i), // index of the neurons that still have a path to outputs
            dXsel_out.get(&i),  // Indexes of the dX values
            cvec_out.get(&i),   // the columns of WRowIdx
            valsel_out.get(&i), // indexes of WValues
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => continue,
        };

        // select Z rows for this time

        let time_one: Seq<f32> = Seq::new(i as f32, i as f32, 1.0);
        let mut idxrs = af::Indexer::default();
        idxrs.set_index(sel_i, 0, None);
        idxrs.set_index(&batch_seq, 1, None);
        idxrs.set_index(&time_one, 2, None);

        let Xtemp: Array<f32> = af::index_gen(Z, idxrs); // membrane potentials value of the neurons that have a path to outputs

        let coo_batch_size: u64 = 1 + (COO_FIND_LIMIT / sel_i.dims()[0].max(1u64));
        let local_out_rows = COO_batch_find(sel_i, &out_rows_global, coo_batch_size);
        let out_present = local_out_rows.elements() as u64;

        // print occasionally (avoid spam)
        // if i % 200 == 0 || i == (Zslices as i64 - 1) || i == 0 {
        //     println!(
        //         "[check] i={} sel_len={} out_present={} (expected up to {})",
        //         i,
        //         sel_i.elements(),
        //         out_present,
        //         output_size
        //     );
        // }
        // surrogate derivative dσ/dx
        let bx: Array<f32> = af::mul(&Xtemp, &af::constant::<f32>(beta, Xtemp.dims()), false);
        let sigma = af::sigmoid(&bx);
        let d_sigma: Array<f32> = {
            let one_minus_sigma = af::sub(&af::constant::<f32>(1.0, sigma.dims()), &sigma, false);
            let core = af::mul(&sigma, &one_minus_sigma, false);
            af::mul(&core, &af::constant::<f32>(beta, core.dims()), false)
        };

        let mut inject: bool = false;

        // println!("dsigma dim{}, errordim{}",d_sigma.dims(),error.dims());
        let dX: Array<f32> = af::mul(&d_sigma, &error, false); // dL/du
                                                               // println!("d_sigma dim{}, error dim{}",d_sigma.dims(),error.dims());

        let error_l2: f32 = af::norm(&error, af::NormType::VECTOR_2, 0.0, 0.0) as f32;
        let dsigma_l2: f32 = af::norm(&d_sigma, af::NormType::VECTOR_2, 0.0, 0.0) as f32;
        let dx_l2: f32 = af::norm(&dX, af::NormType::VECTOR_2, 0.0, 0.0) as f32;

        // periodic log (don’t spam 3000 lines)
        // if (i % log_every) == 0 || i == (Zslices as i64 - 1) || i == 0 {
        //     println!(
        //         "i={:4} inject={} ||errlocal||2={:.3e} ||error||2={:.3e} ||dX||2={:.3e} ||d_sigma||2={:.3e} sel={}",
        //         i, inject, errlocal_l2, error_l2, dx_l2, dsigma_l2, sel_i.elements()
        //     );
        // }

        // ---- bias gradient (H) ----
        if let Some(hsel_i) = Hidxsel_out.get(&i) {
            let gH_prev = af::lookup(grad, hsel_i, 0);
            let gH_add: Array<f32> = af::sum(&dX, 1); // Σ over the this mini batch
            let gH = gH_prev + gH_add; // accumulate over time slices
            let mut idh = af::Indexer::default();
            idh.set_index(hsel_i, 0, None);
            af::assign_gen(grad, &idh, &gH);
        }

        //Get dX(delta) for all sparse edges
        let mut iddx = af::Indexer::default();
        iddx.set_index(dXsel_i, 0, None);
        iddx.set_index(&batch_seq, 1, None);
        let tempdX = af::index_gen(&dX, iddx); // [nnz_rows_for_W, batch, 1, 1]

        // Build input buffer: previous spikes +  inputs
        let mut inx: Array<f32> = af::constant::<f32>(0.0, state_dims); // di/dw indexed by neuron row

        if i > 0 {
            let q_prev = af::slice(Q, i - 1);
            inx = q_prev;
        }
        if Xslices > 1 {
            let x_t: Array<f32> = af::slice(X, i);
            af::assign_seq(&mut inx, &input_range, &x_t);
        } else {
            af::assign_seq(&mut inx, &input_range, &X);
        }
        // println!("dasdsaasd");
        // af_print!("W: {:?} ",W);

        let mut idin = af::Indexer::default();
        idin.set_index(cvec_i, 0, None); // di/dw indexed by edge
        idin.set_index(&batch_seq, 1, None);
        let temp_inx: Array<f32> = af::index_gen(&inx, idin); // [nnz_rows_for_W, batch, 1, 1]

        let mut gW: Array<f32> = af::mul(&tempdX, &temp_inx, false); // compute gW
        gW = af::sum(&gW, 1) + af::lookup(grad, valsel_i, 0);
        let mut idw = af::Indexer::default();
        idw.set_index(valsel_i, 0, None);
        af::assign_gen(grad, &idw, &gW);

        //Propagate Errors
        let wvals = af::lookup(network_params, &sparseval_out[&i], 0);
        let tempW = af::sparse::<f32>(
            nrows_out[&i], // rows: sources for previous time selection
            dX.dims()[0],  // cols: current selection size
            &wvals,
            &sparserow_out[&i],
            &sparsecol_out[&i],
            af::SparseFormat::CSR,
        );
        error = af::matmul(&tempW, &dX, af::MatProp::NONE, af::MatProp::NONE); // error = delt*weight
                                                                               // println!("tempW dim{}, dX dim{}",tempW.dims(),dX.dims());
                                                                               //             println!("error dim{}, derror dim{}",error.dims(),derror.dims());

        drop(tempW);
        // if yslicidx > 0 {
        //     yslicidx -= 1;

        //     let prev_err_out = af::slice(&total_error, yslicidx); // [out, batch, 1, 1]

        //     // next iteration will use `error` in the *previous* selection space.
        //     //  add the label signal that corresponds to that timestep into the
        //     // rows of `error` that correspond to outputs of that selection.
        //     if let Some(sel_prev) = idxsel_out.get(&(i - 1)) {
        //         let coo_batch_size: u64 = 1 + (COO_FIND_LIMIT / sel_prev.dims()[0].max(1u64));
        //         let local_out_prev = COO_batch_find(sel_prev, &out_rows_global, coo_batch_size);

        //         if local_out_prev.dims()[0] > 0 && error.dims()[0] == sel_prev.dims()[0] {
        //             // take the slice of `error` at rows = local_out_prev, add prev_err_out, write back
        //                             println!("YES!");

        //             let mut ridx_get = af::Indexer::default();
        //             ridx_get.set_index(&local_out_prev, 0, None);
        //             ridx_get.set_index(&batch_seq,      1, None);
        //             let patch_old = af::index_gen(&error, ridx_get); // [out, batch, 1, 1]

        //             // Build a write indexer (identical selection)
        //             let mut ridx_put = af::Indexer::default();
        //             ridx_put.set_index(&local_out_prev, 0, None);
        //             ridx_put.set_index(&batch_seq,      1, None);
        //             let patch_new = patch_old + prev_err_out;
        //             af_print!("before",error);
        //             af::assign_gen(&mut error, &ridx_put, &patch_new);
        //             af_print!("after",error);

        //         }
        //     }
        // }
        // if ((i+1)%20 == 0) {

        derror = arrayfire::slice(&total_error, i);

        // println!("error dim{}, derror dim{}",error.dims(),derror.dims());

        error = arrayfire::join(0, &error, &derror);
        // }
        //                 print_i32_vec("out_rows_global", &out_rows_global, 16);
        //         let sel_len = sel_i.elements() as i64;
        // let tail_n: i64 = 10;
        // let start = (sel_len - tail_n).max(0);
        // let sel_tail = arrayfire::rows(&sel_i, start, sel_len - 1);
        // print_i32_vec("sel_i tail", &sel_tail, 32);
    }
}

pub fn state_space_backward_batch2(
    netdata: &NetworkMetadata,
    X: &Array<f32>,                                             // prediction
    network_params: &Array<f32>, //All trainable parameters in the network
    Z: &Array<f32>,              // membrane potential
    Q: &Array<f32>,              // spike
    Y: &Array<f32>,              // targetsa
    loss_grad: impl Fn(&Array<f32>, &Array<f32>) -> Array<f32>, // loss gradient function
    neuron_idx: &Array<i32>,     // Indexes of neurons
    idxsel_out: &nohash_hasher::IntMap<i64, Array<i32>>, // The rows (neurons) that have a path to any output
    valsel_out: &nohash_hasher::IntMap<i64, Array<i32>>, // indexes of WValues that edge targets  idxsel_out
    cvec_out: &nohash_hasher::IntMap<i64, Array<i32>>,   // the columns of WValues
    dXsel_out: &nohash_hasher::IntMap<i64, Array<i32>>,  // Indexes of the dX values
    nrows_out: &nohash_hasher::IntMap<i64, u64>, //row count of the CSR matrix used to propagate error backward
    sparseval_out: &nohash_hasher::IntMap<i64, Array<i32>>, //Indexes of the values in the sparse matrix
    sparserow_out: &nohash_hasher::IntMap<i64, Array<i32>>, //Indexes of the rows in the sparse matrix
    sparsecol_out: &nohash_hasher::IntMap<i64, Array<i32>>, //Indexes of the columns in the sparse matrix
    Hidxsel_out: &nohash_hasher::IntMap<i64, Array<i32>>,   //bias parameter indices
    grad: &mut Array<f32>,                                  //  Gradient of  trainable parameters
) {
    let neuron_size = netdata.neuron_size;
    let input_size = netdata.input_size;
    let output_size = netdata.output_size;
    let proc_num = netdata.proc_num;
    let batch_size = netdata.batch_size;

    let Zslices = Z.dims()[2];
    let Xslices = X.dims()[2];
    let Qslices = Q.dims()[2];
    let Yslices = Y.dims()[2];

    *grad = af::constant::<f32>(0.0, network_params.dims());
    let zero_thr: f32 = 1e-8; // tune if needed
    let log_every: i64 = 100; // print every 100 steps for IT>1

    // global output rows in neuron_idx ordering
    let active_size = neuron_idx.dims()[0];
    let out_rows_global = af::rows(
        neuron_idx,
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );

    let batch_seq: Seq<f32> = Seq::new(0.0, (batch_size - 1) as f32, 1.0);
    let yhat_time_seq: Seq<f32> = Seq::new((1 - 1) as f32, (Qslices - 1) as f32, 1.0);

    let mut yhat_idx = af::Indexer::default();
    yhat_idx.set_index(&out_rows_global, 0, None);
    yhat_idx.set_index(&batch_seq, 1, None);
    yhat_idx.set_index(&yhat_time_seq, 2, None);

    let Yhat = af::index_gen(Q, yhat_idx); // [out, batch, Yslices, 1]

    let total_error = loss_grad(&Yhat, Y); // error gradient

    // error with last label slice
    let mut yslicidx: i64 = (Yslices - 1) as i64;
    let mut error = af::slice(&total_error, yslicidx); // [out, batch, 1, 1] error^t-1

    //  seqs for input overwrite
    let input_seq_rows: Seq<f32> = Seq::new(0.0, (input_size - 1) as f32, 1.0);
    let input_seq_all: Seq<f32> = Seq::<f32>::default();
    let input_range: [Seq<f32>; 2] = [input_seq_rows, input_seq_all];

    let state_dims = Dim4::new(&[neuron_size, batch_size, 1, 1]);

    // surrogate slope
    let beta: f32 = 2.0;
    let mut derror = arrayfire::slice(&total_error, 0);
    for i in (0..Zslices as i64).rev() {
        let (sel_i, dXsel_i, cvec_i, valsel_i) = match (
            idxsel_out.get(&i), // index of the neurons that still have a path to outputs
            dXsel_out.get(&i),  // Indexes of the dX values
            cvec_out.get(&i),   // the columns of WRowIdx
            valsel_out.get(&i), // indexes of WValues
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => continue,
        };

        // select Z rows for this time

        let time_one: Seq<f32> = Seq::new(i as f32, i as f32, 1.0);
        let mut idxrs = af::Indexer::default();
        idxrs.set_index(sel_i, 0, None);
        idxrs.set_index(&batch_seq, 1, None);
        idxrs.set_index(&time_one, 2, None);

        let Xtemp: Array<f32> = af::index_gen(Z, idxrs); // membrane potentials value of the neurons that have a path to outputs

        let coo_batch_size: u64 = 1 + (COO_FIND_LIMIT / sel_i.dims()[0].max(1u64));
        let local_out_rows = COO_batch_find(sel_i, &out_rows_global, coo_batch_size);
        let out_present = local_out_rows.elements() as u64;

        // print occasionally (avoid spam)
        // if i % 200 == 0 || i == (Zslices as i64 - 1) || i == 0 {
        //     println!(
        //         "[check] i={} sel_len={} out_present={} (expected up to {})",
        //         i,
        //         sel_i.elements(),
        //         out_present,
        //         output_size
        //     );
        // }

        // surrogate derivative dσ/dx
        let bx: Array<f32> = af::mul(&Xtemp, &af::constant::<f32>(beta, Xtemp.dims()), false);
        let sigma = af::sigmoid(&bx);
        let d_sigma: Array<f32> = {
            let one_minus_sigma = af::sub(&af::constant::<f32>(1.0, sigma.dims()), &sigma, false);
            let core = af::mul(&sigma, &one_minus_sigma, false);
            af::mul(&core, &af::constant::<f32>(beta, core.dims()), false)
        };

        let mut inject: bool = false;

        let mut error_local = af::constant::<f32>(0.0, Xtemp.dims());
        {
            // print_head_tail_i32(&format!("i={} sel_i", i), sel_i, 10);
            // Map global outputs -> rows in sel_i
            let coo_batch_size: u64 = 1 + (COO_FIND_LIMIT / sel_i.dims()[0].max(1u64));
            let local_out_rows = COO_batch_find(sel_i, &out_rows_global, coo_batch_size); // [out,1,1,1] positions in sel_i
            let inject = local_out_rows.elements() > 0;
            if local_out_rows.dims()[0] > 0 && error.dims()[0] == output_size {
                // scatter the output-space error into current selection rows
                let mut lidx = af::Indexer::default();
                lidx.set_index(&local_out_rows, 0, None);
                lidx.set_index(&batch_seq, 1, None);
                af::assign_gen(&mut error_local, &lidx, &error);
                // println!("111");
            } else if error.dims()[0] == Xtemp.dims()[0] {
                // already in local space (e.g., after first propagation)
                error_local = error.clone();
                // println!("222");
            } else {
                // shape mismatch: fall back to zeros for this step
                error_local = af::constant::<f32>(0.0, Xtemp.dims());
                // println!("333");
            }

            let out_global_at_local = arrayfire::lookup(&sel_i, &local_out_rows, 0);
        }
        // println!("dsigma dim{}, errordim{}",d_sigma.dims(),error.dims());

        let dX: Array<f32> = af::mul(&d_sigma, &error, false); // dL/du
                                                               // println!("d_sigma dim{}, error dim{}",d_sigma.dims(),error.dims());

        let errlocal_l2: f32 = af::norm(&error_local, af::NormType::VECTOR_2, 0.0, 0.0) as f32;
        let error_l2: f32 = af::norm(&error, af::NormType::VECTOR_2, 0.0, 0.0) as f32;
        let dsigma_l2: f32 = af::norm(&d_sigma, af::NormType::VECTOR_2, 0.0, 0.0) as f32;
        let dx_l2: f32 = af::norm(&dX, af::NormType::VECTOR_2, 0.0, 0.0) as f32;

        // periodic log (don’t spam 3000 lines)
        // if (i % log_every) == 0 || i == (Zslices as i64 - 1) || i == 0 {
        //     println!(
        //         "i={:4} inject={} ||errlocal||2={:.3e} ||error||2={:.3e} ||dX||2={:.3e} ||d_sigma||2={:.3e} sel={}",
        //         i, inject, errlocal_l2, error_l2, dx_l2, dsigma_l2, sel_i.elements()
        //     );
        // }

        // ---- bias gradient (H) ----
        if let Some(hsel_i) = Hidxsel_out.get(&i) {
            let gH_prev = af::lookup(grad, hsel_i, 0);
            let gH_add: Array<f32> = af::sum(&dX, 1); // Σ over the this mini batch
            let gH = gH_prev + gH_add; // accumulate over time slices
            let mut idh = af::Indexer::default();
            idh.set_index(hsel_i, 0, None);
            af::assign_gen(grad, &idh, &gH);
        }

        //Get dX(delta) for all sparse edges
        let mut iddx = af::Indexer::default();
        iddx.set_index(dXsel_i, 0, None);
        iddx.set_index(&batch_seq, 1, None);
        let tempdX = af::index_gen(&dX, iddx); // [nnz_rows_for_W, batch, 1, 1]

        // Build input buffer: previous spikes +  inputs
        let mut inx: Array<f32> = af::constant::<f32>(0.0, state_dims); // di/dw indexed by neuron row

        if i > 0 {
            let q_prev = af::slice(Q, i - 1);
            inx = q_prev;
        }
        if Xslices > 1 {
            let x_t: Array<f32> = af::slice(X, i);
            af::assign_seq(&mut inx, &input_range, &x_t);
        } else {
            af::assign_seq(&mut inx, &input_range, &X);
        }
        // println!("dasdsaasd");
        // af_print!("W: {:?} ",W);

        let mut idin = af::Indexer::default();
        idin.set_index(cvec_i, 0, None); // di/dw indexed by edge
        idin.set_index(&batch_seq, 1, None);
        let temp_inx: Array<f32> = af::index_gen(&inx, idin); // [nnz_rows_for_W, batch, 1, 1]

        let mut gW: Array<f32> = af::mul(&tempdX, &temp_inx, false); // compute gW
        gW = af::sum(&gW, 1) + af::lookup(grad, valsel_i, 0);
        let mut idw = af::Indexer::default();
        idw.set_index(valsel_i, 0, None);
        af::assign_gen(grad, &idw, &gW);

        //Propagate Errors
        let wvals = af::lookup(network_params, &sparseval_out[&i], 0);
        let tempW = af::sparse::<f32>(
            nrows_out[&i], // rows: sources for previous time selection
            dX.dims()[0],  // cols: current selection size
            &wvals,
            &sparserow_out[&i],
            &sparsecol_out[&i],
            af::SparseFormat::CSR,
        );
        error = af::matmul(&tempW, &dX, af::MatProp::NONE, af::MatProp::NONE); // error = delt*weight
                                                                               // println!("tempW dim{}, dX dim{}",tempW.dims(),dX.dims());
                                                                               //             println!("error dim{}, derror dim{}",error.dims(),derror.dims());

        drop(tempW);
        // if yslicidx > 0 {
        //     yslicidx -= 1;

        //     let prev_err_out = af::slice(&total_error, yslicidx); // [out, batch, 1, 1]

        //     // next iteration will use `error` in the *previous* selection space.
        //     //  add the label signal that corresponds to that timestep into the
        //     // rows of `error` that correspond to outputs of that selection.
        //     if let Some(sel_prev) = idxsel_out.get(&(i - 1)) {
        //         let coo_batch_size: u64 = 1 + (COO_FIND_LIMIT / sel_prev.dims()[0].max(1u64));
        //         let local_out_prev = COO_batch_find(sel_prev, &out_rows_global, coo_batch_size);

        //         if local_out_prev.dims()[0] > 0 && error.dims()[0] == sel_prev.dims()[0] {
        //             // take the slice of `error` at rows = local_out_prev, add prev_err_out, write back
        //                             println!("YES!");

        //             let mut ridx_get = af::Indexer::default();
        //             ridx_get.set_index(&local_out_prev, 0, None);
        //             ridx_get.set_index(&batch_seq,      1, None);
        //             let patch_old = af::index_gen(&error, ridx_get); // [out, batch, 1, 1]

        //             // Build a write indexer (identical selection)
        //             let mut ridx_put = af::Indexer::default();
        //             ridx_put.set_index(&local_out_prev, 0, None);
        //             ridx_put.set_index(&batch_seq,      1, None);
        //             let patch_new = patch_old + prev_err_out;
        //             af_print!("before",error);
        //             af::assign_gen(&mut error, &ridx_put, &patch_new);
        //             af_print!("after",error);

        //         }
        //     }
        // }
        // if ((i+1)%20 == 0) {
        derror = arrayfire::slice(&total_error, i);
        // println!("error dim{}, derror dim{}",error.dims(),derror.dims());
        error = arrayfire::join(0, &error, &derror);
        // }
        //                 print_i32_vec("out_rows_global", &out_rows_global, 16);
        //         let sel_len = sel_i.elements() as i64;
        // let tail_n: i64 = 10;
        // let start = (sel_len - tail_n).max(0);
        // let sel_tail = arrayfire::rows(&sel_i, start, sel_len - 1);
        // print_i32_vec("sel_i tail", &sel_tail, 32);
    }
}

// pub fn state_space_backward_batch2(
//     netdata: &NetworkMetadata,

//     X: &Array<f32>,                 // prediction / input
//     network_params: &Array<f32>,    // all trainable parameters
//     Z: &Array<f32>,                 // pre-reset membrane potential V_prop
//     Q: &Array<f32>,                 // spike
//     Y: &Array<f32>,                 // targets

//     loss_grad: impl Fn(&Array<f32>, &Array<f32>) -> Array<f32>,

//     neuron_idx: &Array<i32>,

//     idxsel_out: &nohash_hasher::IntMap<i64, Array<i32>>,
//     valsel_out: &nohash_hasher::IntMap<i64, Array<i32>>,
//     cvec_out: &nohash_hasher::IntMap<i64, Array<i32>>,
//     dXsel_out: &nohash_hasher::IntMap<i64, Array<i32>>,

//     nrows_out: &nohash_hasher::IntMap<i64, u64>,

//     sparseval_out: &nohash_hasher::IntMap<i64, Array<i32>>,
//     sparserow_out: &nohash_hasher::IntMap<i64, Array<i32>>,
//     sparsecol_out: &nohash_hasher::IntMap<i64, Array<i32>>,

//     Hidxsel_out: &nohash_hasher::IntMap<i64, Array<i32>>,

//     grad: &mut Array<f32>,
// ) {


//     let neuron_size = netdata.neuron_size;
//     let input_size = netdata.input_size;
//     let output_size = netdata.output_size;
//     let batch_size = netdata.batch_size;

//     let Zslices = Z.dims()[2];
//     let Xslices = X.dims()[2];
//     let Qslices = Q.dims()[2];
//     let Yslices = Y.dims()[2];


//     *grad = af::constant::<f32>(
//         0.0,
//         network_params.dims(),
//     );



//     let active_size = neuron_idx.dims()[0];

//     let out_rows_global = af::rows(
//         neuron_idx,
//         (active_size - output_size) as i64,
//         (active_size - 1) as i64,
//     );



//     let batch_seq: Seq<f32> =
//         Seq::new(
//             0.0,
//             (batch_size - 1) as f32,
//             1.0,
//         );


//     let yhat_time_seq: Seq<f32> =
//         Seq::new(
//             0.0,
//             (Qslices - 1) as f32,
//             1.0,
//         );

//     let mut yhat_idx =
//         af::Indexer::default();

//     yhat_idx.set_index(
//         &out_rows_global,
//         0,
//         None,
//     );

//     yhat_idx.set_index(
//         &batch_seq,
//         1,
//         None,
//     );

//     yhat_idx.set_index(
//         &yhat_time_seq,
//         2,
//         None,
//     );

//     let Yhat =
//         af::index_gen(
//             Q,
//             yhat_idx,
//         );



//     let total_error =
//         loss_grad(
//             &Yhat,
//             Y,
//         );

//     // Start from final label/output slice
//     let mut yslicidx: i64 =
//         (Yslices - 1) as i64;

//     let mut error =
//         af::slice(
//             &total_error,
//             yslicidx,
//         );



//     let input_seq_rows: Seq<f32> =
//         Seq::new(
//             0.0,
//             (input_size - 1) as f32,
//             1.0,
//         );

//     let input_seq_all: Seq<f32> =
//         Seq::<f32>::default();

//     let input_range: [Seq<f32>; 2] =
//         [
//             input_seq_rows,
//             input_seq_all,
//         ];

//     let state_dims =
//         Dim4::new(
//             &[
//                 neuron_size,
//                 batch_size,
//                 1,
//                 1,
//             ]
//         );



//     let beta: f32 = 2.0;



//     let tau_m: f32 = 100.0;
//     let dt: f32 = 1.0;

//     let lif_alpha: f32 =
//         1.0 - dt / tau_m;       // 0.99

    
//     let mut dU_future_global =
//         af::constant::<f32>(
//             0.0,
//             state_dims,
//         );

//     let mut derror =
//         af::slice(
//             &total_error,
//             0,
//         );

   
//     for i in (0..Zslices as i64).rev() {


//         let (
//             sel_i,
//             dXsel_i,
//             cvec_i,
//             valsel_i,
//         ) = match (
//             idxsel_out.get(&i),
//             dXsel_out.get(&i),
//             cvec_out.get(&i),
//             valsel_out.get(&i),
//         ) {
//             (
//                 Some(a),
//                 Some(b),
//                 Some(c),
//                 Some(d),
//             ) => (
//                 a,
//                 b,
//                 c,
//                 d,
//             ),

//             _ => continue,
//         };

    

//         let time_one =
//             Seq::<f32>::new(
//                 i as f32,
//                 i as f32,
//                 1.0,
//             );

//         let mut idxrs =
//             af::Indexer::default();

//         idxrs.set_index(
//             sel_i,
//             0,
//             None,
//         );

//         idxrs.set_index(
//             &batch_seq,
//             1,
//             None,
//         );

//         idxrs.set_index(
//             &time_one,
//             2,
//             None,
//         );

//         let Xtemp: Array<f32> =
//             af::index_gen(
//                 Z,
//                 idxrs,
//             );


//         let coo_batch_size: u64 =
//             1
//             + (
//                 COO_FIND_LIMIT
//                 / sel_i.dims()[0].max(1u64)
//             );

//         let local_out_rows =
//             COO_batch_find(
//                 sel_i,
//                 &out_rows_global,
//                 coo_batch_size,
//             );

     
//         // ========================================================

//         let bx: Array<f32> =
//             af::mul(
//                 &Xtemp,
//                 &af::constant::<f32>(
//                     beta,
//                     Xtemp.dims(),
//                 ),
//                 false,
//             );

//         let sigma =
//             af::sigmoid(
//                 &bx,
//             );

//         let d_sigma: Array<f32> = {

//             let one_minus_sigma =
//                 af::sub(
//                     &af::constant::<f32>(
//                         1.0,
//                         sigma.dims(),
//                     ),
//                     &sigma,
//                     false,
//                 );

//             let core =
//                 af::mul(
//                     &sigma,
//                     &one_minus_sigma,
//                     false,
//                 );

//             af::mul(
//                 &core,
//                 &af::constant::<f32>(
//                     beta,
//                     core.dims(),
//                 ),
//                 false,
//             )
//         };


//         let mut error_local =
//             af::constant::<f32>(
//                 0.0,
//                 Xtemp.dims(),
//             );

//         {
//             let coo_batch_size: u64 =
//                 1
//                 + (
//                     COO_FIND_LIMIT
//                     / sel_i.dims()[0].max(1u64)
//                 );

//             let local_out_rows =
//                 COO_batch_find(
//                     sel_i,
//                     &out_rows_global,
//                     coo_batch_size,
//                 );

//             if local_out_rows.dims()[0] > 0
//                 && error.dims()[0] == output_size
//             {
//                 let mut lidx =
//                     af::Indexer::default();

//                 lidx.set_index(
//                     &local_out_rows,
//                     0,
//                     None,
//                 );

//                 lidx.set_index(
//                     &batch_seq,
//                     1,
//                     None,
//                 );

//                 af::assign_gen(
//                     &mut error_local,
//                     &lidx,
//                     &error,
//                 );

//             } else if error.dims()[0]
//                 == Xtemp.dims()[0]
//             {
//                 error_local =
//                     error.clone();

//             } else {

//                 error_local =
//                     af::constant::<f32>(
//                         0.0,
//                         Xtemp.dims(),
//                     );
//             }
//         }



//         let dU_spike: Array<f32> =
//             af::mul(
//                 &d_sigma,
//                 &error,
//                 false,
//             );

//         let mut future_idx =
//             af::Indexer::default();

//         future_idx.set_index(
//             sel_i,
//             0,
//             None,
//         );

//         future_idx.set_index(
//             &batch_seq,
//             1,
//             None,
//         );

//         let dU_future_local =
//             af::index_gen(
//                 &dU_future_global,
//                 future_idx,
//             );

//         let S_t_global =
//             af::slice(
//                 Q,
//                 i,
//             );

//         let mut spike_idx =
//             af::Indexer::default();

//         spike_idx.set_index(
//             sel_i,
//             0,
//             None,
//         );

//         spike_idx.set_index(
//             &batch_seq,
//             1,
//             None,
//         );

//         let S_t_local =
//             af::index_gen(
//                 &S_t_global,
//                 spike_idx,
//             );

//         let one_minus_spike_local =
//             af::sub(
//                 &af::constant::<f32>(
//                     1.0,
//                     S_t_local.dims(),
//                 ),
//                 &S_t_local,
//                 false,
//             );

    
//         let temporal_gate =
//             af::mul(
//                 &one_minus_spike_local,
//                 &dU_future_local,
//                 false,
//             );

//         let temporal_dU =
//             af::mul(
//                 &temporal_gate,
//                 &af::constant::<f32>(
//                     lif_alpha,
//                     temporal_gate.dims(),
//                 ),
//                 false,
//             );

    
//         let dX: Array<f32> =
//             af::add(
//                 &dU_spike,
//                 &temporal_dU,
//                 false,
//             );



//         /*
//         if i % 100 == 0
//             || i == (Zslices as i64 - 1)
//             || i == 0
//         {
//             let err_l2 =
//                 af::norm(
//                     &error,
//                     af::NormType::VECTOR_2,
//                     0.0,
//                     0.0,
//                 ) as f32;

//             let surrogate_l2 =
//                 af::norm(
//                     &d_sigma,
//                     af::NormType::VECTOR_2,
//                     0.0,
//                     0.0,
//                 ) as f32;

//             let spike_l2 =
//                 af::norm(
//                     &dU_spike,
//                     af::NormType::VECTOR_2,
//                     0.0,
//                     0.0,
//                 ) as f32;

//             let temporal_l2 =
//                 af::norm(
//                     &temporal_dU,
//                     af::NormType::VECTOR_2,
//                     0.0,
//                     0.0,
//                 ) as f32;

//             let total_l2 =
//                 af::norm(
//                     &dX,
//                     af::NormType::VECTOR_2,
//                     0.0,
//                     0.0,
//                 ) as f32;

//             println!(
//                 "t={} error={:.3e} surrogate={:.3e} spike_grad={:.3e} temporal_grad={:.3e} total_dU={:.3e}",
//                 i,
//                 err_l2,
//                 surrogate_l2,
//                 spike_l2,
//                 temporal_l2,
//                 total_l2
//             );
//         }
//         */

//         if let Some(hsel_i) =
//             Hidxsel_out.get(&i)
//         {
//             let gH_prev =
//                 af::lookup(
//                     grad,
//                     hsel_i,
//                     0,
//                 );

//             let gH_add: Array<f32> =
//                 af::sum(
//                     &dX,
//                     1,
//                 );

//             let gH =
//                 gH_prev + gH_add;

//             let mut idh =
//                 af::Indexer::default();

//             idh.set_index(
//                 hsel_i,
//                 0,
//                 None,
//             );

//             af::assign_gen(
//                 grad,
//                 &idh,
//                 &gH,
//             );
//         }



//         let mut iddx =
//             af::Indexer::default();

//         iddx.set_index(
//             dXsel_i,
//             0,
//             None,
//         );

//         iddx.set_index(
//             &batch_seq,
//             1,
//             None,
//         );

//         let tempdX =
//             af::index_gen(
//                 &dX,
//                 iddx,
//             );

        
//         let mut inx: Array<f32> =
//             af::constant::<f32>(
//                 0.0,
//                 state_dims,
//             );

//         if i > 0 {

//             let q_prev =
//                 af::slice(
//                     Q,
//                     i - 1,
//                 );

//             inx = q_prev;
//         }

//         // External input replaces input neurons
//         if Xslices > 1 {

//             let x_t: Array<f32> =
//                 af::slice(
//                     X,
//                     i,
//                 );

//             af::assign_seq(
//                 &mut inx,
//                 &input_range,
//                 &x_t,
//             );

//         } else {

//             af::assign_seq(
//                 &mut inx,
//                 &input_range,
//                 X,
//             );
//         }

      
//         let mut idin =
//             af::Indexer::default();

//         idin.set_index(
//             cvec_i,
//             0,
//             None,
//         );

//         idin.set_index(
//             &batch_seq,
//             1,
//             None,
//         );

//         let temp_inx =
//             af::index_gen(
//                 &inx,
//                 idin,
//             );

     

//         let mut gW: Array<f32> =
//             af::mul(
//                 &tempdX,
//                 &temp_inx,
//                 false,
//             );

//         gW =
//             af::sum(
//                 &gW,
//                 1,
//             )
//             +
//             af::lookup(
//                 grad,
//                 valsel_i,
//                 0,
//             );

//         let mut idw =
//             af::Indexer::default();

//         idw.set_index(
//             valsel_i,
//             0,
//             None,
//         );

//         af::assign_gen(
//             grad,
//             &idw,
//             &gW,
//         );

        
//         let wvals =
//             af::lookup(
//                 network_params,
//                 &sparseval_out[&i],
//                 0,
//             );

//         let tempW =
//             af::sparse::<f32>(
//                 nrows_out[&i],

//                 dX.dims()[0],

//                 &wvals,

//                 &sparserow_out[&i],
//                 &sparsecol_out[&i],

//                 af::SparseFormat::CSR,
//             );

//         error =
//             af::matmul(
//                 &tempW,
//                 &dX,
//                 af::MatProp::NONE,
//                 af::MatProp::NONE,
//             );

//         drop(tempW);

 

//         dU_future_global =
//             af::constant::<f32>(
//                 0.0,
//                 state_dims,
//             );

//         let mut put_du_idx =
//             af::Indexer::default();

//         put_du_idx.set_index(
//             sel_i,
//             0,
//             None,
//         );

//         put_du_idx.set_index(
//             &batch_seq,
//             1,
//             None,
//         );

//         af::assign_gen(
//             &mut dU_future_global,
//             &put_du_idx,
//             &dX,
//         );



//         derror =
//             af::slice(
//                 &total_error,
//                 i,
//             );

//         error =
//             af::join(
//                 0,
//                 &error,
//                 &derror,
//             );
//     }
// }