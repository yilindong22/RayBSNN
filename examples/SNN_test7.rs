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
pub fn repeat_each_traj_step(x: &af::Array<f32>, t: u64) -> af::Array<f32> {
    let d = x.dims();
    let in_dim = d[0];
    let b_dim  = d[1];
    let it_dim = d[2];

    // 1) Make a new axis for repetition by tiling into dim3:
    //    [1,B,IT,1] -> [1,B,IT,T]
    let x4 = af::tile(x, af::Dim4::new(&[1, 1, 1, t]));

    // 2) Swap (IT, T) so T is next to IT in the right order:
    //    [1,B,IT,T] -> [1,B,T,IT]
    let x4 = af::reorder(&x4, af::Dim4::new(&[0, 1, 3, 2]));

    // 3) Flatten (T,IT) -> IT*T (this yields repetition groups)
    //    [1,B,T,IT] -> [1,B,IT*T,1]
    af::moddims(&x4, af::Dim4::new(&[in_dim, b_dim, it_dim * t, 1]))
}



pub fn delta_encode(
    x: &af::Array<f32>,   // [1,B,IT,1]
    gain: f32,
    clip01: bool,
) -> af::Array<f32> {

    let d = x.dims();
    let b  = d[1];
    let it = d[2];
    assert!(it >= 2);

    // dx: [1,B,IT-1,1]
    let dx = af::diff1(x, 2);

    // pad zero at t=0 -> [1,B,IT,1]
    let z0 = af::constant::<f32>(0.0, af::Dim4::new(&[1, b, 1, 1]));
    let mut delta = af::join(2, &z0, &dx);

    if gain != 1.0 {
        delta = &delta * gain;
    }

    if clip01 {
        delta = (&delta + 1.0f32) * 0.5f32;
        delta = af::clamp(&delta, &0.0f32, &1.0f32, false);
    }

    delta
}


// Delta Modulation (one channel): spike when |x[t]-x[t-1]| > theta
// Input:  x  [in, B, IT, 1]
// Output: spk[in, B, IT, 1] in {0,1}
pub fn delta_mod_encode_one_channel_abs(
    x: &af::Array<f32>,
    theta: f32,          // threshold in same scale as x
) -> af::Array<f32> {
    let d = x.dims();
    let in_dim = d[0];
    let b      = d[1];
    let it     = d[2];
    assert!(it >= 2);

    // dx: [in, B, IT-1, 1]
    let dx = af::diff1(x, 2);

    // pad zero at t=0 -> [in, B, IT, 1]
    let z0 = af::constant::<f32>(0.0, af::Dim4::new(&[in_dim, b, 1, 1]));
    let dxp = af::join(2, &z0, &dx);

    // spk = 1 if |dx| > theta else 0
    let adx = af::abs(&dxp);
    let th  = af::constant::<f32>(theta, adx.dims());
    af::gt(&adx, &th, false).cast::<f32>() // {0,1}
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


// Encoding decimal numbers into spikes over time_steps by using bernoulli trail 
//  X : data probabilities between [0,1], shape:  [input_size,batch_size,time_steps,1]
// time_steps: number of timesteps for the spikes
// Returns spikes in {0,1} with shape [input_size, batch, time_steps, 1].

	pub fn bernoulli_spike_encode(
		X: &af::Array<f32>,   
		time_steps: u64
	) -> af::Array<f32> {

		assert!(time_steps > 0);

		let xdims = X.dims();

		// 1) ensure probabilities are in [0,1]
		let zero = 0.0f32;
		let one  = 1.0f32;
		let mut P = af::clamp(X, &zero, &one, false);

		// 2) make P have exactly `time_steps` on dim=2 without invalid moddims
		if xdims[2] == time_steps {
		} else if xdims[2] == 1 {
			P = af::tile(&P, af::Dim4::new(&[1, 1, time_steps, 1]));
		} else {
			// has some T' != time_steps -> average over its time axis, then tile
			let tprime = xdims[2] as f32;
			let mean_over_time = af::sum(&P, 2) / tprime;             // [in, batch, 1, 1]
			P = af::tile(&mean_over_time, af::Dim4::new(&[1, 1, time_steps, 1]));
		}
		// 3) sample Bernoulli(P)
		let U = af::randu::<f32>(P.dims());                           // [in, batch, time_steps, 1]
		af::le(&U, &P, false).cast::<f32>()                           // {0,1} spikes
	}


    fn main() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(10);

	//Set CUDA and GPU Device 0
	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);

	arrayfire::set_seed(10  as u64);


	
        // Architecture knobs (starter)
let conn_vec: Vec<u64>        = vec![50];     // ok to keep
let neuron_size_vec: Vec<u64> = vec![250]; // ok to keep, can try 200–600 later
	let dir_path = "/tmp/".to_string() ;

// Input dims
let arg_input_vec: Vec<u64> = vec![1];

let max_input_size: u64 = 1;
let mut input_size: u64 = 1;

// Output ma 
let max_output_size: u64 = 4;
let output_size: u64 = 4;

// Capacity buffer
    let max_neuron_size = input_size + neuron_size_vec[0] + output_size; // 

// Time-series packing
let mut batch_size: u64 = 250;
let mut traj_size: u64  = 128;

// Multi-proc propagation
let mut proc_num: u64 = 25;        // keep as in figure-4 code

// Dataset sizes
let train_size: u64 = 1000;
let test_size: u64  = 4000;
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
	




	//Load dataset
	println!("Load Dataset");


	let TOTAL_RSSI_TESTX = raybnn::export::dataloader_f32::file_to_hash_cpu_batch_blocks(
		"./test_data/TwoPatterns_test_x.csv",
traj_size as usize,
		(batch_size * input_size) as usize,
	);


	let RSSI_TESTY = raybnn::export::dataloader_f32::file_to_hash_cpu(
		"./test_data/TwoPatterns_test_y_onehot.csv",
		output_size,
		batch_size
	);
	arrayfire::sync(DEVICE);


	let mut RSSI_TESTX = TOTAL_RSSI_TESTX.clone();



	let TOTAL_RSSI_TRAINX: HashMap<u64, Vec<f32>, std::hash::BuildHasherDefault<nohash_hasher::NoHashHasher<u64>>> = raybnn::export::dataloader_f32::file_to_hash_cpu_batch_blocks(
    	"./test_data/TwoPatterns_train_x.csv",
traj_size as usize,
		(batch_size * input_size) as usize,
    );


	let RSSI_TRAINY: HashMap<u64, Vec<f32>, std::hash::BuildHasherDefault<nohash_hasher::NoHashHasher<u64>>> = raybnn::export::dataloader_f32::file_to_hash_cpu(
    	"./test_data/TwoPatterns_train_y_onehot.csv",
    	output_size,
		batch_size
    );	
	arrayfire::sync(DEVICE);
	
	let mut RSSI_TRAINX = TOTAL_RSSI_TRAINX.clone();

	    let mut prev_target_neuron_size = 0;



	for ii in 0..arg_input_vec.len()
	{

let mut acc100_streak: u32 = 0;


		RSSI_TRAINX = TOTAL_RSSI_TRAINX.clone();

		RSSI_TESTX = TOTAL_RSSI_TESTX.clone();

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


		




		arrayfire::device_gc();
		arrayfire::sync(DEVICE);
		
		if (input_size != max_input_size)
		{
			let mut tempRSSI = TOTAL_RSSI_TRAINX.clone();

			for (key, value) in &TOTAL_RSSI_TRAINX {
				let mut tempvec = value.clone();
				
				let veclen =  tempvec.len() as u64;
				let mut formatarr = arrayfire::Array::new(&tempvec, arrayfire::Dim4::new(&[max_input_size, veclen/max_input_size, 1, 1]));

				formatarr = arrayfire::rows(&formatarr, 0, (input_size-1) as i64);


				tempvec = vec!(f32::default();formatarr.elements());
				formatarr.host(&mut tempvec);

				tempRSSI.insert(key.clone(), tempvec);
			}

			RSSI_TRAINX = tempRSSI;
		}
		arrayfire::sync(DEVICE);

		let T: u64 = 1; // this sets the time steps 
		let time_steps_total: u64 = traj_size * T;  // total SNN steps per trajectory
		let Z_dims = arrayfire::Dim4::new(&[arch_search.neural_network.netdata.neuron_size,batch_size*1,time_steps_total,1]);
		// [arch_search.neural_network.netdata.neuron_size,batch_size,1,1]
		let mut Z = arrayfire::constant::<f32>(0.0,Z_dims);  // membrane potential
		let mut Q = arrayfire::constant::<f32>(0.0,Z_dims);  // spikes



		let mut alpha0: f32 = 0.001;

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
let steps_x = TOTAL_RSSI_TRAINX.len() as u64;
let steps_y = RSSI_TRAINY.len() as u64;

let steps_x2 = TOTAL_RSSI_TESTX.len() as u64;
let steps_y2 = RSSI_TESTY.len() as u64;


let steps_per_epoch: u64 = steps_x.min(steps_y).max(1);
let steps_per_epoch2: u64 = steps_x2.min(steps_y2).max(1);


		let mut max_epoch = 1200;


		let mut loss_val = 100000.0;
		let mut loss_val2 = 100000.0;
		let mut prev_loss = 100000.0;

		arrayfire::device_gc();
		arrayfire::sync(DEVICE);

		let start_time = std::time::Instant::now();
		let out_cols = arrayfire::Dim4::new(&[1, 3, 1, 1]); // columns: pred,label,rate
		let mut pred_true_rate = af::constant::<f32>(0.0, out_cols);   



		for i in 0..max_epoch
		{
			batch_idx = i % steps_per_epoch;
			train_X =  arrayfire::Array::new(&RSSI_TRAINX[&batch_idx], train_X_dims);

			Y = arrayfire::Array::new(&RSSI_TRAINY[&batch_idx], Y_dims);
	
// af_print!("train_X",train_X);
af::set_seed((1234) as u64); // or constant 1234 for identical spikes every epoch
let in_dim    = train_X.dims()[0];
let batch_dim = train_X.dims()[1];
let it_dim    = train_X.dims()[2];
let t_dim     = T as u64;


	time_slice_stats("X[t=0..T-1]", &train_X, 0, (traj_size as i64)-1);
// [in,batch,IT,1] -> [in,batch,IT,T]
let x4 = af::tile(&train_X, af::Dim4::new(&[1, 1, 1, t_dim]));

// reorder so T becomes the “fast” axis: [in,batch,T,IT]
let x4 = af::reorder(&x4, af::Dim4::new(&[0, 1, 3, 2]));

// flatten (T,IT) -> IT*T : [in,batch,IT*T,1]

let gain = 500.0f32; // start with 500~2000
let x_enc: Array<f32> = delta_encode(&train_X, gain, true);
// let X_spk = bernoulli_spike_encode(&prob_K, time_steps_total);
// let prob_K = repeat_each_traj_step(&x_enc, T);
println!("train_X {:.6}", af::sum_all(&train_X).0 / train_X.elements() as f32);

let X_spk: Array<f32> = delta_mod_encode_one_channel_abs(&x_enc, 0.1);
println!("X_spk mean = {:.6}", af::sum_all(&X_spk).0 / X_spk.elements() as f32);
let y_scalar_dims = af::Dim4::new(&[output_size, batch_size, 1, 1]);
let Y_scalar = af::Array::new(&RSSI_TRAINY[&batch_idx], y_scalar_dims);
let Y_seq = af::tile(&Y_scalar, af::Dim4::new(&[1, 1, traj_size, 1]));

// stats_f32("Y_scalar", &Y_scalar);
// stats_f32("Y_seq", &Y_seq);

// // sum over classes should be 1 for every sample (one-hot)
// let ysum = af::sum(&Y_scalar, 0);   // [1, batch, 1, 1]
// stats_f32("Y_scalar sum_over_class", &ysum);

// show min/max of that sum
// let (mn, _) = af::min_all(&ysum);
// let (mx, _) = af::max_all(&ysum);
// println!("Y_scalar sum_over_class min/max = {:.3e} / {:.3e}", mn as f32, mx as f32);
let (_tv, true_idx) = af::imax(&Y_scalar, 0);
// af_print!("true_idx", true_idx);
let mask0 = af::eq(&true_idx, &af::constant::<u32>(0, true_idx.dims()), false).cast::<f32>();
let mask1 = af::eq(&true_idx, &af::constant::<u32>(1, true_idx.dims()), false).cast::<f32>();
// println!("true0={}, true1={}", af::sum_all(&mask0).0, af::sum_all(&mask1).0);


			raybnn::neural::network_snn_f32::state_space_forward_batch_snn(
				&arch_search.neural_network.netdata,
				&X_spk,
			
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
af_print!("{}",pred_rate);

af_print!("Y_scalar",Y_scalar);

loss_val = raybnn::optimal::snnloss_f32::mse_spike_rate_seq(&Yhat, &Y_seq,T);
println!("loss_val {}, i {}",loss_val,i);
	let (cm_train_vec, correct_train, total_train) = batch_confusion_and_accuracy(&pred_rate, &target_rate);
	let acc_train = correct_train as f32 / (total_train as f32);

	// // Pretty-print confusion matrix
	let c = output_size as u64;
	let cm_train_arr = af::Array::new(&cm_train_vec, af::Dim4::new(&[c, c, 1, 1]));
	af_print!("Train Confusion (rows=true, cols=pred)", cm_train_arr);
	println!("Epoch {}: loss = {:.6}, train acc = {:.2}%", i, loss_val, acc_train * 100.0);
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
				&X_spk,
				&arch_search.neural_network.network_params,
				&Z,
				&Q,
				&Y,
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

println!("||gW||2 = {:.6e}", af::norm(&gW, af::NormType::VECTOR_2, 1.0, 1.0));
println!("||gH||2 = {:.6e}", af::norm(&gH, af::NormType::VECTOR_2, 1.0, 1.0));


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

			if (input_size != max_input_size)
		{
			let mut tempRSSI = TOTAL_RSSI_TESTX.clone();

			for (key, value) in &TOTAL_RSSI_TESTX {
				let mut tempvec = value.clone();
				
				let veclen =  tempvec.len() as u64;
				let mut formatarr = arrayfire::Array::new(&tempvec, arrayfire::Dim4::new(&[max_input_size, veclen/max_input_size, 1, 1]));


				formatarr = arrayfire::rows(&formatarr, 0, (input_size-1) as i64);

				tempvec = vec!(f32::default();formatarr.elements());
				formatarr.host(&mut tempvec);

				tempRSSI.insert(key.clone(), tempvec);
			}

			RSSI_TESTX = tempRSSI;
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

for batch_idx in 0..(steps_per_epoch2)
		{
			println!("batch_idx{}",batch_idx);
			test_X =  arrayfire::Array::new(&RSSI_TESTX[&batch_idx], test_X_dims);  // go through batches 

			Y = arrayfire::Array::new(&RSSI_TESTY[&batch_idx], Y_dims);

			// arrayfire::set_slices(&mut X, &test_X, 0,(1-1) as i64);
			// af_print!("X:",test_X);
af::set_seed((1234) as u64); // or constant 1234 for identical spikes every epoch

let gain = 500.0f32; // start with 500~2000
let x_enc = delta_encode(&test_X, gain, true);
// let prob_K = repeat_each_traj_step(&x_enc, T);

// let X_spk = bernoulli_spike_encode(&prob_K, time_steps_total);

// let X_spk = delta_mod_encode_one_channel_abs(&x_enc, 0.0);

			raybnn::neural::network_snn_f32::state_space_forward_batch_snn(
				&arch_search.neural_network.netdata,
				&test_X,
				
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
			

			// // 	//Get Yhat
			
			



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