/*
Test RayBNN-SNN model on Mnist Dataset 
*/
extern crate arrayfire;
extern crate raybnn;
use std::collections::HashMap;
use nohash_hasher;
use rand::Rng;                          
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;
use raybnn::interface::regularization_f32::rcs_regularization_f32 as rcs;
use arrayfire::{af_print, print};
use arrayfire::{Seq, index,neq, count_all, imax,eq,Array};
use std::fs::File;
use std::io::{BufWriter, Write};
use rayon::prelude::*;
use raybnn::{physics::update_f32::add_neuron_option_type, optimal::loss_f32::{MSE, MSE_grad}};
use rand::SeedableRng;
use arrayfire as af;

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


#[allow(unused_must_use)]
fn main() {
let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(10);

	//Set CUDA and GPU Device 0
	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);

	arrayfire::set_seed(10  as u64);

	arrayfire::sync(DEVICE);
	arrayfire::device_gc();
	arrayfire::sync(DEVICE);
	

	// let conn_vec: Vec<u64> =        vec![60];
	//  let neuron_size_vec: Vec<u64> =  vec![800];
	// let arg_input_vec: Vec<u64> =    vec![1000];
	

	let conn_vec: Vec<u64> =        vec![30];
	let neuron_size_vec: Vec<u64> = vec![1200];
	let arg_input_vec: Vec<u64> =    vec![784];




	let dir_path = "/tmp/".to_string() ;
	//Set More Neural Network Parameters

	let max_input_size: u64 = 784;
	let mut input_size: u64 = 784;


	let max_output_size: u64 = 10;
	let output_size: u64 = 10;

    let max_neuron_size = input_size + neuron_size_vec[0] + output_size; // 


	//let mut batch_size: u64 = 64;
	let mut batch_size: u64 = 2500;

	let mut traj_size = 1;


	let mut proc_num = 3;


	let train_size: u64 = 60000;
	let test_size: u64 = 10000;
	let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);



	//Create Start Neural Network
	println!("Creating Network");

	let mut arch_search = raybnn::interface::automatic_f32::create_start_archtecture(

		input_size,
		max_input_size,

		output_size,
		max_output_size,
		max_neuron_size,
		batch_size,
		1,

	
		&dir_path
	);


	arch_search.neural_network.netdata.proc_num = proc_num;
	



	//Load dataset
	println!("Load Dataset");


	

    let TOTAL_RSSI_TESTX = raybnn::export::dataloader_f32::file_to_hash_cpu(
		"./test_data/MNIST_test_X.csv",
		max_input_size,
		batch_size*traj_size
	);


	let RSSI_TESTY = raybnn::export::dataloader_f32::file_to_hash_cpu(
		"./test_data/MNIST_test_Y1h.csv",
		output_size,
		batch_size*traj_size
	);
	arrayfire::sync(DEVICE);


	let mut RSSI_TESTX = TOTAL_RSSI_TESTX.clone();



	let TOTAL_RSSI_TRAINX = raybnn::export::dataloader_f32::file_to_hash_cpu(
    	"./test_data/MNIST_train_X.csv",
    	max_input_size,
		batch_size*traj_size
    );


	let RSSI_TRAINY = raybnn::export::dataloader_f32::file_to_hash_cpu(
    	"./test_data/MNIST_train_Y1h.csv",
    	output_size,
		batch_size*traj_size
    );
	arrayfire::sync(DEVICE);
	
	let mut RSSI_TRAINX = TOTAL_RSSI_TRAINX.clone();


    let mut prev_target_neuron_size = 0;







	for ii in 0..arg_input_vec.len()
	{

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

		let T: u64 = 20; // this sets the time steps 
		let traj_steps = 1+proc_num-1;
		let time_steps_total: u64 = traj_size * T;  // total SNN steps per trajectory
		let Z_dims = arrayfire::Dim4::new(&[arch_search.neural_network.netdata.neuron_size,batch_size*1,time_steps_total,1]);
		// [arch_search.neural_network.netdata.neuron_size,batch_size,1,1]
		let mut Z = arrayfire::constant::<f32>(0.0,Z_dims);  // membrane potential
		let mut Q = arrayfire::constant::<f32>(0.0,Z_dims);  // spikes



		let mut alpha0: f32 = 0.0001;

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
let mut Y = arrayfire::constant::<f32>(0.0, Y_dims);

let mut batch_idx = 0;
let epoch_num: u64 = (train_size / (batch_size * traj_size)) - 1;



		let mut max_epoch = 1200;


		let mut loss_val = 100000.0;
		let mut loss_val2 = 100000.0;
		let mut prev_loss = 100000.0;

		arrayfire::device_gc();
		arrayfire::sync(DEVICE);

		let start_time = std::time::Instant::now();
		let out_cols = arrayfire::Dim4::new(&[1, 3, 1, 1]); // columns: pred,label,rate
		let mut pred_true_rate = af::constant::<f32>(0.0, out_cols);
		// let mut rng = rand::thread_rng();
		for i in 0..max_epoch
		{

		// let batch_idx = rng.gen_range(0..epoch_num); // lightweight shuffle-by-sampling
			batch_idx = i % epoch_num;

			train_X =  arrayfire::Array::new(&RSSI_TRAINX[&batch_idx], train_X_dims);
			Y = arrayfire::Array::new(&RSSI_TRAINY[&batch_idx], Y_dims);
			
			arrayfire::set_slices(&mut X, &train_X, 0,(1-1) as i64);

			let tiled = af::tile(&train_X, af::Dim4::new(&[1, 1, T, 1])); // [in, batch, T, 1]
			let full_time = [arrayfire::Seq::<f32>::default(),
            arrayfire::Seq::<f32>::default(),
            arrayfire::Seq::new(0.0, (T-1) as f32, 1.0)];
			af::assign_seq(&mut X, &full_time, &tiled);



			let X_spk: arrayfire::Array<f32> = bernoulli_spike_encode(&X, T);



			raybnn::neural::network_snn_f32::state_space_forward_batch_snn(
				&arch_search.neural_network.netdata,
				&X_spk,
				
				&arch_search.neural_network.WRowIdxCSR,
				&arch_search.neural_network.WColIdx,
			
				&Wseqs,
				&Hseqs,
				&arch_search.neural_network.network_params,
				T,
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

let target_rate = af::max(&Y, 2);                 // target value

let y_one: arrayfire::Array<f32> = arrayfire::max(&Y, 2);                // one hot version to integer

let (cm_train_vec, correct_train, total_train) = batch_confusion_and_accuracy(&pred_rate, &y_one);
let acc_train = correct_train as f32 / (total_train as f32);

	let c = output_size as u64;
	let cm_train_arr = af::Array::new(&cm_train_vec, af::Dim4::new(&[c, c, 1, 1]));
	if (i%24 == 0){
	af_print!("Train Confusion (rows=true, cols=pred)", cm_train_arr);
	println!("Epoch {}: loss = {:.6}, train acc = {:.2}%", i, loss_val, acc_train * 100.0);
	let (total_spikes, _) = af::sum_all::<f32>(&Q);
let total_entries = Q.elements() as f32;
let firing_rate = total_spikes / total_entries;
println!("Global firing rate = {}", firing_rate);
println!("Global sparsity = {}", 1.0 - firing_rate);
let num_edges = arch_search.neural_network.WColIdx.dims()[0];
println!("Number of nonzero edges = {}", num_edges);

	}
	loss_val = raybnn::optimal::snnloss_f32::mse_spike_rate(&Yhat, &y_one);

			println!("loss_val {}, i {}",loss_val,i);


			if i == 0
			{
				loss_val2 = loss_val;
			}
			else
			{
				loss_val2 = 0.99*loss_val2 + 0.01*loss_val;
			}

			raybnn::neural::network_snn_f32::state_space_backward_batch2(
				&arch_search.neural_network.netdata,
				&X_spk,
		
				&arch_search.neural_network.network_params,
				&Z,
				&Q,
				&Y,
				raybnn::optimal::snnloss_f32::mse_spike_rate_grad,
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
			);


			raybnn::optimal::gd_f32::adam(
				0.9
				,0.999
				,&mut grad
				,&mut mt
				,&mut vt
			);
		
			alpha0 = (0.002/(loss_val  +  1e-10)).min(0.007);

			arch_search.neural_network.network_params = arch_search.neural_network.network_params + (alpha0*-grad.clone());


			if (i > 300) && ((i % 50) == 0)
			{
				if (prev_loss-loss_val).abs()/loss_val < 0.007
				{
					break;
				}

				prev_loss = loss_val;
			}
	
		}

		
		
		arrayfire::sync(DEVICE);
		println!("loss_val {}", loss_val);
		

		let elapsed: f32 = start_time.elapsed().as_secs_f32();
		println!("elapsed {}", elapsed);

 		// arrayfire::device_gc();

		// //Change Input Dimensions of Testing dataset

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

		let traj_steps = 1+proc_num-1;


		let Z_dims = arrayfire::Dim4::new(&[arch_search.neural_network.netdata.neuron_size,batch_size*1,T,1]);
		let mut Z = arrayfire::constant::<f32>(0.0,Z_dims);
		let mut Q = arrayfire::constant::<f32>(0.0,Z_dims);



		let Qslices: u64 = Q.dims()[2];


		let X_dims = arrayfire::Dim4::new(&[input_size,batch_size*1,T,1]);
		let mut X = arrayfire::constant::<f32>(0.0,X_dims);



		let test_X_dims = arrayfire::Dim4::new(&[input_size,batch_size,1,1]);
		let mut test_X =  arrayfire::constant::<f32>(0.0,temp_dims);
		let Y_dims = arrayfire::Dim4::new(&[output_size,batch_size,1,1]);
		let mut Y =  arrayfire::constant::<f32>(0.0,temp_dims);
		let mut batch_idx = 0;
		let epoch_num = (test_size/ (batch_size*1) );



		let temp_Y_dims = arrayfire::Dim4::new(&[1,output_size,1,1]);

		let mut Yhat_arr = arrayfire::constant::<f32>(0.0,temp_Y_dims);
		let mut Y_arr = arrayfire::constant::<f32>(0.0,temp_Y_dims);


		let newdims: arrayfire::Dim4 = arrayfire::Dim4::new(&[output_size,batch_size*1,T,1]);

		arrayfire::device_gc();
		arrayfire::sync(DEVICE);
		let c = output_size as usize;
		let mut test_cm: Vec<u32> = vec![0; c * c];
		let mut test_correct_sum: u64 = 0;
		let mut test_total_sum:   u64 = 0;
		let mut all_preds: Vec<f32>  = Vec::new(); // predicted class index
		let mut all_trues: Vec<f32>  = Vec::new(); // true class index


		for batch_idx in 0..epoch_num
		{
			test_X =  arrayfire::Array::new(&RSSI_TESTX[&batch_idx], test_X_dims);  // go through batches 

			Y = arrayfire::Array::new(&RSSI_TESTY[&batch_idx], Y_dims);
			arrayfire::set_slices(&mut X, &test_X, 0,(1-1) as i64);

			let tiled = af::tile(&test_X, af::Dim4::new(&[1, 1, T, 1])); // [in, batch, T, 1]
			let full_time = [arrayfire::Seq::<f32>::default(),
                 arrayfire::Seq::<f32>::default(),
                 arrayfire::Seq::new(0.0, (T-1) as f32, 1.0)];
			af::assign_seq(&mut X, &full_time, &tiled);
				let X_spk: arrayfire::Array<f32> = bernoulli_spike_encode(&X, T);


			raybnn::neural::network_snn_f32::state_space_forward_batch_snn(
				&arch_search.neural_network.netdata,
				&X_spk,
				
				&arch_search.neural_network.WRowIdxCSR,
				&arch_search.neural_network.WColIdx,
			
				&Wseqs,
				&Hseqs,
				&arch_search.neural_network.network_params,
				T,
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


