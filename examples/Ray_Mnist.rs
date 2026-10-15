

/*
Plot Figure 4 Ru ning RayBNN for the Alcala Dataset

Run the 10 Fold Testing for the Alcala Dataset in Figure 4
Note that CUDA has compile the kernels at runtime so the first run is slower. Tested on RTX 3090.

Generated files
./info_*.csv          Information about the training time and number of parameters
./test_act_*.csv      Actual test results
./test_pred_*.csv     Predicted test results
*/







extern crate arrayfire;
extern crate raybnn;
use std::collections::HashMap;
use nohash_hasher;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;





use raybnn::physics::update_f32::add_neuron_option_type;

fn argmax_row(row: &[f32]) -> usize {
    let mut max_idx = 0usize;
    let mut max_val = row[0];

    for i in 1..row.len() {
        if row[i] > max_val {
            max_val = row[i];
            max_idx = i;
        }
    }

    max_idx
}


#[allow(unused_must_use)]
fn main() {

	//Set CUDA and GPU Device 0
	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);


	
	//Run the code 11 times
	//The first run is slower because it has to compile CUDA/OpenCL kernels
	for fold in 0..1
	{



	arrayfire::set_seed(10 + fold as u64);

	arrayfire::sync(DEVICE);
	arrayfire::device_gc();
	arrayfire::sync(DEVICE);

	let conn_vec: Vec<u64> =        vec![30];
	let neuron_size_vec: Vec<u64> = vec![1200];
	let arg_input_vec: Vec<u64> =    vec![784];





	let dir_path = "/tmp/".to_string() ;

	//Set More Neural Network Parameters

let max_input_size: u64 = 784;
	let mut input_size: u64 = 784;


	let max_output_size: u64 = 10;
	let output_size: u64 = 10;

	// let max_neuron_size: u64 = 2000;
let max_neuron_size = 2500;


	//let mut batch_size: u64 = 64;
	let mut batch_size: u64 = 2500;

	let mut traj_size = 1;


	let mut proc_num = 3;


	let train_size: u64 = 60000;
	let test_size: u64 = 10000;
	let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);


	//Create Start Neural Network
	println!("Creating Network 40 db");

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







	//Transfer Learning Training Loop


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

		
		//Initialize Input Neurons

		/* 
		let input_neurons = raybnn::physics::initial_f32::golden_spiral_3D(
			arch_search.neural_network.netdata.sphere_rad+0.2,
			input_size
		);
		*/

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
		
	
		//Modify Input Dimensions

		if (prev_target_neuron_size !=  neuron_size_vec[ii])
		{

			let new_neuron_num =  neuron_size_vec[ii] - (arch_search.neural_network.neuron_idx.dims()[0] - input_size - output_size);
			println!("Iteration {}, new_neuron_num: {}", ii, new_neuron_num);

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




		//Modify training dataset

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






		let traj_steps = traj_size+proc_num-1;
println!("Computing training accuracy matrix...");

let eval_traj_size = 1;
let eval_traj_steps = eval_traj_size + proc_num - 1;

		let Z_dims = arrayfire::Dim4::new(&[arch_search.neural_network.netdata.neuron_size,batch_size,traj_steps,1]);
		let mut Z = arrayfire::constant::<f32>(0.0,Z_dims);
		let mut Q = arrayfire::constant::<f32>(0.0,Z_dims);



	 	let mut alpha0: f32 = 0.0001;

		//let mut alpha0: f32 = 0.001;





		let mut active_size = arch_search.neural_network.neuron_idx.dims()[0];
		let idxsel = arrayfire::rows(&arch_search.neural_network.neuron_idx, (active_size-output_size)  as i64, (active_size-1)  as i64);
		let Qslices: u64 = Q.dims()[2];




		let mut WRowIdxCOO = raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);





		let total_param_size = arch_search.neural_network.network_params.dims()[0];
		
		let mt_dims = arrayfire::Dim4::new(&[total_param_size,1,1,1]);
		let mut mt = arrayfire::constant::<f32>(0.0,mt_dims);
		let mut vt = arrayfire::constant::<f32>(0.0,mt_dims);
		let mut grad = arrayfire::constant::<f32>(0.0,mt_dims);



		let mut Wseqs = [arrayfire::Seq::default()];
		let mut Hseqs = [arrayfire::Seq::default()];
		let mut Aseqs = [arrayfire::Seq::default()];
		let mut Bseqs = [arrayfire::Seq::default()];
		let mut Cseqs = [arrayfire::Seq::default()];
		let mut Dseqs = [arrayfire::Seq::default()];
		let mut Eseqs = [arrayfire::Seq::default()];






		let X_dims = arrayfire::Dim4::new(&[input_size,batch_size,traj_steps,1]);
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
		let mut Aidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
		let mut Bidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
		let mut Cidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
		let mut Didxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
		let mut Eidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
		let mut combidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();


		let mut dAseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();
		let mut dBseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();
		let mut dCseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();
		let mut dDseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();
		let mut dEseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();




		let WValuesdims0 =  arch_search.neural_network.WColIdx.dims()[0];

		let network_paramsdims0 =  arch_search.neural_network.network_params.dims()[0];
	
		let Hdims0 =  (network_paramsdims0 -  WValuesdims0)/6; 
	


		//Create backward graph

		raybnn::graph::path_f32::find_path_backward_group2(
			&arch_search.neural_network.netdata,
			traj_steps,
			traj_size, 
			&WRowIdxCOO,
			&arch_search.neural_network.WColIdx,
			&arch_search.neural_network.neuron_idx,
		
		
			WValuesdims0,
			Hdims0,
			Hdims0,
			Hdims0,
			Hdims0,
			Hdims0,
			Hdims0,
		
		
			
			&mut idxsel_out,
			&mut valsel_out,

			&mut cvec_out,
			&mut dXsel_out,

			&mut nrows_out,
			&mut sparseval_out,
			&mut sparserow_out,
			&mut sparsecol_out,
			
		
		
			&mut Hidxsel_out,
			&mut Aidxsel_out,
			&mut Bidxsel_out,
			&mut Cidxsel_out,
			&mut Didxsel_out,
			&mut Eidxsel_out,
			&mut combidxsel_out,
		
		
		
		

			&mut dAseqs_out,
			&mut dBseqs_out,
			&mut dCseqs_out,
			&mut dDseqs_out,
			&mut dEseqs_out,
		
		
		
			
			&mut Wseqs,
			&mut Hseqs,
			&mut Aseqs,
			&mut Bseqs,
			&mut Cseqs,
			&mut Dseqs,
			&mut Eseqs
		
		);



		



		let train_X_dims = arrayfire::Dim4::new(&[input_size,batch_size,traj_size,1]);
		let mut train_X =  arrayfire::constant::<f32>(0.0,temp_dims);
		let Y_dims = arrayfire::Dim4::new(&[output_size,batch_size,traj_size,1]);
		let mut Y =  arrayfire::constant::<f32>(0.0,temp_dims);
		let mut batch_idx = 0;
		let epoch_num = (train_size/ (batch_size*traj_size) )-1;







		let mut max_epoch = 2000;




		let mut loss_val = 100000.0;
		let mut loss_val2 = 100000.0;
		let mut prev_loss = 100000.0;

		arrayfire::device_gc();
		arrayfire::sync(DEVICE);

		let start_time = std::time::Instant::now();

		//Training Loop
		
		for i in 0..max_epoch
		{
			

			batch_idx = i % epoch_num;

			train_X =  arrayfire::Array::new(&RSSI_TRAINX[&batch_idx], train_X_dims);

			Y = arrayfire::Array::new(&RSSI_TRAINY[&batch_idx], Y_dims);
			arrayfire::set_slices(&mut X, &train_X, 0,(traj_size-1) as i64);

			
			//Forward pass
			let mut confusion_matrix = vec![vec![0u64; 10]; 10];

let mut correct: u64 = 0;
let mut total: u64 = 0;

			raybnn::neural::network_f32::state_space_forward_batch(
				&arch_search.neural_network.netdata,
				&X,
				
				&arch_search.neural_network.WRowIdxCSR,
				&arch_search.neural_network.WColIdx,
			
			
				&Wseqs,
				&Hseqs,
				&Aseqs,
				&Bseqs,
				&Cseqs,
				&Dseqs,
				&Eseqs,
				&arch_search.neural_network.network_params,
			
			
			
				&mut Z,
				&mut Q
			);




			//Get Yhat
			let mut idxrs = arrayfire::Indexer::default();
			let seq1 = arrayfire::Seq::new(0.0f32, (batch_size-1) as f32, 1.0);
			let seq2 = arrayfire::Seq::new((proc_num-1) as f32, (Qslices-1) as f32, 1.0);
			idxrs.set_index(&idxsel, 0, None);
			idxrs.set_index(&seq1, 1, None);
			idxrs.set_index(&seq2, 2, None);
			let Yhat = arrayfire::index_gen(&Q, idxrs);
			loss_val = raybnn::optimal::loss_f32::RMSE(&Yhat,&Y);
   let mut yhat_vec = vec![0f32; Yhat.elements()];
    let mut y_vec = vec![0f32; Y.elements()];

    Yhat.host(&mut yhat_vec);
    Y.host(&mut y_vec);
			

			if i == 0
			{
				loss_val2 = loss_val;
			}
			else
			{
				loss_val2 = 0.99*loss_val2 + 0.01*loss_val;
			}


			//Backward pass
			// println!("Z dims: {:?}", Z.dims());
			raybnn::neural::network_f32::state_space_backward_group2(
				&arch_search.neural_network.netdata,
				&X,
			
			

				&arch_search.neural_network.network_params,
			
			
			
			
				&Z,
				&Q,
				&Y,
				raybnn::optimal::loss_f32::MSE_grad,
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
				&Aidxsel_out,
				&Bidxsel_out,
				&Cidxsel_out,
				&Didxsel_out,
				&Eidxsel_out,
				&combidxsel_out,



				&dAseqs_out,
				&dBseqs_out,
				&dCseqs_out,
				&dDseqs_out,
				&dEseqs_out,
			

				
				&mut grad,
			);
			


			// Right after backprop:
		// let grad_norm = arrayfire::sum_all(&arrayfire::abs(&grad)).0;
		// 	println!("Epoch {}, grad_norm = {}", i, grad_norm);

			//Update weights with ADAM

			raybnn::optimal::gd_f32::adam(
				0.9
				,0.999
				,&mut grad
				,&mut mt
				,&mut vt
			);
	



			alpha0 = (0.002/(loss_val  +  1e-10)).min(0.007);
		//	alpha0 = (0.002/(loss_val  + 1e-10)).min(0.007);


			if ii == 11
			{
				alpha0 = alpha0.min(0.005);
			}

			arch_search.neural_network.network_params = arch_search.neural_network.network_params + (alpha0*-grad.clone());

			//    if (loss_val < 0.00035) && (depend == 100){
			// 	depend = 10;
       		//  let elapsed_time = start_time.elapsed().as_secs_f32();
			// 	println!("===================================================================");
        	// 	println!("Epoch {}: loss = {}, elapsed time = {}s", i, loss_val, elapsed_time);
    		// }


			if i % 50 == 0 {
				let elapsed_time = start_time.elapsed().as_secs_f32();
				println!("Epoch {}: Test RMSE = {}, Time elapsed = {}s", i, loss_val, elapsed_time);
                
                 // ArrayFire layout: output_size x batch_size
    for sample_idx in 0..batch_size as usize {
        let mut pred_row = vec![0f32; output_size as usize];
        let mut true_row = vec![0f32; output_size as usize];

        for class_idx in 0..output_size as usize {
            let idx = class_idx + sample_idx * output_size as usize;

            pred_row[class_idx] = yhat_vec[idx];
            true_row[class_idx] = y_vec[idx];
        }

        let pred_class = argmax_row(&pred_row);
        let true_class = argmax_row(&true_row);

        confusion_matrix[true_class][pred_class] += 1;

        if pred_class == true_class {
            correct += 1;
        }

        total += 1;
    }
    let train_accuracy = correct as f32 / total as f32;

println!("Training Accuracy = {:.4}", train_accuracy);
println!("Training Confusion Matrix:");
println!("Rows = actual labels, columns = predicted labels");

for i in 0..10 {
    println!("{:?}", confusion_matrix[i]);
}
			}
			



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
		println!("loss_val {}", loss_val);

		let elapsed: f32 = start_time.elapsed().as_secs_f32();
		println!("elapsed {}", elapsed);

		traj_size = 1;

		arrayfire::device_gc();

		//Change Input Dimensions of Testing dataset

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




		let traj_steps = traj_size+proc_num-1;


		let Z_dims = arrayfire::Dim4::new(&[arch_search.neural_network.netdata.neuron_size,batch_size,traj_steps,1]);
		let mut Z = arrayfire::constant::<f32>(0.0,Z_dims);
		let mut Q = arrayfire::constant::<f32>(0.0,Z_dims);



		let Qslices: u64 = Q.dims()[2];


		let X_dims = arrayfire::Dim4::new(&[input_size,batch_size,traj_steps,1]);
		let mut X = arrayfire::constant::<f32>(0.0,X_dims);



		let test_X_dims = arrayfire::Dim4::new(&[input_size,batch_size,traj_size,1]);
		let mut test_X =  arrayfire::constant::<f32>(0.0,temp_dims);
		let Y_dims = arrayfire::Dim4::new(&[output_size,batch_size,traj_size,1]);
		let mut Y =  arrayfire::constant::<f32>(0.0,temp_dims);
		let mut batch_idx = 0;
		let epoch_num = (test_size/ (batch_size*traj_size) );



		let temp_Y_dims = arrayfire::Dim4::new(&[1,output_size,1,1]);

		let mut Yhat_arr = arrayfire::constant::<f32>(0.0,temp_Y_dims);
		let mut Y_arr = arrayfire::constant::<f32>(0.0,temp_Y_dims);


		let newdims = arrayfire::Dim4::new(&[output_size,batch_size*traj_size,1,1]);


		//Compute testing dataset

		arrayfire::device_gc();
		arrayfire::sync(DEVICE);
        
		for batch_idx in 0..epoch_num
		{

			test_X =  arrayfire::Array::new(&RSSI_TESTX[&batch_idx], test_X_dims);

			Y = arrayfire::Array::new(&RSSI_TESTY[&batch_idx], Y_dims);
			arrayfire::set_slices(&mut X, &test_X, 0,(traj_size-1) as i64);



			raybnn::neural::network_f32::state_space_forward_batch(
				&arch_search.neural_network.netdata,
				&X,
				
				&arch_search.neural_network.WRowIdxCSR,
				&arch_search.neural_network.WColIdx,
			
			
				&Wseqs,
				&Hseqs,
				&Aseqs,
				&Bseqs,
				&Cseqs,
				&Dseqs,
				&Eseqs,
				&arch_search.neural_network.network_params,
			
			
			
				&mut Z,
				&mut Q
			);


			//Get Yhat
			let mut idxrs = arrayfire::Indexer::default();
			let seq1 = arrayfire::Seq::new(0.0f32, (batch_size-1) as f32, 1.0);
			let seq2 = arrayfire::Seq::new((proc_num-1) as f32, (Qslices-1) as f32, 1.0);
			idxrs.set_index(&idxsel, 0, None);
			idxrs.set_index(&seq1, 1, None);
			idxrs.set_index(&seq2, 2, None);
			let mut Yhat = arrayfire::index_gen(&Q, idxrs);
			


			Yhat = arrayfire::flat(&Yhat);
			Y = arrayfire::flat(&Y);


			Yhat = arrayfire::moddims(&Yhat, newdims);
			Y = arrayfire::moddims(&Y, newdims);


		
			Yhat = arrayfire::transpose(&Yhat, false);
			Y = arrayfire::transpose(&Y, false);
		

			Yhat_arr = arrayfire::join(0, &Yhat_arr, &Yhat);
			Y_arr = arrayfire::join(0, &Y_arr, &Y);
		}


		Yhat_arr = arrayfire::rows(&Yhat_arr,1,(Yhat_arr.dims()[0]-1) as i64);
		Y_arr = arrayfire::rows(&Y_arr,1,(Y_arr.dims()[0]-1) as i64);

		// =====================================================
// Compute TEST accuracy
// =====================================================

let mut test_yhat_vec = vec![0f32; Yhat_arr.elements()];
let mut test_y_vec = vec![0f32; Y_arr.elements()];

Yhat_arr.host(&mut test_yhat_vec);
Y_arr.host(&mut test_y_vec);

// After transpose, Yhat_arr and Y_arr should be:
// rows = samples, columns = classes
let num_test_samples = Yhat_arr.dims()[0] as usize;
let num_classes = output_size as usize;

let mut test_correct: u64 = 0;
let mut test_total: u64 = 0;

let mut test_confusion_matrix = vec![vec![0u64; 10]; 10];

for sample_idx in 0..num_test_samples {
    let mut pred_row = vec![0f32; num_classes];
    let mut true_row = vec![0f32; num_classes];

    for class_idx in 0..num_classes {
        let idx = sample_idx + class_idx * num_test_samples;

        pred_row[class_idx] = test_yhat_vec[idx];
        true_row[class_idx] = test_y_vec[idx];
    }

    let pred_class = argmax_row(&pred_row);
    let true_class = argmax_row(&true_row);

    test_confusion_matrix[true_class][pred_class] += 1;

    if pred_class == true_class {
        test_correct += 1;
    }

    test_total += 1;
}

let test_accuracy = test_correct as f32 / test_total as f32;

println!("Test Accuracy = {:.4}", test_accuracy);
println!("Test Accuracy = {:.2}%", test_accuracy * 100.0);

println!("Test Confusion Matrix:");
println!("Rows = actual labels, columns = predicted labels");

for i in 0..10 {
    println!("{:?}", test_confusion_matrix[i]);
}
		arrayfire::device_gc();
		arrayfire::sync(DEVICE);


		let totalparam = arch_search.neural_network.network_params.dims()[0] as f32;

		let v0: [f32; 2] = [elapsed, totalparam];
		let info = arrayfire::Array::new(&v0, arrayfire::Dim4::new(&[2, 1, 1, 1]));


		//Save dataset


		// let save_filename = format!("info_{}_{}.csv",fold,input_size);
		// raybnn::export::dataloader_f32::write_arr_to_csv(
		// 	&save_filename,
		// 	&info
		// );

		

		// let save_filename = format!("test_pred_{}_{}.csv",fold,input_size);
		// raybnn::export::dataloader_f32::write_arr_to_csv(
		// 	&save_filename,
		// 	&Yhat_arr
		// );



		// let save_filename = format!("test_act_{}_{}.csv",fold,input_size);
		// raybnn::export::dataloader_f32::write_arr_to_csv(
		// 	&save_filename,
		// 	&Y_arr
		// );


		








	}

	}
}
