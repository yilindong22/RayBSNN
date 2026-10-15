extern crate arrayfire;
use crate::neural::network_f32::network_metadata_type;
use std::collections::HashMap;
use nohash_hasher;

use rayon::array;
use rayon::prelude::*;

use std::fs::File;
use std::io::Write;

use std::fs;
use nohash_hasher::IntMap;          // ← make IntMap available


use crate::export::dataloader_i32::vec_to_str as vec_to_str_i32;
use crate::export::dataloader_i32::str_to_vec as str_to_vec_i32;


use crate::export::dataloader_u64::vec_cpu_to_str as vec_cpu_to_str_u64;
use crate::export::dataloader_u64::str_to_vec_cpu as str_to_vec_cpu_u64;


use crate::neural::network_f32::neural_network_type;


use crate::neural::network_f32::create_nullnetdata;


use std::io::{self, prelude::*, BufReader};




pub fn vec_to_str(
	invec: &arrayfire::Array<f32>
	) -> String  {


	let mut vec0 = vec!(f32::default();invec.elements());
	invec.host(&mut vec0);
	let mut s0 = format!("{:?}",vec0);
	s0 = s0.replace("[", "");
	s0 = s0.replace("]", "");
	s0 = s0.replace(" ", "");

	s0
}


pub fn vec_cpu_to_str(
	invec: &Vec<f32>
	) -> String  {

	let mut s0 = format!("{:?}",invec.clone());
	s0 = s0.replace("[", "");
	s0 = s0.replace("]", "");
	s0 = s0.replace(" ", "");

	s0
}





pub fn str_to_vec_cpu(
	instr: &str
) -> Vec<f32>  {

	let mut vecf32: Vec<f32> = Vec::new();


	let mut newline = instr.replace("\n", "");
	newline = newline.replace(" ", "");

	if newline.len() > 0
	{
		let strvec: Vec<&str> = newline.split(",").collect();
		let ssize: u64 = strvec.len() as u64;

		
		for i in 0u64..ssize
		{
			let value:f32 = strvec[i as usize].parse::<f32>().unwrap();
			vecf32.push(value);
		}
	}

	vecf32
}




pub fn str_to_vec(
	instr: &str
	) -> arrayfire::Array<f32>  {

	let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let mut outarr = arrayfire::constant::<f32>(0.0,temp_dims);



	let mut newline = instr.replace("\n", "");
	newline = newline.replace(" ", "");

	let strvec: Vec<&str> = newline.split(",").collect();
	let ssize: u64 = strvec.len() as u64;

	if (ssize > 1)
	{
		let mut vecf32: Vec<f32> = Vec::new();
		for i in 0u64..ssize
		{
			let value:f32 = strvec[i as usize].parse::<f32>().unwrap();
			vecf32.push(value);
		}

		let new_dims = arrayfire::Dim4::new(&[ssize,1,1,1]);
		outarr = arrayfire::Array::new(&vecf32, new_dims);
	}

	outarr
}





pub fn str_to_matrix(
	instr: &str,
	dims: arrayfire::Dim4
	) -> arrayfire::Array<f32>  {

	let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let mut outarr = arrayfire::constant::<f32>(0.0,temp_dims);



	let mut newline = instr.replace("\n", "");
	newline = newline.replace(" ", "");

	let strvec: Vec<&str> = newline.split(",").collect();
	let ssize: u64 = strvec.len() as u64;

	if (ssize > 1)
	{
		let mut vecf32: Vec<f32> = Vec::new();
		for i in 0u64..ssize
		{
			let value:f32 = strvec[i as usize].parse::<f32>().unwrap();
			vecf32.push(value);
		}


		outarr = arrayfire::Array::new(&vecf32, dims);
	}

	outarr
}







pub fn extract_file_info2(filepath: &str) -> Vec<f32>
{
    let targetstr = filepath.clone().replace(".csv", "");

    let strsplit:Vec<&str> = targetstr.split('_').collect();
    
    let mut outdata: Vec<f32> = Vec::new();
    
    let mut idx = 0;
    
    let mut parse_state = 0;
    for tmp in strsplit
    {
    
        if tmp.contains("cube") && (parse_state == 0)
        {
            parse_state = 1;
        }
        
        if (parse_state == 1)
        {
            if idx == 1
            {
                //println!("data {}",tmp);
                let elem = tmp.clone().parse::<f32>().unwrap();
                outdata.push(elem);
				break;
            }
            
            idx = idx + 1;
        }
        
        
    }
    
    
    outdata
}









pub fn file_to_vec_cpu2(filename: &str) -> Vec<f32> {
    let contents = std::fs::read_to_string(filename).expect("error");
    let mut out = Vec::<f32>::new();

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        // split by comma, parse
        for s in line.split(',') {
            let s = s.trim();
            if s.is_empty() { continue; }
            out.push(s.parse::<f32>().unwrap());
        }
    }
    out
}



pub fn file_to_vec_cpu(
	filename: &str
) -> Vec<f32>  {
	let contents = fs::read_to_string(filename).expect("error");

	contents.par_split('\n').map(str_to_vec_cpu ).flatten_iter().collect()
}






pub fn file_to_matrix(
	filename: &str,
	dims: arrayfire::Dim4
	) -> arrayfire::Array<f32>  {

	let arr = file_to_vec_cpu(filename);

	let dims2 = arrayfire::Dim4::new(&[dims[1], dims[0], 1, 1]);
	let mut outarr = arrayfire::Array::new(&arr, dims2);


	arrayfire::transpose(&outarr,false)
}










pub fn file_to_hash_cpu(
	filename: &str,
	sample_size: u64,
	batch_size: u64
	) -> nohash_hasher::IntMap<u64, Vec<f32> >  {


	let arr = file_to_vec_cpu(filename);

	let arr_size = arr.len() as u64;
	let item_num = (arr_size/(sample_size*batch_size));

	let mut lookup: nohash_hasher::IntMap<u64, Vec<f32> >  = nohash_hasher::IntMap::default();
	let mut start:usize = 0;
	let mut end:usize = 0;

	for i in 0..item_num
	{
		start = (i*(sample_size*batch_size)) as usize;
		end = ((i+1)*(sample_size*batch_size)) as usize;
		// println!("Key: {}, Value: {:#?}", i, (&arr[start..end]).to_vec());

		lookup.insert(i, (&arr[start..end]).to_vec() );
	}

	lookup
}





pub fn compute_stats(
	input_size: u64,
	dataset: &nohash_hasher::IntMap<u64, Vec<f32> >,

	mean: &mut arrayfire::Array<f32>,
	stdev: &mut arrayfire::Array<f32>,
	)
	{


	let mut tempvec = dataset[&0].clone();
	
	let mut veclen =  tempvec.len() as u64;
	let mut formatarr = arrayfire::Array::new(&tempvec, arrayfire::Dim4::new(&[input_size, veclen/input_size, 1, 1]));

	*mean = arrayfire::mean(&formatarr, 1);
	*stdev = arrayfire::stdev_v2(&formatarr,arrayfire::VarianceBias::SAMPLE, 1);


	for (key, value) in dataset {
		tempvec = value.clone();
		
		veclen =  tempvec.len() as u64;
		formatarr = arrayfire::Array::new(&tempvec, arrayfire::Dim4::new(&[input_size, veclen/input_size, 1, 1]));

		let tempmean = arrayfire::mean(&formatarr, 1);
		let tempstdev = arrayfire::stdev_v2(&formatarr,arrayfire::VarianceBias::SAMPLE, 1);

		*mean = (mean.clone() + tempmean)/ 2.0f32;
		*stdev = (stdev.clone() + tempstdev)/ 2.0f32;
	}


	
	let eps = 0.001f32;

	//  (eps > stdev )
	let CMP1 = arrayfire::gt(&eps, stdev, false);

	let selidx = arrayfire::locate(&CMP1);	

	if selidx.dims()[0] > 0
	{
		let mut ones = arrayfire::constant::<f32>(1.0,selidx.dims());

		

		let mut idxrs2 = arrayfire::Indexer::default();
		idxrs2.set_index(&selidx, 0, None);
		arrayfire::assign_gen(stdev, &idxrs2, &ones);
	}

}









pub fn normalize_dataset(
	input_size: u64,


	mean: &arrayfire::Array<f32>,
	stdev: &arrayfire::Array<f32>,


	dataset: &nohash_hasher::IntMap<u64, Vec<f32> >,
	)  ->  nohash_hasher::IntMap<u64, Vec<f32> >
	{

	let mut tempdata = dataset.clone();


	for (key, value) in dataset {
		let tempvec = value.clone();
		
		let veclen =  tempvec.len() as u64;
		let mut formatarr = arrayfire::Array::new(&tempvec, arrayfire::Dim4::new(&[input_size, veclen/input_size, 1, 1]));

		//formatarr = (formatarr - mean)/stdev;
		formatarr = arrayfire::sub(&formatarr , mean, true);
		formatarr = arrayfire::div(&formatarr , stdev, true);


		let mut tempvec = vec!(f32::default();formatarr.elements());
		formatarr.host(&mut tempvec);

		tempdata.insert(key.clone(), tempvec);
	}

	tempdata
}










pub fn largefile_to_hash_cpu(
	filename: &str,
	) -> nohash_hasher::IntMap<u64, Vec<f32> >  {


	let mut lookup: nohash_hasher::IntMap<u64, Vec<f32> >  = nohash_hasher::IntMap::default();

	let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

	let mut idx = 0;
    for line in reader.lines() {
		let tempstr = line.unwrap().clone();
        let outvec = str_to_vec_cpu(&tempstr);
		lookup.insert(idx, outvec.clone());
		idx = idx + 1;
    }


	lookup

}






pub fn shuffle_hash_cpu(
	dataX: &mut nohash_hasher::IntMap<u64, Vec<f32> > ,
	dataY: &mut nohash_hasher::IntMap<u64, Vec<f32> >
	)
	{

	let totalsize = dataX.keys().len() as u64;


	let randarr_dims = arrayfire::Dim4::new(&[totalsize,1,1,1]);

	let randarr= arrayfire::randu::<f32>(randarr_dims);

	let (_, idx) = arrayfire::sort_index(&randarr, 0, false);





	let mut idxvec = vec!(u32::default();idx.elements());
	idx.host(&mut idxvec);







	let mut newX: nohash_hasher::IntMap<u64, Vec<f32> >  = nohash_hasher::IntMap::default();
	let mut newY: nohash_hasher::IntMap<u64, Vec<f32> >  = nohash_hasher::IntMap::default();


	let mut newidx: u64 = 0;
	for i in 0..totalsize
	{
		newidx = idxvec[i as usize] as u64;

		newX.insert(newidx, (*dataX)[&i].clone() );
		newY.insert(newidx, (*dataY)[&i].clone() );
	}

	*dataX =  newX;
	*dataY =  newY;

}




















pub fn format_vec_to_matrix(
	input: &Vec<f32>,
	dims: arrayfire::Dim4
	) -> arrayfire::Array<f32>  {

	let dims2 = arrayfire::Dim4::new(&[dims[1], dims[0], 1, 1]);
	let outarr = arrayfire::Array::new(input, dims2);


	arrayfire::transpose(&outarr,false)
}









pub fn file_to_hash_matrix(
	filename: &str,
	sample_size: u64,
	batch_size: u64,
	dims: arrayfire::Dim4
	) -> nohash_hasher::IntMap<u64, arrayfire::Array<f32>  >  {


	let mut lookup2: nohash_hasher::IntMap<u64, arrayfire::Array<f32>  >   = nohash_hasher::IntMap::default();

	let lookup = file_to_hash_cpu(
		filename,
		sample_size,
		batch_size);

	let item_num = lookup.len() as u64;

    let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	let mut temparr = arrayfire::constant::<f32>(0.0, temp_dims);
	for i in 0..item_num
	{
		temparr = arrayfire::Array::new(&lookup[&i], dims);

		lookup2.insert(i, temparr);
	}


	lookup2
}









pub fn write_arr_to_csv(
	filename: &str,
	arr: &arrayfire::Array<f32>
	)
{




	let mut wtr0: Vec<String> = Vec::new();

	let item_num = arr.dims()[0] as i64;

	for i in 0..item_num
	{
		let cur_item = arrayfire::row(arr,i);
		let s0 = vec_to_str(&cur_item);
		wtr0.push(s0);
	}
	

	let mut file0 = File::create(filename).unwrap();
	writeln!(file0, "{}", wtr0.join("\n"));
}






pub fn write_vec_cpu_to_csv(
	filename: &str,
	invec: &Vec<f32>
	)
{

	let mut wtr0 = vec_cpu_to_str(invec);

	//wtr0 = wtr0.replace(",", "\n");
	
	let mut file0 = File::create(filename).unwrap();
	writeln!(file0, "{}", wtr0);
}










pub fn hash_batch_to_files(
	filename: &str,
	hash_map: &nohash_hasher::IntMap<u64, Vec<f32> >,
){


	for batch_idx in 0..(hash_map.len() as u64)
	{
		let cur_batch = hash_map[&batch_idx].clone();

		let tmpfilename = format!("{}_batch_idx_{}.yhat",filename,batch_idx);

		write_vec_cpu_to_csv(
			&tmpfilename,
			&cur_batch
		);
	}

}











pub fn save_network(
		filename: &str,
		netdata: &network_metadata_type,
		WValues: &arrayfire::Array<f32>,
		WRowIdxCSR: &arrayfire::Array<i32>,
		WColIdx: &arrayfire::Array<i32>,
		H: &arrayfire::Array<f32>,
		A: &arrayfire::Array<f32>,
		B: &arrayfire::Array<f32>,
		C: &arrayfire::Array<f32>,
		D: &arrayfire::Array<f32>,
		E: &arrayfire::Array<f32>,
		glia_pos: &arrayfire::Array<f32>,
		neuron_pos: &arrayfire::Array<f32>,
		neuron_idx: &arrayfire::Array<i32>
	)
{


	let neuron_size: u64 = netdata.neuron_size.clone();
	let input_size: u64 = netdata.input_size.clone();
	let output_size: u64 = netdata.output_size.clone();
	let proc_num: u64 = netdata.proc_num.clone();
	let active_size: u64 = netdata.active_size.clone();
	let space_dims: u64 = netdata.space_dims.clone();
	let step_num: u64 = netdata.step_num.clone();
	let batch_size: u64 = netdata.batch_size.clone();


	let del_unused_neuron: bool = netdata.del_unused_neuron.clone();


	let time_step: f32 = netdata.time_step.clone();
	let nratio: f32 = netdata.nratio.clone();
	let neuron_std: f32 = netdata.neuron_std.clone();
	let sphere_rad: f32 = netdata.sphere_rad.clone();
	let neuron_rad: f32 = netdata.neuron_rad.clone();
	let con_rad: f32 = netdata.con_rad.clone();
	let init_prob: f32 = netdata.init_prob.clone();
	let add_neuron_rate: f32 = netdata.add_neuron_rate.clone();
	let del_neuron_rate: f32 = netdata.del_neuron_rate.clone();
	let center_const: f32 = netdata.center_const.clone();
	let spring_const: f32 = netdata.spring_const.clone();
	let repel_const: f32 = netdata.repel_const.clone();




	let mut wtr0: Vec<String> = Vec::new();

	let s0 = vec_to_str(WValues);
	wtr0.push(s0);

	let s0 = vec_to_str_i32(WRowIdxCSR);
	wtr0.push(s0);

	let s0 = vec_to_str_i32(WColIdx);
	wtr0.push(s0);






	let s0 = vec_to_str(H);
	wtr0.push(s0);

	let s0 = vec_to_str(A);
	wtr0.push(s0);

	let s0 = vec_to_str(B);
	wtr0.push(s0);

	let s0 = vec_to_str(C);
	wtr0.push(s0);

	let s0 = vec_to_str(D);
	wtr0.push(s0);

	let s0 = vec_to_str(E);
	wtr0.push(s0);

	let s0 = vec_to_str(glia_pos);
	wtr0.push(s0);

	let s0 = vec_to_str(neuron_pos);
	wtr0.push(s0);

	let s0 = vec_to_str_i32(neuron_idx);
	wtr0.push(s0);



	let netarr0: Vec<u64> = vec![
		neuron_size,
		input_size,
		output_size,
		proc_num,
		active_size,
		space_dims,
		step_num,
		batch_size,
		del_unused_neuron as u64
		];

	let s0 = vec_cpu_to_str_u64(&netarr0);
	wtr0.push(s0);


	let netarr1: Vec<f32> = vec![
		time_step,
		nratio,
		neuron_std,
		sphere_rad,
		neuron_rad,
		con_rad,
		init_prob,
		add_neuron_rate,
		del_neuron_rate,
		center_const,
		spring_const,
		repel_const
		];

	let s0 = vec_cpu_to_str(&netarr1);
	wtr0.push(s0);

	let mut file0 = File::create(filename).unwrap();
	writeln!(file0, "{}", wtr0.join("\n"));


}













pub fn save_network2(
	filename: &str,
	neural_network: &neural_network_type
)
{
	let WValuesdims0 =  neural_network.WColIdx.dims()[0];

	let network_paramsdims0 =  neural_network.network_params.dims()[0];

	let Hdims0 =  (network_paramsdims0 -  WValuesdims0)/6; 



	let Wstart = 0;
	let Wend = (WValuesdims0  as i64) - 1;

	let Hstart = Wend + 1; 
	let Hend = Hstart + (Hdims0 as i64) - 1;

	let Astart = Hend + 1; 
	let Aend = Astart + (Hdims0 as i64) - 1;

	let Bstart = Aend + 1; 
	let Bend = Bstart + (Hdims0 as i64) - 1;

	let Cstart = Bend + 1; 
	let Cend = Cstart + (Hdims0 as i64) - 1;

	let Dstart = Cend + 1; 
	let Dend = Dstart + (Hdims0 as i64) - 1;

	let Estart = Dend + 1; 
	let Eend = Estart + (Hdims0 as i64) - 1;


	let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
	let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
	let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
	let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
	let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
	let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
	let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];



    let WValues = arrayfire::index(&neural_network.network_params, &Wseqs);
    let H = arrayfire::index(&neural_network.network_params, &Hseqs);
    let A = arrayfire::index(&neural_network.network_params, &Aseqs);
    let B = arrayfire::index(&neural_network.network_params, &Bseqs);
    let C = arrayfire::index(&neural_network.network_params, &Cseqs);
    let D = arrayfire::index(&neural_network.network_params, &Dseqs);
    let E = arrayfire::index(&neural_network.network_params, &Eseqs);



	save_network(
		filename,
		&neural_network.netdata,
		&WValues,
		&neural_network.WRowIdxCSR,
		&neural_network.WColIdx,
		&H,
		&A,
		&B,
		&C,
		&D,
		&E,
		&neural_network.glia_pos,
		&neural_network.neuron_pos,
		&neural_network.neuron_idx
	);



}












pub fn load_network(
		filename: &str,
		netdata: &mut network_metadata_type,
		WValues: &mut arrayfire::Array<f32>,
		WRowIdxCSR: &mut arrayfire::Array<i32>,
		WColIdx: &mut arrayfire::Array<i32>,
		H: &mut arrayfire::Array<f32>,
		A: &mut arrayfire::Array<f32>,
		B: &mut arrayfire::Array<f32>,
		C: &mut arrayfire::Array<f32>,
		D: &mut arrayfire::Array<f32>,
		E: &mut arrayfire::Array<f32>,
		glia_pos: &mut arrayfire::Array<f32>,
		neuron_pos: &mut arrayfire::Array<f32>,
		neuron_idx: &mut arrayfire::Array<i32>,

	)
{
	let mut netarr0: Vec<u64> = Vec::new();
	let mut netarr1: Vec<f32> = Vec::new();



	let contents = fs::read_to_string(filename).expect("error");
	let mut lines = contents.split("\n");
	let mut i = 0;
	for line in lines {
		match i {
			0 => *WValues = str_to_vec(
					&line
				),
			1 => *WRowIdxCSR = str_to_vec_i32(
					&line
				),
			2 => *WColIdx = str_to_vec_i32(
					&line
				),
			3 => *H = str_to_vec(
					&line
				),
			4 => *A = str_to_vec(
					&line
				),
			5 => *B = str_to_vec(
				&line
			),
			6 => *C = str_to_vec(
				&line
			),

			7 => *D = str_to_vec(
				&line
			),

			8 => *E = str_to_vec(
				&line
			),

			9 => *glia_pos = str_to_vec(
				&line
			),


			10 => *neuron_pos = str_to_vec(
				&line
			),

			11 => *neuron_idx = str_to_vec_i32(
				&line
			),

			12 => netarr0 = str_to_vec_cpu_u64(
				&line
			),

			13 => netarr1 = str_to_vec_cpu(
				&line
			),

			_ => println!("error"),
		}

		i = i + 1;
	}





	let neuron_size: u64 = netarr0[0];
	let input_size: u64 = netarr0[1];
	let output_size: u64 = netarr0[2];
	let proc_num: u64 = netarr0[3];
	let active_size: u64 = netarr0[4];
	let space_dims: u64 = netarr0[5];
	let step_num: u64 = netarr0[6];
	let batch_size: u64 = netarr0[7];


	let del_unused_neuron: bool = (netarr0[8] == 1);


	let time_step: f32 = netarr1[0];
	let nratio: f32 = netarr1[1];
	let neuron_std: f32 = netarr1[2];
	let sphere_rad: f32 = netarr1[3];
	let neuron_rad: f32 = netarr1[4];
	let con_rad: f32 = netarr1[5];
	let init_prob: f32 = netarr1[6];
	let add_neuron_rate: f32 = netarr1[7];
	let del_neuron_rate: f32 = netarr1[8];
	let center_const: f32 = netarr1[9];
	let spring_const: f32 = netarr1[10];
	let repel_const: f32 = netarr1[11];






	let newnetdata = network_metadata_type {
		neuron_size: neuron_size,
	    input_size: input_size,
		output_size: output_size,
		proc_num: proc_num,
		active_size: active_size,
		space_dims: space_dims,
		step_num: step_num,
		batch_size: batch_size,
		del_unused_neuron: del_unused_neuron,

		time_step: time_step,
		nratio: nratio,
		neuron_std: neuron_std,
		sphere_rad: sphere_rad,
		neuron_rad: neuron_rad,
		con_rad: con_rad,
		init_prob: init_prob,
		add_neuron_rate: add_neuron_rate,
		del_neuron_rate: del_neuron_rate,
		center_const: center_const,
		spring_const: spring_const,
		repel_const: repel_const
	};

	*netdata =  newnetdata;



	let g_dims = arrayfire::Dim4::new(&[(glia_pos.dims()[0]/space_dims) as u64 ,space_dims,1,1]);
	*glia_pos = arrayfire::moddims(glia_pos, g_dims);

	let n_dims = arrayfire::Dim4::new(&[(neuron_pos.dims()[0]/space_dims) as u64 ,space_dims,1,1]);
	*neuron_pos = arrayfire::moddims(neuron_pos, n_dims);


}





pub fn load_network2(
	filename: &str
)  ->  neural_network_type
{

	let temp_dims = arrayfire::Dim4::new(&[4,1,1,1]);

	let mut glia_pos = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut neuron_pos = arrayfire::constant::<f32>(0.0,temp_dims);



	
	let mut H = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut A = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut B = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut C = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut D = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut E = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut neuron_idx = arrayfire::constant::<i32>(0,temp_dims);




	let mut WValues = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut WRowIdxCSR = arrayfire::constant::<i32>(0,temp_dims);
	let mut WColIdx = arrayfire::constant::<i32>(0,temp_dims);


	let mut netdata = create_nullnetdata();
		let v_dims = arrayfire::Dim4::new(&[ netdata.neuron_size, netdata.batch_size,1,1,]);
	let mut Vmem   = arrayfire::constant::<f32>(0.0, v_dims);  // membrane potentials
	let mut Spikes = arrayfire::constant::<u32>(0   , v_dims); // spike flags

	load_network(
		filename,
		&mut netdata,
		&mut WValues,
		&mut WRowIdxCSR,
		&mut WColIdx,
		&mut H,
		&mut A,
		&mut B,
		&mut C,
		&mut D,
		&mut E,
		&mut glia_pos,
		&mut neuron_pos,
		
		&mut neuron_idx,

	);



	let total_param_size = WValues.dims()[0]  +  H.dims()[0]  +  A.dims()[0]    +  B.dims()[0]    +  C.dims()[0]    +  D.dims()[0]   +  E.dims()[0]  ;
	let network_params_dims = arrayfire::Dim4::new(&[total_param_size,1,1,1]);



	let Wstart = 0;
	let Wend = (WValues.dims()[0]  as i64) - 1;

	let Hstart = Wend + 1; 
	let Hend = Hstart + (H.dims()[0] as i64) - 1;

	let Astart = Hend + 1; 
	let Aend = Astart + (A.dims()[0] as i64) - 1;

	let Bstart = Aend + 1; 
	let Bend = Bstart + (B.dims()[0] as i64) - 1;

	let Cstart = Bend + 1; 
	let Cend = Cstart + (C.dims()[0] as i64) - 1;

	let Dstart = Cend + 1; 
	let Dend = Dstart + (D.dims()[0] as i64) - 1;

	let Estart = Dend + 1; 
	let Eend = Estart + (E.dims()[0] as i64) - 1;


	let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
	let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
	let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
	let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
	let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
	let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
	let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];



	let mut network_params = arrayfire::constant::<f32>(0.0,network_params_dims);
	arrayfire::assign_seq(&mut network_params, &Wseqs, &WValues);
	arrayfire::assign_seq(&mut network_params, &Hseqs, &H);
	arrayfire::assign_seq(&mut network_params, &Aseqs, &A);
	arrayfire::assign_seq(&mut network_params, &Bseqs, &B);
	arrayfire::assign_seq(&mut network_params, &Cseqs, &C);
	arrayfire::assign_seq(&mut network_params, &Dseqs, &D);	
	arrayfire::assign_seq(&mut network_params, &Eseqs, &E);	



	let mut  neural_network:   neural_network_type = neural_network_type {
		netdata: netdata,
		WRowIdxCSR: WRowIdxCSR,
		WColIdx: WColIdx,
		network_params: network_params,
		glia_pos: glia_pos,
		neuron_pos: neuron_pos,
		neuron_idx: neuron_idx,
		Vmem: Vmem,
		Spikes: Spikes
	};

	neural_network
}









pub fn load_network_structure(
	filename: &str,

	glia_pos: &mut arrayfire::Array<f32>,
	neuron_pos: &mut arrayfire::Array<f32>,
	neuron_idx: &mut arrayfire::Array<i32>
)
{
		
	let mut netarr0: Vec<u64> = Vec::new();
	let mut netarr1: Vec<f32> = Vec::new();



	let contents = fs::read_to_string(filename).expect("error");
	let mut lines = contents.split("\n");
	let mut i = 0;
	for line in lines {
		match i {
			0 => println!("error"),
			1 => println!("error"),
			2 => println!("error"),
			3 => println!("error"),
			4 => println!("error"),
			5 => println!("error"),
			6 => println!("error"),
			7 => println!("error"),
			8 => println!("error"),

			9 => *glia_pos = str_to_vec(
				&line
			),


			10 => *neuron_pos = str_to_vec(
				&line
			),

			11 => *neuron_idx = str_to_vec_i32(
				&line
			),

			12 => netarr0 = str_to_vec_cpu_u64(
				&line
			),

			13 => netarr1 = str_to_vec_cpu(
				&line
			),

			_ => println!("error"),
		}

		i = i + 1;
	}





	let neuron_size: u64 = netarr0[0];
	let input_size: u64 = netarr0[1];
	let output_size: u64 = netarr0[2];
	let proc_num: u64 = netarr0[3];
	let active_size: u64 = netarr0[4];
	let space_dims: u64 = netarr0[5];
	let step_num: u64 = netarr0[6];
	let batch_size: u64 = netarr0[7];


	let del_unused_neuron: bool = (netarr0[8] == 1);


	let time_step: f32 = netarr1[0];
	let nratio: f32 = netarr1[1];
	let neuron_std: f32 = netarr1[2];
	let sphere_rad: f32 = netarr1[3];
	let neuron_rad: f32 = netarr1[4];
	let con_rad: f32 = netarr1[5];
	let init_prob: f32 = netarr1[6];
	let add_neuron_rate: f32 = netarr1[7];
	let del_neuron_rate: f32 = netarr1[8];
	let center_const: f32 = netarr1[9];
	let spring_const: f32 = netarr1[10];
	let repel_const: f32 = netarr1[11];






	let newnetdata = network_metadata_type {
		neuron_size: neuron_size,
		input_size: input_size,
		output_size: output_size,
		proc_num: proc_num,
		active_size: active_size,
		space_dims: space_dims,
		step_num: step_num,
		batch_size: batch_size,
		del_unused_neuron: del_unused_neuron,

		time_step: time_step,
		nratio: nratio,
		neuron_std: neuron_std,
		sphere_rad: sphere_rad,
		neuron_rad: neuron_rad,
		con_rad: con_rad,
		init_prob: init_prob,
		add_neuron_rate: add_neuron_rate,
		del_neuron_rate: del_neuron_rate,
		center_const: center_const,
		spring_const: spring_const,
		repel_const: repel_const
	};

	//*netdata =  newnetdata;



	let g_dims = arrayfire::Dim4::new(&[(glia_pos.dims()[0]/space_dims) as u64 ,space_dims,1,1]);
	*glia_pos = arrayfire::moddims(glia_pos, g_dims);

	let n_dims = arrayfire::Dim4::new(&[(neuron_pos.dims()[0]/space_dims) as u64 ,space_dims,1,1]);
	*neuron_pos = arrayfire::moddims(neuron_pos, n_dims);


}


/// Normalize a flat slice (row-major: rows × features) **per feature** in-place.
/// - `feature_cnt`: number of features (columns) per row.
///
/// Panics if `flat.len() % feature_cnt != 0`.
pub fn normalize_minmax_per_feature_f32(flat: &mut [f32], feature_cnt: usize) {
    assert_eq!(
        flat.len() % feature_cnt,
        0,
        "flat.len() ({}) is not a multiple of feature_cnt ({})",
        flat.len(),
        feature_cnt
    );

    // 1) Compute per-feature min/max
    let mut mins = vec![f32::INFINITY; feature_cnt];
    let mut maxs = vec![f32::NEG_INFINITY; feature_cnt];

    for row in flat.chunks_exact(feature_cnt) {
        for (c, &v) in row.iter().enumerate() {
            if v < mins[c] {
                mins[c] = v;
            }
            if v > maxs[c] {
                maxs[c] = v;
            }
        }
    }

    // 2) Scale each element in-place: (x - min) / (max - min), or 0.0 if constant column
    for row in flat.chunks_exact_mut(feature_cnt) {
        for (c, x) in row.iter_mut().enumerate() {
            let range = maxs[c] - mins[c];
            if range.abs() < std::f32::EPSILON {
                *x = 0.0;
            } else {
                *x = (*x - mins[c]) / range;
            }
        }
    }
}


pub fn normalized_file_to_hash_cpu_manual(
    feature_count: usize,
    filename: &str,
    sample_size: u64,
    batch_size: u64,
) -> nohash_hasher::IntMap<u64, Vec<f32>> {
    // 1) Read entire file to a single string
    let mut flat = file_to_vec_cpu(filename);

    // 3) Ensure total length is a multiple of (sample_size × batch_size)
    let block_size = (sample_size * batch_size) as usize;
	// if flat.len() % block_size != 0 {
    //     panic!(
    //         "Flat.len() = {} is not divisible by sample_size×batch_size = {}",
    //         flat.len(),
    //         block_size
    //     );
    // }
    // 4) Apply per-feature normalization (in-place)
    normalize_minmax_per_feature_f32(&mut flat, feature_count);

    let total_len = flat.len() as u64;
    let item_num = total_len / (sample_size * batch_size);
    let mut lookup: IntMap<u64, Vec<f32>> = IntMap::default();

    for i in 0..item_num {
        let start = (i * sample_size * batch_size) as usize;
        let end = ((i + 1) * sample_size * batch_size) as usize;
        lookup.insert(i, flat[start..end].to_vec());
    }

    lookup
}

pub fn file_to_hash_cpu_timeseries_lines(
    filename: &str,
    sample_size: usize, // T = 150
    batch_size: usize,
) -> nohash_hasher::IntMap<u64, Vec<f32>> {
    let contents = std::fs::read_to_string(filename).expect("read file failed");

    // Parse each non-empty line into Vec<f32> of length sample_size
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for (ln, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() { continue; }

        let mut v: Vec<f32> = Vec::with_capacity(sample_size);
        for s in line.split(',') {
            let s = s.trim();
            if s.is_empty() { continue; }
            let x: f32 = s.parse().unwrap_or_else(|_| panic!("parse error line {}", ln + 1));
            if !x.is_finite() { panic!("non-finite value line {}", ln + 1); }
            v.push(x);
        }
        if v.len() != sample_size {
            panic!("line {} has {} values, expected {}", ln + 1, v.len(), sample_size);
        }
        rows.push(v);
    }

    if rows.len() % batch_size != 0 {
        panic!("num lines {} not divisible by batch_size {}", rows.len(), batch_size);
    }

    let item_num = rows.len() / batch_size;
    let mut lookup: nohash_hasher::IntMap<u64, Vec<f32>> = nohash_hasher::IntMap::default();

    for item in 0..item_num {
        let start = item * batch_size;
        let chunk = &rows[start..start + batch_size]; // chunk[b][t]

        // Flatten in ArrayFire order for dims [1, batch, T, 1]:
        // t outer, b inner
        let mut flat: Vec<f32> = Vec::with_capacity(batch_size * sample_size);
        for t in 0..sample_size {
            for b in 0..batch_size {
                flat.push(chunk[b][t]);
            }
        }

        lookup.insert(item as u64, flat);
    }

    lookup
}

pub fn file_to_hash_cpu_batch_blocks(
    filename: &str,
    time_len: usize,     // T
    batch_size: usize,   // B lines per block
) -> IntMap<u64, Vec<f32>> {
    let s = std::fs::read_to_string(filename).expect("read failed");

    // read all rows first
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for (ln, line) in s.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let mut v: Vec<f32> = Vec::with_capacity(time_len);
        for tok in line.split(',') {
            let tok = tok.trim();
            if tok.is_empty() { continue; }
            let x: f32 = tok.parse().unwrap_or_else(|_| panic!("parse error line {}", ln + 1));
            if !x.is_finite() { panic!("non-finite line {}", ln + 1); }
            v.push(x);
        }
        if v.len() != time_len {
            panic!("line {} has {} values, expected {}", ln + 1, v.len(), time_len);
        }
        rows.push(v);
    }

    if rows.len() % batch_size != 0 {
        panic!("rows {} not divisible by batch_size {}", rows.len(), batch_size);
    }

    let mut map: IntMap<u64, Vec<f32>> = IntMap::default();
    let blocks = rows.len() / batch_size;

    for bi in 0..blocks {
        let chunk = &rows[bi * batch_size..(bi + 1) * batch_size]; // chunk[b][t]

        // flatten for dims [1,B,T,1]: t outer, b inner
        let mut flat: Vec<f32> = Vec::with_capacity(batch_size * time_len);
        for t in 0..time_len {
            for b in 0..batch_size {
                flat.push(chunk[b][t]);
            }
        }

        map.insert(bi as u64, flat);
    }

    map
}

use std::io::{Read, BufRead, Seek, SeekFrom};

#[derive(Clone, Copy, Debug)]
pub struct ShdMeta { pub n: usize, pub c: usize, pub t: usize }

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