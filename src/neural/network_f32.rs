extern crate arrayfire;
use arrayfire::{self as af, af_print};

use std::collections::HashMap;
use nohash_hasher;

use crate::neural::activation_f32::UAF;
use crate::neural::activation_f32::deriUAF;


use crate::graph::large_sparse_i32::COO_batch_find;

use serde::{Serialize, Deserialize};
use crate::graph::tree_i32::find_unique;


const COO_find_limit: u64 = 1500000000;


const one: f32 = 1.0;

#[derive(Debug, Default)]
pub struct ProbeTraceAnn {
    pub neuron: i32,
    pub batch:  i32,
    pub t:      Vec<i32>,
    pub wx:     Vec<f32>,
    pub wq:     Vec<f32>,
    pub bias:   Vec<f32>,
    pub z:      Vec<f32>, // pre-activation
    pub q:      Vec<f32>, // post-activation (after UAF)
}
fn host_scalar(a: &af::Array<f32>) -> f32 {
    let mut h = [0.0f32; 1];
    a.host(&mut h);
    h[0]
}

fn csr_row_span(rowptr: &af::Array<i32>, row: i32) -> (i32, i32) {
    let r0 = { let r = af::row(rowptr, row as i64);   let mut h=[0i32;1]; r.host(&mut h); h[0] };
    let r1 = { let r = af::row(rowptr, (row+1) as i64); let mut h=[0i32;1]; r.host(&mut h); h[0] };
    (r0, r1)
}
fn wx_wq_row_scalar_ann(
    n: i32, b: i32, t: i32,
    input_size: i32, neuron_size: i32,
    WRowIdxCSR: &af::Array<i32>,
    WColIdx:    &af::Array<i32>,
    Wvals:      &af::Array<f32>,       // in CSR order
    X_t:        &af::Array<f32>,       // [in, B, 1, 1]
    S:          &af::Array<f32>,       // [neuron, B, 1, 1], drive before matmul
) -> (f32, f32)
{
    let (start, end) = csr_row_span(WRowIdxCSR, n);
    if end <= start { return (0.0, 0.0); }
    let nnz = (end - start) as i64;

    // row n: columns & values
    let cols_slice = af::rows(WColIdx, start as i64, end as i64 - 1);
    let vals_slice = af::rows(Wvals,   start as i64, end as i64 - 1);

    let mut cols_h = vec![0i32; nnz as usize];
    let mut vals_h = vec![0f32; nnz as usize];
    cols_slice.host(&mut cols_h);
    vals_slice.host(&mut vals_h);

    // X_t[:, b]
    let r_in = af::Seq::new(0.0, (input_size - 1) as f32, 1.0);
    let cb   = af::Seq::new(b as f32, b as f32, 1.0);
    let mut ix = af::Indexer::default();
    ix.set_index(&r_in, 0, None);
    ix.set_index(&cb,   1, None);
    let x_b = af::index_gen(X_t, ix);
    let mut x_h = vec![0f32; x_b.elements()];
    x_b.host(&mut x_h);

    // S[:, b]
    let r_all = af::Seq::new(0.0, (neuron_size - 1) as f32, 1.0);
    let mut is = af::Indexer::default();
    is.set_index(&r_all, 0, None);
    is.set_index(&cb,    1, None);
    let s_b = af::index_gen(S, is);
    let mut s_h = vec![0f32; s_b.elements()];
    s_b.host(&mut s_h);

    // accumulate WX (input cols) and WQ (non-input cols)
    let mut wx = 0.0f32;
    let mut wq = 0.0f32;
    for k in 0..(nnz as usize) {
        let j = cols_h[k] as i32;
        let w = vals_h[k];
        if j < input_size {
            wx += w * x_h[j as usize];
        } else {
            wq += w * s_h[j as usize];
        }
    }
    (wx, wq)
}



// ------------------------------------------------------------
// (1) Print contributions into ANN row n at time t, batch b
//     Splits WX (from inputs) and WQ (from previous post-acts).
// ------------------------------------------------------------
pub fn ann_print_row_contrib(
    n: i32, t: i32, b: i32,
    input_size: i32, neuron_size: i32,
    WRowIdxCSR: &af::Array<i32>,
    WColIdx:    &af::Array<i32>,
    Wvals:      &af::Array<f32>,   // CSR-ordered values
    X_t:        &af::Array<f32>,   // [in, B, 1, 1] at time t
    S_drive:    &af::Array<f32>,   // [neuron, B, 1, 1] drive used this step
) {
    // get row span (convert to i64 for ArrayFire rows())
    let (start32, end32) = csr_row_span(WRowIdxCSR, n);
    let (start, end) = (start32 as i64, end32 as i64);
    let nnz = (end - start).max(0) as usize;

    // host buffers
    let mut cols_h: Vec<i32> = Vec::new();
    let mut vals_h: Vec<f32> = Vec::new();

    if end > start {
        let cols = af::rows(WColIdx, start, end - 1);
        let vals = af::rows(Wvals,   start, end - 1);
        cols_h.resize(cols.elements(), 0);
        vals_h.resize(vals.elements(), 0.0);
        cols.host(&mut cols_h);
        vals.host(&mut vals_h);
    }

    // scalar at (row=j, batch=b)
    let at_nb = |A: &af::Array<f32>, j: i32| -> f32 {
        let rj = af::Seq::new(j as f32, j as f32, 1.0);
        let cb = af::Seq::new(b as f32, b as f32, 1.0);
        let mut ix = af::Indexer::default();
        ix.set_index(&rj, 0, None);
        ix.set_index(&cb, 1, None);
        host_scalar(&af::index_gen(A, ix))
    };

    println!("row {n} has {} incoming edges", nnz);
    println!("t={t:>2}  {:<4} {:<6} {:>14} {:>14} {:>14}",
             "j", "kind", "W[n,j]", "src", "contrib");

    let mut sum_wx = 0.0f32;
    let mut sum_wq = 0.0f32;

    for k in 0..nnz {
        let j = cols_h[k];
        let w = vals_h[k];
        if j < input_size {
            let x = at_nb(X_t, j);
            let c = w * x;
            sum_wx += c;
            println!("{:>4} {:<6} {:>+14.6} {:>+14.6} {:>+14.6}",
                     j, "IN", w, x, c);
        } else {
            let qprev = at_nb(S_drive, j); // previous post-activation for non-inputs
            let c = w * qprev;
            sum_wq += c;
            println!("{:>4} {:<6} {:>+14.6} {:>+14.6} {:>+14.6}",
                     j, "REC", w, qprev, c);
        }
    }

    println!("      -- totals: WX={:+.6}, WQ={:+.6}\n", sum_wx, sum_wq);
}

// ------------------------------------------------------------------
// (2) Print consumers of Q[n,t] (i.e., rows i where W[i,n] != 0).
//     Scans each CSR row and looks for column == n.
// ------------------------------------------------------------------
pub fn ann_print_consumers_of(
    n: i32,
    neuron_size: i32,
    WRowIdxCSR: &af::Array<i32>,
    WColIdx:    &af::Array<i32>,
    Wvals:      &af::Array<f32>,
) {
    let mut consumers: Vec<(i32, f32)> = Vec::new();

    for i in 0..neuron_size {
        let (s32, e32) = csr_row_span(WRowIdxCSR, i);
        let (s, e) = (s32 as i64, e32 as i64);
        if e <= s { continue; }

        let cols = af::rows(WColIdx, s, e - 1);
        let vals = af::rows(Wvals,   s, e - 1);

        let mut ch = vec![0i32; cols.elements()];
        let mut vh = vec![0f32; vals.elements()];
        cols.host(&mut ch);
        vals.host(&mut vh);

        for (cj, w) in ch.into_iter().zip(vh.into_iter()) {
            if cj == n {
                consumers.push((i, w));
            }
        }
    }

    println!("consumers of Q[n={n}] (targets i with W[i,{n}]≠0):");
    for (i, w) in consumers {
        println!("  target row i={:<3} with W[i,{n}]={:+.6}", i, w);
    }
}


#[derive(Serialize, Deserialize)]
pub struct network_metadata_type {
    pub neuron_size: u64,
    pub input_size: u64,
	pub output_size: u64,
	pub proc_num: u64,
	pub active_size: u64,
	pub space_dims: u64,
	pub step_num: u64,
    pub batch_size: u64,
	pub del_unused_neuron: bool,
	pub time_step: f32,
	pub nratio: f32,
	pub neuron_std: f32,
	pub sphere_rad: f32,
	pub neuron_rad: f32,
	pub con_rad: f32,
    pub init_prob: f32,
    pub add_neuron_rate: f32,
    pub del_neuron_rate: f32,
	pub center_const: f32,
	pub spring_const: f32,
	pub repel_const: f32
}





#[derive(Serialize, Deserialize)]
pub struct neural_network_type {
    pub netdata: network_metadata_type,
    pub WRowIdxCSR: arrayfire::Array<i32>,
    pub WColIdx: arrayfire::Array<i32>,
    pub network_params: arrayfire::Array<f32>,
    pub glia_pos: arrayfire::Array<f32>,
    pub neuron_pos: arrayfire::Array<f32>,
    pub neuron_idx: arrayfire::Array<i32>,

     /*** NEW state for SNN ***/
     pub Vmem  : arrayfire::Array<f32>,   // membrane voltage per neuron
     pub Spikes: arrayfire::Array<u32>,   // last-step spike (0/1) per neuron
}








pub fn print_netdata(
    netdata: &network_metadata_type
)
{
    println!("\n\n******Network Information******");
    println!("neuron_size: {}",netdata.neuron_size);
    println!("input_size: {}",netdata.input_size);
    println!("output_size: {}",netdata.output_size);
    println!("proc_num: {}",netdata.proc_num);
    println!("active_size: {}",netdata.active_size);
    println!("space_dims: {}",netdata.space_dims);
    println!("step_num: {}",netdata.step_num);
    println!("batch_size: {}",netdata.batch_size);
    println!("del_unused_neuron: {}",netdata.del_unused_neuron);



    println!("time_step: {}",netdata.time_step);
    println!("nratio: {}",netdata.nratio);
    println!("neuron_std: {}",netdata.neuron_std);
    println!("sphere_rad: {}",netdata.sphere_rad);
    println!("neuron_rad: {}",netdata.neuron_rad);
    println!("con_rad: {}",netdata.con_rad);
    println!("init_prob: {}",netdata.init_prob);
    println!("add_neuron_rate: {}",netdata.add_neuron_rate);
    println!("del_neuron_rate: {}",netdata.del_neuron_rate);
    println!("center_const: {}",netdata.center_const);
    println!("spring_const: {}",netdata.spring_const);
    println!("repel_const: {}",netdata.repel_const);
    println!("************\n\n");
}














pub fn clone_netdata(netdata: &network_metadata_type) -> network_metadata_type
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



    newnetdata
}





pub fn clone_neural_network(neural_network: &neural_network_type) -> neural_network_type
{


    let mut new_neural_network =    neural_network_type {
        netdata: clone_netdata(&neural_network.netdata),
        WRowIdxCSR: neural_network.WRowIdxCSR.clone(),
        WColIdx: neural_network.WColIdx.clone(),
        network_params: neural_network.network_params.clone(),
        glia_pos: neural_network.glia_pos.clone(),
        neuron_pos: neural_network.neuron_pos.clone(),
        neuron_idx: neural_network.neuron_idx.clone(),
        /* NEW */
        Vmem:          neural_network.Vmem.clone(),
        Spikes:        neural_network.Spikes.clone()

    };
    
    
    new_neural_network
}








pub fn create_nullnetdata() -> network_metadata_type
{

	let netdata: network_metadata_type = network_metadata_type {
		neuron_size: 0,
	    input_size: 0,
		output_size: 0,
		proc_num: 0,
		active_size: 0,
		space_dims: 0,
		step_num: 0,
		batch_size: 0,
		del_unused_neuron: false,

		time_step: 0.0,
		nratio: 0.0,
		neuron_std: 0.0,
		sphere_rad: 0.0,
		neuron_rad: 0.0,
		con_rad: 0.0,
		init_prob: 0.0,
		add_neuron_rate: 0.0,
		del_neuron_rate: 0.0,
		center_const: 0.0,
		spring_const: 0.0,
		repel_const: 0.0
	};



    netdata
}












pub fn UAF_initial_as_identity(
		netdata: &network_metadata_type,
		//H: &mut arrayfire::Array<f32>,
		A: &mut arrayfire::Array<f32>,
		B: &mut arrayfire::Array<f32>,
		C: &mut arrayfire::Array<f32>,
		D: &mut arrayfire::Array<f32>,
		E: &mut arrayfire::Array<f32>
	)
    {
        let neuron_size: u64 = netdata.neuron_size.clone();
		let input_size: u64 = netdata.input_size.clone();
		let output_size: u64 = netdata.output_size.clone();
		let proc_num: u64 = netdata.proc_num.clone();
		let active_size: u64 = netdata.active_size.clone();
		let space_dims: u64 = netdata.space_dims.clone();
		let step_num: u64 = netdata.step_num.clone();


		let del_unused_neuron: bool = netdata.del_unused_neuron.clone();


		let time_step: f32 = netdata.time_step.clone();
		let nratio: f32 = netdata.nratio.clone();
		let neuron_std: f32 = netdata.neuron_std.clone();
		let sphere_rad: f32 = netdata.sphere_rad.clone();
		let neuron_rad: f32 = netdata.neuron_rad.clone();
		let con_rad: f32 = netdata.con_rad.clone();
		let center_const: f32 = netdata.center_const.clone();
		let spring_const: f32 = netdata.spring_const.clone();
		let repel_const: f32 = netdata.repel_const.clone();





		let H_dims = arrayfire::Dim4::new(&[neuron_size,1,1,1]);
        //  *H = 0.001*neuron_std*arrayfire::randn::<f32>(H_dims);
        *A = one + 0.0001*neuron_std*arrayfire::randn::<f32>(H_dims);
		*B = 0.0001*neuron_std*arrayfire::randn::<f32>(H_dims);
		*C = 0.00001*neuron_std*arrayfire::randn::<f32>(H_dims);
		*D = -one + 0.00001*neuron_std*arrayfire::randn::<f32>(H_dims);
		*E = 0.00001*neuron_std*arrayfire::randn::<f32>(H_dims);


}








pub fn UAF_initial_as_tanh(
    netdata: &network_metadata_type,
    //H: &mut arrayfire::Array<f32>,
    A: &mut arrayfire::Array<f32>,
    B: &mut arrayfire::Array<f32>,
    C: &mut arrayfire::Array<f32>,
    D: &mut arrayfire::Array<f32>,
    E: &mut arrayfire::Array<f32>
)
{
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();


    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();


    let time_step: f32 = netdata.time_step.clone();
    let nratio: f32 = netdata.nratio.clone();
    let neuron_std: f32 = netdata.neuron_std.clone();
    let sphere_rad: f32 = netdata.sphere_rad.clone();
    let neuron_rad: f32 = netdata.neuron_rad.clone();
    let con_rad: f32 = netdata.con_rad.clone();
    let center_const: f32 = netdata.center_const.clone();
    let spring_const: f32 = netdata.spring_const.clone();
    let repel_const: f32 = netdata.repel_const.clone();





    let H_dims = arrayfire::Dim4::new(&[neuron_size,1,1,1]);
    //   *H = 0.0000001*neuron_std*arrayfire::randn::<f32>(H_dims);
    *A = 2.12616013f32 + 0.0001*neuron_std*arrayfire::randn::<f32>(H_dims);
    *B = (1.0f32/2.12616013f32) + 0.0001*neuron_std*arrayfire::randn::<f32>(H_dims);
    *C = 0.0000001*neuron_std*arrayfire::randn::<f32>(H_dims);
    *D = 2.12616013f32 + 0.0001*neuron_std*arrayfire::randn::<f32>(H_dims);
    *E = -1.0f32 +  0.0001*neuron_std*arrayfire::randn::<f32>(H_dims);


}








pub fn xavier_init(
    in_idx: &arrayfire::Array<i32>,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
    neuron_size: u64,
    depth: u64,

    WValues: &mut arrayfire::Array<f32>,
    H: &mut arrayfire::Array<f32>,
)
{
    *WValues = 0.000001f32*arrayfire::randn::<f32>(WValues.dims());


    let H_dims = arrayfire::Dim4::new(&[neuron_size,1,1,1]);
    *H = 0.000001f32*arrayfire::randn::<f32>(H_dims);



    let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);

    let mut out_idx = in_idx.clone();
    let mut valsel = arrayfire::constant::<i32>(0,temp_dims);

    let COO_batch_size = 1 + ((COO_find_limit/WColIdx.dims()[0]) as u64);


    let mut input_degree = 0;
    let mut output_degree = 0;

    for i in 0..depth
    {
        valsel = COO_batch_find(WColIdx, &out_idx, COO_batch_size);
        if valsel.dims()[0] == 0
        {
            break;
        }
        out_idx = arrayfire::lookup(WRowIdxCOO, &valsel, 0);

        out_idx = find_unique(&out_idx, neuron_size);


        output_degree = valsel.dims()[0];

        let mulitiplier = (6.0f32/((input_degree + output_degree) as f32) ).sqrt()*2.0f32;
        let mut newWValues = arrayfire::randu::<f32>(valsel.dims());
        newWValues = (newWValues - 0.5f32)*( mulitiplier );

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&valsel, 0, None);
        arrayfire::assign_gen(WValues, &idxrs, &newWValues);

        drop(newWValues);
        drop(idxrs);


        let mut newH = arrayfire::randu::<f32>(out_idx.dims());
        newH = (newH - 0.5f32)*( mulitiplier );

        let mut idxrs2 = arrayfire::Indexer::default();
        idxrs2.set_index(&out_idx, 0, None);
        arrayfire::assign_gen(H, &idxrs2, &newH);



        if out_idx.dims()[0] == 0
        {
            break;
        }

        input_degree = valsel.dims()[0];
    }


}











/*
Forward pass using CSR weighted adjacency sparse matrices and UAF. 
Generates all internal states and the neural network output


Inputs
netdata:             Neural Network Metadata
X:                   Input Matrix
WRowIdxCSR:          Row sparse matrix of the weighted adjacency matrix
WColIdx:             Column sparse matrix of the weighted adjacency matrix
Wseqs:               Indexes of the weight parameters
Hseqs:               Indexes of the bias parameters
Aseqs:               Indexes of the UAF A parameters
Bseqs:               Indexes of the UAF B parameters
Cseqs:               Indexes of the UAF C parameters
Dseqs:               Indexes of the UAF D parameters
Eseqs:               Indexes of the UAF E parameters
network_params:      All trainable parameters in the network


Outputs:
Z:                   Internal State Matrix Z   
Q:                   Internal State Matrix Q

*/

pub fn state_space_forward_batch(
    netdata: &network_metadata_type,
    X: &arrayfire::Array<f32>,
    
    WRowIdxCSR: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,


    Wseqs: &[arrayfire::Seq<i32>; 1],
    Hseqs: &[arrayfire::Seq<i32>; 1],
    Aseqs: &[arrayfire::Seq<i32>; 1],
    Bseqs: &[arrayfire::Seq<i32>; 1],
    Cseqs: &[arrayfire::Seq<i32>; 1],
    Dseqs: &[arrayfire::Seq<i32>; 1],
    Eseqs: &[arrayfire::Seq<i32>; 1],
    network_params: &arrayfire::Array<f32>,



    Z: &mut arrayfire::Array<f32>,
    Q: &mut arrayfire::Array<f32>
) {

    let neuron_size: u64 = netdata.neuron_size.clone();
    //let proc_num: u64 = netdata.proc_num.clone();
    let input_size: u64 = netdata.input_size.clone();
    let batch_size: u64 = netdata.batch_size.clone();

    let Zslices:i64 = Z.dims()[2] as i64;

    let X_slices:i64 = X.dims()[2] as i64;

    let S_dims = arrayfire::Dim4::new(&[neuron_size,batch_size,1,1]);
    let mut S = arrayfire::constant::<f32>(0.0, S_dims);


    let mut tempx =  arrayfire::slice(X, 0);
    let seqs = &[arrayfire::Seq::new(0.0f32, (input_size-1) as f32, 1.0f32),  arrayfire::Seq::default()];






    let H = arrayfire::index(&network_params, Hseqs);
    let A = arrayfire::index(&network_params, Aseqs);
    let B = arrayfire::index(&network_params, Bseqs);
    let C = arrayfire::index(&network_params, Cseqs);
    let D = arrayfire::index(&network_params, Dseqs);
    let E = arrayfire::index(&network_params, Eseqs);



    let WValues = arrayfire::index(network_params, Wseqs);


    let W = arrayfire::sparse::<f32>(
        neuron_size,
        neuron_size,
        &WValues,
        WRowIdxCSR,
        WColIdx,
        arrayfire::SparseFormat::CSR
    );
    

    for i in 0i64..Zslices
    {
        if X_slices > 1
        {
            tempx =  arrayfire::slice(X, i);
            arrayfire::assign_seq(&mut S, seqs, &tempx);
            drop(tempx);
        }
        else
        {
            arrayfire::assign_seq(&mut S, seqs, &X);
        }

        // println!("{:?}",i);

        //             af_print!( "S     S {:?}", S);
        
        S = arrayfire::matmul(&W, &S, arrayfire::MatProp::NONE, arrayfire::MatProp::NONE);

        S = arrayfire::add(&S, &H, true);
        arrayfire::set_slice(Z, &S, i);

        S = UAF(&S,&A,&B,&C,&D,&E);
        arrayfire::set_slice(Q, &  S , i);

        
    }


    
}



















/*
Backward pass using CSR weighted adjacency sparse matrices and UAF.
Generates the gradients of the sparse weighted adjacency matrix


Inputs
netdata:             Neural Network Metadata
X:                   Input Matrix
network_params:      All trainable parameters in the network
Z:                   Internal State matrix Z
Q:                   Internal State matrix Q
Y:                   Target Matrix to fit
loss_grad:           Loss function
neuron_idx:          Indexes of the neurons
idxsel_out:          Indexes of the UAF parameters
valsel_out:          Indexes of the UAF values
cvec_out:            Indexes of the UAF column sparse vector
dXsel_out:           Indexes of the dX values
nrows_out:           Number of rows in UAF
sparseval_out:       Indexes of the values in the sparse matrix
sparserow_out:       Indexes of the rows in the sparse matrix
sparsecol_out:       Indexes of the columns in the sparse matrix
Hidxsel_out:         Indexes of the bias vector H
Aidxsel_out:         Indexes of the UAF vector A
Bidxsel_out:         Indexes of the UAF vector B
Cidxsel_out:         Indexes of the UAF vector C
Didxsel_out:         Indexes of the UAF vector D
Eidxsel_out:         Indexes of the UAF vector E
combidxsel_out:      Indexes of the all vectors
dAseqs_out:          Indexes of the dA
dBseqs_out:          Indexes of the dB
dCseqs_out:          Indexes of the dC
dDseqs_out:          Indexes of the dD
dEseqs_out:          Indexes of the dE


Outputs:
grad:                   Gradient of all trainable parameters


*/

pub fn state_space_backward_group2(
    netdata: &network_metadata_type,
    X: &arrayfire::Array<f32>,



    network_params: &arrayfire::Array<f32>,




    Z: &arrayfire::Array<f32>,
    Q: &arrayfire::Array<f32>,
    Y: &arrayfire::Array<f32>,
    loss_grad: impl Fn(&arrayfire::Array<f32>, &arrayfire::Array<f32>) -> arrayfire::Array<f32>,
    neuron_idx: &arrayfire::Array<i32>,



    idxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    valsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    
    cvec_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    dXsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    
    nrows_out: &nohash_hasher::IntMap<i64, u64 >,
    sparseval_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    sparserow_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    sparsecol_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,



    Hidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Aidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Bidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Cidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Didxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Eidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    combidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,




    dAseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dBseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dCseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dDseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dEseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,




    grad: &mut arrayfire::Array<f32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();



    let batch_size: u64 = netdata.batch_size.clone();



    //Set output to zero
    *grad = arrayfire::constant::<f32>(0.0,network_params.dims());








    //Get current selection of neurons
    let active_size = neuron_idx.dims()[0];
    let idxsel = arrayfire::rows(neuron_idx, (active_size-output_size) as i64, (active_size-1)   as i64);


    let Qslices: u64 = Q.dims()[2];
    let Yslices: u64 = Y.dims()[2];


    //Get Yhat
    let mut idxrs = arrayfire::Indexer::default();
    let seq1 = arrayfire::Seq::new(0.0f32, (batch_size-1) as f32, 1.0);
    let seq2: arrayfire::Seq<f32> = arrayfire::Seq::new((proc_num-1) as f32, (Qslices-1) as f32, 1.0);
    idxrs.set_index(&idxsel, 0, None);
    idxrs.set_index(&seq1, 1, None);
    idxrs.set_index(&seq2, 2, None);
    let Yhat = arrayfire::index_gen(Q, idxrs);
    drop(idxsel);

    let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let S_dims = arrayfire::Dim4::new(&[neuron_size,batch_size,1,1]);
    

    //Calculate error
    let total_error = loss_grad(&Yhat,Y);
    let mut yslicidx: i64 = (Yslices-1) as i64;
    let mut error = arrayfire::slice(&total_error, yslicidx);


    let Zslices: i64 = Z.dims()[2] as i64;

    let X_slices: i64 = X.dims()[2] as i64;

    



    let mut inx = arrayfire::constant::<f32>(0.0,S_dims);

    let mut tempx =  arrayfire::slice(X, 0);


    let seqs = &[arrayfire::Seq::new(0.0f32, (input_size-1) as f32, 1.0f32),  arrayfire::Seq::default()];


    let mut Xtemp = arrayfire::slice(Z, 0);


    let mut sA = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut sB = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut sC = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut sD = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut sE = arrayfire::constant::<f32>(0.0,temp_dims);




    let mut dX = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut dA = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut dB = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut dC = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut dD = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut dE = arrayfire::constant::<f32>(0.0,temp_dims);




    let mut tempW = arrayfire::constant::<f32>(0.0,temp_dims);


    let mut gtemperr = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut tempinx = arrayfire::constant::<f32>(0.0,temp_dims);



    let mut tempdX = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut tempgW = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut tempgH = arrayfire::constant::<f32>(0.0,temp_dims);




    let mut derror = arrayfire::slice(&total_error,  0);






    let batchseq = arrayfire::Seq::new(0.0f32, (batch_size-1) as f32, 1.0);
    let mut sliceseq = arrayfire::Seq::new((proc_num-1) as f32, (Qslices-1) as f32, 1.0);




    let mut keys = arrayfire::constant::<i32>(0,temp_dims);
    let mut vals = arrayfire::constant::<f32>(0.0,temp_dims);




    let mut UAFgroup = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut tileerror = arrayfire::constant::<f32>(0.0,temp_dims);
    let tileerror_dims = arrayfire::Dim4::new(&[5,1,1,1]);



    //Main loop
    for i in (0i64..Zslices).rev() {
        // println!("[DEBUG] idxsel_out[{}] len = {}", i, idxsel_out[&i].elements());

       //Select X value
        let mut idxrs = arrayfire::Indexer::default();
        sliceseq = arrayfire::Seq::new(i as f32, i as f32, 1.0);
        idxrs.set_index(&idxsel_out[&i], 0, None);
        idxrs.set_index(&batchseq, 1, None);
        idxrs.set_index(&sliceseq, 2, None);
        Xtemp = arrayfire::index_gen(Z, idxrs);


        //Get current UAF parameters
        sA = arrayfire::lookup(&network_params, &Aidxsel_out[&i], 0);

        sB = arrayfire::lookup(&network_params, &Bidxsel_out[&i], 0);

        sC = arrayfire::lookup(&network_params, &Cidxsel_out[&i], 0);

        sD = arrayfire::lookup(&network_params, &Didxsel_out[&i], 0);

        sE = arrayfire::lookup(&network_params, &Eidxsel_out[&i], 0);


        //Compute derivative of UAF
        deriUAF(&Xtemp,
            &sA,
            &sB,
            &sC,
            &sD,
            &sE,
            &mut dX,
            &mut dA,
            &mut dB,
            &mut dC,
            &mut dD,
            &mut dE);
        drop(Xtemp);
        drop(sA);
        drop(sB);
        drop(sC);
        drop(sD);
        drop(sE);

        //Compute dX
                    //   println!("[DEBUG]   dx1.dims() = {:?}", dX.dims());
        // println!("dX dim{}, error dim{}",dX.dims(),error.dims());

        dX = arrayfire::mul(&dX, &error, false);
            //             println!("[DEBUG]   dX.dims()   = {:?}", dX.dims());
            //   println!("[DEBUG]   dx2.dims() = {:?}", dX.dims());

        //Update H
        tempgH = arrayfire::lookup(grad, &Hidxsel_out[&i], 0) + (arrayfire::sum(&dX, 1) );


        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&Hidxsel_out[&i], 0, None);
        arrayfire::assign_gen(grad, &idxrs, &tempgH);
        drop(tempgH);



        //Join all

        UAFgroup = arrayfire::constant::<f32>(0.0,arrayfire::Dim4::new(&[dA.dims()[0]*5 , dA.dims()[1],1,1]));
        arrayfire::assign_seq(&mut UAFgroup, &dAseqs_out[&i], &dA);
        arrayfire::assign_seq(&mut UAFgroup, &dBseqs_out[&i], &dB);
        arrayfire::assign_seq(&mut UAFgroup, &dCseqs_out[&i], &dC);
        arrayfire::assign_seq(&mut UAFgroup, &dDseqs_out[&i], &dD);
        arrayfire::assign_seq(&mut UAFgroup, &dEseqs_out[&i], &dE);



        tileerror =  arrayfire::tile(&error, tileerror_dims);

        UAFgroup = arrayfire::mul(&tileerror, &UAFgroup, false);
        drop(tileerror);

        UAFgroup = arrayfire::sum(&UAFgroup, 1) + arrayfire::lookup(grad, &combidxsel_out[&i],  0);

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&combidxsel_out[&i], 0, None);
        arrayfire::assign_gen(grad, &idxrs, &UAFgroup);
        drop(UAFgroup);










        //Get dX of each row
        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&dXsel_out[&i], 0, None);
        idxrs.set_index(&batchseq, 1, None);
        tempdX = arrayfire::index_gen(&dX, idxrs);
        












        //Get input values
        inx = arrayfire::constant::<f32>(0.0,S_dims);

        if (i > 0)
        {
            inx = arrayfire::slice(Q, (i-1) );
        }


        if X_slices > 1
        {
            tempx =  arrayfire::slice(X, i);
            arrayfire::assign_seq(&mut inx, seqs, &tempx);
            drop(tempx);
        }
        else
        {
            arrayfire::assign_seq(&mut inx, seqs, &X);
        }


        //Upadate gW
        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&cvec_out[&i], 0, None);
        idxrs.set_index(&batchseq, 1, None);
        tempinx = arrayfire::index_gen(&inx, idxrs);
        drop(inx);

        tempgW = arrayfire::mul(&tempdX, &tempinx, false);
        drop(tempinx);
        tempgW = (arrayfire::sum(&tempgW, 1) )+ arrayfire::lookup(grad, &valsel_out[&i], 0);



        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&valsel_out[&i], 0, None);
        arrayfire::assign_gen(grad, &idxrs, &tempgW);
        drop(tempgW);


        


        //Propagate Errors
        tempW = arrayfire::lookup(network_params, &sparseval_out[&i], 0);

        tempW = arrayfire::sparse::<f32>(
            nrows_out[&i],
            dX.dims()[0],
            &tempW,
            &sparserow_out[&i],
            &sparsecol_out[&i],
            arrayfire::SparseFormat::CSR
        );
        
        
        error = arrayfire::matmul(&tempW,
            &dX,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::NONE
        );
                // println!("tempW dim{}, dX dim{}",tempW.dims(),dX.dims());

        
        drop(tempW);
        



        //Add new Y error
        if (yslicidx > 0)
        {
            yslicidx = yslicidx - 1;
            derror = arrayfire::slice(&total_error,  yslicidx);

        // println!("error dim{}, derror dim{}",error.dims(),derror.dims());

            error = arrayfire::join(0, &error, &derror);
            
        }



}





}

pub fn state_space_forward_batch_with_probe(
    netdata: &network_metadata_type,
    X: &af::Array<f32>,

    WRowIdxCSR: &af::Array<i32>,
    WColIdx:    &af::Array<i32>,

    Wseqs: &[af::Seq<i32>; 1],
    Hseqs: &[af::Seq<i32>; 1],
    Aseqs: &[af::Seq<i32>; 1],
    Bseqs: &[af::Seq<i32>; 1],
    Cseqs: &[af::Seq<i32>; 1],
    Dseqs: &[af::Seq<i32>; 1],
    Eseqs: &[af::Seq<i32>; 1],

    network_params: &af::Array<f32>,

    Z: &mut af::Array<f32>,   // pre-activation per t  (== S right after W*S+H)
    Q: &mut af::Array<f32>,   // post-activation per t (== S after UAF)

    mut probe: Option<&mut ProbeTraceAnn>,
) {
    use arrayfire as af;

    let neuron_size = netdata.neuron_size as i64;
    let input_size  = netdata.input_size  as i64;
    let batch_size  = netdata.batch_size  as i64;

    let T   = Z.dims()[2] as i64;
    let Xin = X.dims()[2] as i64;

    // State / drive vector S (updated in-place across time)
    let s_dims = af::Dim4::new(&[neuron_size as u64, batch_size as u64, 1, 1]);
    let mut S  = af::constant::<f32>(0.0, s_dims);

    // bind Seqs (avoid E0716)
    let rows_in   = af::Seq::new(0.0, (input_size - 1) as f32, 1.0);
    let all_inp   = &[rows_in, af::Seq::default()];

    // parameters
    let H = af::index(network_params, Hseqs); // [neuron, 1, 1, 1]
    let A = af::index(network_params, Aseqs);
    let B = af::index(network_params, Bseqs);
    let C = af::index(network_params, Cseqs);
    let D = af::index(network_params, Dseqs);
    let E = af::index(network_params, Eseqs);

    let Wvals = af::index(network_params, Wseqs);
    let W = af::sparse::<f32>(
        neuron_size as u64, neuron_size as u64,
        &Wvals, WRowIdxCSR, WColIdx, af::SparseFormat::CSR
    );

    for t in 0..T {
        // 1) slice current inputs
        let X_t = if Xin > 1 { af::slice(X, t) } else { X.clone() };

        // 2) build drive *in-place*:
        //    - overwrite input rows with X_t
        //    - keep non-input rows as previous post-activation (recurrent state)
        af::assign_seq(&mut S, all_inp, &X_t);

        // For probe: capture the exact drive used this step
        let S_drive = S.clone();

        // 3) pre-activation: S = W*S + H
        S = af::matmul(&W, &S, af::MatProp::NONE, af::MatProp::NONE);
        S = af::add(&S, &H, true);

        // Store Z_t (== current S)
        af::set_slice(Z, &S, t);

        // 4) post-activation: S = UAF(S)
        S = UAF(&S, &A, &B, &C, &D, &E);

        // Store Q_t (== current S)
        af::set_slice(Q, &S, t);

        // 5) probe (WX/WQ split, bias, z, q for one neuron/batch)
        if let Some(tr) = &mut probe {
            let n = tr.neuron;
            let b = tr.batch;

            // bias[n]
            let bias_n = host_scalar(&af::row(&H, n as i64));

            // z[n,b] from Z_t (which we stored as S before activation):
            let z_nb = {
                let rn = af::Seq::new(n as f32, n as f32, 1.0);
                let cb = af::Seq::new(b as f32, b as f32, 1.0);
                let mut ix = af::Indexer::default();
                ix.set_index(&rn, 0, None);
                ix.set_index(&cb, 1, None);

                // read back the same pre-activation we just wrote into Z at time t
                let Z_t = af::slice(Z, t);
                 if t == T-1 && tr.neuron == 10 && tr.batch == 0 {
        let Wvals = af::index(network_params, Wseqs); // same as you build W with
        ann_print_row_contrib(
            tr.neuron as i32, t as i32, tr.batch as i32,
            input_size as i32, neuron_size as i32,
            WRowIdxCSR, WColIdx, &Wvals,
            &X_t, &S_drive
        );
        ann_print_consumers_of(
            tr.neuron as i32,
            neuron_size as i32,
            WRowIdxCSR, WColIdx, &Wvals
        );
    }
                host_scalar(&af::index_gen(&Z_t, ix))
            };

            // q[n,b] from current S (post-activation)
            let q_nb = {
                let rn = af::Seq::new(n as f32, n as f32, 1.0);
                let cb = af::Seq::new(b as f32, b as f32, 1.0);
                let mut ix = af::Indexer::default();
                ix.set_index(&rn, 0, None);
                ix.set_index(&cb, 1, None);
                host_scalar(&af::index_gen(&S, ix))
                
            };

            // WX/WQ computed against the drive vector:
            //  - inputs use X_t (current)
            //  - non-inputs use S_drive (previous step's post-activation)
            let (wx, wq) = wx_wq_row_scalar_ann(
                n, b, t as i32,
                input_size as i32, neuron_size as i32,
                WRowIdxCSR, WColIdx, &Wvals,
                &X_t, &S_drive,
            );

            tr.t.push(t as i32);
            tr.wx.push(wx);
            tr.wq.push(wq);
            tr.bias.push(bias_n);
            tr.z.push(z_nb);
            tr.q.push(q_nb);
        }
    }
}
