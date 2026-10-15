//! Supervised SNN objective functions (f32 flavour)
extern crate arrayfire;
use arrayfire::{Array, Dim4};

use arrayfire as af;
const EPS: f32 = 1e-8;
// --- helper ---------------------------------------------------------------
pub fn spike_counts(spikes: &Array<f32>) -> Array<f32> {
    arrayfire::sum(spikes, 2)
}
fn af_stats_f32(name: &str, a: &af::Array<f32>) {
    let (mn, _) = af::min_all(a);
    let (mx, _) = af::max_all(a);
    let (mean, _) = af::mean_all(a);
    println!(
        "[{}] dims={:?} min={} max={} mean={}",
        name,
        a.dims(),
        mn,
        mx,
        mean
    );
}

// --- 1) Rate-coded soft-max cross-entropy -------------------------------
pub fn rate_softmax_ce(
    spikes:  &Array<f32>,   // [out, batch, T, 1]
    targets: &Array<f32>,   // one-hot [out, batch, 1, 1]
) -> f32 {
    // logits = spike counts
    let mut logits = spike_counts(spikes);               // [out, batch, 1, 1]

    // numerical stability
    let max_per_batch = arrayfire::max(&logits, 0);
    logits = logits - max_per_batch;

    let exp_logits = arrayfire::exp(&logits);
    let sum_exp    = arrayfire::sum(&exp_logits, 0) + EPS;
    let probs      = arrayfire::div(&exp_logits, &sum_exp, true);

    let log_p  = arrayfire::log(&(probs + EPS));
    let ce_mat = -arrayfire::mul(targets, &log_p, false);
   let (sum_ce, _) = arrayfire::sum_all(&ce_mat);              // scalar
    let batch = targets.dims()[1] as f32;
    sum_ce as f32 / batch
}

/// dCE / d(logits)
pub fn rate_softmax_ce_grad(
    spikes:  &Array<f32>,
    targets: &Array<f32>,
) -> Array<f32> {
    let logits         = {
        let mut tmp: Array<f32> = spike_counts(spikes);
        let max_b   = arrayfire::max(&tmp, 0);
        tmp - max_b
    };
    let exp_logits = arrayfire::exp(&logits);
    let sum_exp    = arrayfire::sum(&exp_logits, 0) + EPS;
    let probs      = arrayfire::div(&exp_logits, &sum_exp, true);

    let batch = targets.dims()[1] as f32;
    (probs - targets) / batch
}


pub fn mse_spike_rate(
    spikes:  &af::Array<f32>,   // [out,batch,T,1]
    targets: &af::Array<f32>,   // rate targets in [0,1], [out,batch,1,1]
) -> f32 {
    let T = spikes.dims()[2] as f32;
    let r = spike_counts(spikes) / T;
    let diff = r - targets;
    let sq = af::mul(&diff, &diff, false);
    let (sum_sq, _) = af::sum_all(&sq);
    (sum_sq as f32) / (targets.dims()[1] as f32)
}

pub fn mse_spike_rate_grad(
    spikes:  &af::Array<f32>,
    targets: &af::Array<f32>,
) -> af::Array<f32> {
    let T = spikes.dims()[2] as f32;
    let r = spike_counts(spikes) / T;
    let b = targets.dims()[1] as f32;
    let dlogits = (r - targets) * (2.0f32 / (b * T)); // extra /T from r=c/T
    let Trep = spikes.dims()[2];
    af::tile(&dlogits, af::Dim4::new(&[1, 1, Trep, 1]))
}

pub fn mse_spike_rate_seq(
    spikes:  &af::Array<f32>,   // [out,batch,K,1]   (K = T*I_T)
    targets: &af::Array<f32>,   // [out,batch,I_T,1]
    T: u64,
) -> f32 {
    // 1) reshape spikes to separate T and I_T
    let dims = spikes.dims();
    let out = dims[0];
    let b   = dims[1];
    let K   = dims[2];
    let IT  = targets.dims()[2];
    assert_eq!(K, T * IT);

    // spikes4: [out,batch,T,I_T]
    let spikes4 = af::moddims(
        spikes,
        af::Dim4::new(&[out, b, T, IT])
    );

    // 2) count spikes over T: [out,batch,1,I_T]
    let counts = af::sum(&spikes4, 2);
    let r = counts / (T as f32);  // [out,batch,1,I_T]

    // 3) reshape r to [out,batch,I_T,1] to match targets
    let r2 = af::moddims(&r, af::Dim4::new(&[out, b, IT, 1]));

    let diff = r2 - targets;      // [out,batch,I_T,1]
    let sq   = af::mul(&diff, &diff, false);
    let (sum_sq, _) = af::sum_all(&sq);

    // average over batch * I_T
    let denom = (b * IT) as f32;
    (sum_sq as f32) / denom
}
pub fn mse_spike_rate_seq_grad(
    spikes:  &af::Array<f32>,   // [out,B,K,1] where K=IT*T
    targets: &af::Array<f32>,   // [out,B,IT,1]
) -> af::Array<f32> {
    let T = 1;
    let d = spikes.dims();
    let out = d[0];
    let b   = d[1];
    let k   = d[2];
    let it  = targets.dims()[2];
    assert_eq!(k, it * T);

    // IMPORTANT: choose the reshape that matches your flattening convention.
    // If your K is laid out as [IT blocks][T inside], use:
    let spikes4 = af::moddims(spikes, af::Dim4::new(&[out, b, it, T])); // [out,B,IT,T]

    let counts = af::sum(&spikes4, 3);            // sum over T -> [out,B,IT,1]
    let r      = counts / (T as f32);             // rate per IT block

    // dL/dr averaged over (B*IT)
    let denom = (b * it) as f32;
    let dldr  = (r - targets) * (2.0f32 / denom); // [out,B,IT,1]

    // dr/ds = 1/T
    let dldspk = dldr * (1.0f32 / (T as f32));    // [out,B,IT,1]

    // tile across T inside each IT block: [out,B,IT,T]
    let grad4 = af::tile(&dldspk, af::Dim4::new(&[1, 1, 1, T]));

    // flatten back to [out,B,K,1]
    af::moddims(&grad4, af::Dim4::new(&[out, b, k, 1]))
}

pub fn ce_spike_rate_global_grad(
    spikes:  &af::Array<f32>,   // [out,b,K,1]
    targets: &af::Array<f32>,   // [out,b,1,1]
    alpha: f32,
    eps: f32,
) -> af::Array<f32> {
    let dims = spikes.dims();
    let out = dims[0];
    let b   = dims[1];
    let k   = dims[2]; // K

    // counts: [out,b,1,1]
    let counts = af::sum(spikes, 2);
    let rate   = counts / (k as f32);

    // logits: [out,b,1,1]
    let logits = &rate * alpha;

    // stable softmax over dim 0 (class dim)
    let max_logits = af::max(&logits, 0); // [1,b,1,1]
    let max_logits = af::tile(&max_logits, af::Dim4::new(&[out, 1, 1, 1]));
    let shifted = logits - max_logits;

    let exps = af::exp(&shifted);
    let sum_exps = af::sum(&exps, 0); // [1,b,1,1]
    let sum_exps = af::tile(&sum_exps, af::Dim4::new(&[out, 1, 1, 1]));
    let softmax = exps / (sum_exps + eps); // [out,b,1,1]

    // dL/dlogits = (softmax - targets) / b   (mean over batch)
    let dldlogits = (softmax - targets) / (b as f32);

    // dL/dr = alpha * dL/dlogits
    let dldr = dldlogits * alpha; // [out,b,1,1]

    // each spike contributes 1/K to the rate
    let dlds = dldr * (1.0f32 / (k as f32)); // [out,b,1,1]

    // broadcast to every spike time step -> [out,b,K,1]
    af::tile(&dlds, af::Dim4::new(&[1, 1, k, 1]))
}


pub fn ce_spike_rate_global(
    spikes:  &af::Array<f32>,   // [out,b,K,1]
    targets: &af::Array<f32>,   // [out,b,1,1] one-hot
    alpha: f32,
    eps: f32,
) -> f32 {
    let dims = spikes.dims();
    let out = dims[0];
    let b   = dims[1];
    let k   = dims[2];

    // rate: [out,b,1,1]
    let counts = af::sum(spikes, 2);
    let rate   = counts / (k as f32);

    // logits: [out,b,1,1]
    let logits = &rate * alpha;
//     let mean_over_c = af::mean(&rate, 0);                         // [1,b,1,1]
// let mean_over_c = af::tile(&mean_over_c, af::Dim4::new(&[out,1,1,1]));
// let centered = &rate - mean_over_c;
// let logits   = centered * alpha;

    // stable log-softmax over class dim (dim 0)
    let max_logits = af::max(&logits, 0); // [1,b,1,1]
    let max_logits = af::tile(&max_logits, af::Dim4::new(&[out, 1, 1, 1]));
    let shifted = &logits - max_logits;

    let exps = af::exp(&shifted);
    let sum_exps = af::sum(&exps, 0); // [1,b,1,1]
    let sum_exps = af::tile(&sum_exps, af::Dim4::new(&[out, 1, 1, 1]));

    let log_sum_exps = af::log(&(sum_exps + eps)); // [out,b,1,1]
    let log_softmax  = shifted - log_sum_exps;     // [out,b,1,1]

    // CE = - mean_b sum_c y * log_softmax
    let per_entry = -(targets * log_softmax);      // [out,b,1,1]
    let loss_mat  = af::sum(&per_entry, 0);        // [1,b,1,1]  sum over classes
    let (loss_sum, _) = af::sum_all(&loss_mat);    // scalar sum over batch
    (loss_sum as f32) / (b as f32)
}
