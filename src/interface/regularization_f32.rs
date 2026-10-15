// src/interface/regularization_f32.rs
pub mod rcs_regularization_f32 {
    extern crate arrayfire;
    use arrayfire as af;
    use af::{Array, Dim4};

    #[inline] fn eps() -> f32 { 1e-6 }

    // ---- helpers ---------------------------------------------------------
    #[inline] fn clamp01(x: &Array<f32>) -> Array<f32> {
        af::minof(
            &af::maxof(x, &af::constant::<f32>(0.0, x.dims()), false),
            &af::constant::<f32>(1.0, x.dims()),
            false,
        )
    }

    // bool -> f32 using select(a, cond, b)
    #[inline] fn to_f32_bool(cond: &Array<bool>) -> Array<f32> {
        af::select(
            &af::constant::<f32>(1.0, cond.dims()),
            cond,
            &af::constant::<f32>(0.0, cond.dims()),
        )
    }

    #[inline] fn l2norm_cols(x: &Array<f32>) -> Array<f32> {
        af::sqrt(&af::sum(&af::mul(x, x, false), 0)) // [1,B,1,1]
    }

    // d/du (1 - cos(u,v)) per column
    fn d_one_minus_cos_du(u: &Array<f32>, v: &Array<f32>) -> Array<f32> {
        // u,v: [N,B]
        let dot = af::sum(&af::mul(u, v, false), 0); // [1,B]
        let nu  = l2norm_cols(u) + af::constant::<f32>(eps(), dot.dims());
        let nv  = l2norm_cols(v) + af::constant::<f32>(eps(), dot.dims());

        let inv_nu = af::constant::<f32>(1.0, nu.dims()) / nu.clone(); // [1,B]
        let inv_nv = af::constant::<f32>(1.0, nv.dims()) / nv.clone(); // [1,B]
        let cos    = dot * (inv_nu.clone() * inv_nv.clone());          // [1,B]

        let cos_over_nu2 = cos * (inv_nu.clone() * inv_nu.clone());    // [1,B]
        let t1 = af::tile(&cos_over_nu2, Dim4::new(&[u.dims()[0], 1, 1, 1])) * u;      // [N,B]
        let t2 = af::tile(&(inv_nu * inv_nv), Dim4::new(&[v.dims()[0], 1, 1, 1])) * v; // [N,B]
        t1 - t2
    }

    #[inline]
    fn build_rates(spikes_t: &Array<f32>, v_t: &Array<f32>, v_reset: f32, v_thr: f32) -> Array<f32> {
        let denom = (v_thr - v_reset).max(1e-6);
        let norm_v = clamp01(&((v_t - af::constant::<f32>(v_reset, v_t.dims()))
                            /  af::constant::<f32>(denom,   v_t.dims())));
        0.5f32 * spikes_t + 0.5f32 * norm_v
    }

    #[inline]
    fn slice_nb_2d(x: &Array<f32>, t: i64, n: u64, b: u64) -> Array<f32> {
        let sl = af::slice(x, t); // [N,B,1,1]
        af::moddims(&sl, Dim4::new(&[n, b, 1, 1])) // -> [N,B]
    }

    #[inline]
    fn col_bool(x: &Array<bool>, col: i64, b: u64) -> Array<bool> {
        let c = af::cols(x, col, col); // [B,1,1,1]
        af::moddims(&c, Dim4::new(&[b, 1, 1, 1]))
    }

    /// Direct-training RCS for one layer.
    /// Q,Z: [N,B,T,1] (f32); pred_ok: [B,T] (bool)
    pub fn rcs_direct_layer(
        Q: &Array<f32>,
        Z: &Array<f32>,
        pred_ok: &Array<bool>,
        v_reset: f32,
        v_thr: f32,
        n_last: i32,
        alpha: f32,
    ) -> (f32, Array<f32>, Array<f32>) {
        if alpha == 0.0 {
            return (0.0, af::constant::<f32>(0.0, Q.dims()), af::constant::<f32>(0.0, Z.dims()));
        }

        // avoid temporary drop of dims()
        let dims_binding = Q.dims();
        let dQ = dims_binding.get();
        let n = dQ[0];
        let b = dQ[1];
        let tmax = dQ[2] as i64;
        let n_last = n_last.max(1) as i64;
        let m_req  = ((n_last as f32) * 0.5).ceil() as i64; // e.g., half of N
let mut ok_mask_all = af::constant::<bool>(false, Dim4::new(&[b, dQ[2], 1, 1]));
        let mut loss_acc: f32 = 0.0;
        let mut dQ_all = af::constant::<f32>(0.0, Q.dims());
        let mut dZ_all = af::constant::<f32>(0.0, Z.dims());

        // ok_mask_all: [B,T] bool — “all correct in last N”
        for t in 0..tmax {
    let all_true = col_bool(pred_ok, t, b); // [B,1,1,1]
    let s_rows_ok: af::Seq<f32> = af::Seq::<f32>::default();
    let s_time_ok: af::Seq<f32> = af::Seq::<f32>::new(t as f32, t as f32, 1.0);
    let mut idx_ok = af::Indexer::default();
    idx_ok.set_index(&s_rows_ok, 0, None);
    idx_ok.set_index(&s_time_ok, 1, None);
    af::assign_gen(&mut ok_mask_all, &idx_ok, &all_true);
}



        for t in 0..tmax {
            // eligible batches at time t
            let use_t_bool = af::cols(&ok_mask_all, t, t); // [B,1]
            let (any, _) = af::any_true_all(&use_t_bool);
            if !any { continue; }

            let use_t = af::moddims(&use_t_bool, Dim4::new(&[b, 1, 1, 1])); // [B,1] bool

            // r_t
            let spikes_t = slice_nb_2d(Q, t, n, b);
            let v_t      = slice_nb_2d(Z, t, n, b);
            let r_t = build_rates(&spikes_t, &v_t, v_reset, v_thr); // [N,B]

            // r_des = mean of last N rates
            let start = (t - n_last + 1).max(0);
            let mut acc = af::constant::<f32>(0.0, r_t.dims());
            let mut denom = 0.0f32;
            for k in start..=t {
                let r_k = build_rates(&slice_nb_2d(Q, k, n, b), &slice_nb_2d(Z, k, n, b), v_reset, v_thr);
                acc = acc + r_k;
                denom += 1.0;
            }
            let r_des = acc.clone() / af::constant::<f32>(denom.max(1.0), acc.dims()); // [N,B]

            // mask eligible batch items ([B,1] -> [N,B])
            let use_t_f = to_f32_bool(&use_t); // [B,1] f32
            let mask_nb = af::tile(&af::transpose(&use_t_f, false), Dim4::new(&[n, 1, 1, 1])); // [N,B]

            let r_t_m   = r_t.clone()   * mask_nb.clone();
            let r_des_m = r_des.clone() * mask_nb.clone();

            // loss_t = mean_b (1 - cos)
            let dot  = af::sum(&af::mul(&r_t_m, &r_des_m, false), 0); // [1,B]
            let nu   = l2norm_cols(&r_t_m)   + af::constant::<f32>(eps(), dot.dims());
            let nv   = l2norm_cols(&r_des_m) + af::constant::<f32>(eps(), dot.dims());
            let cos  = dot / (nu * nv); // [1,B]
            let one_minus = af::constant::<f32>(1.0, cos.dims()) - cos;
// one_minus: [1,B,1,1]
let use_t_f  = to_f32_bool(&use_t);                 // [B,1,1,1]
let use_cols = af::transpose(&use_t_f, false);      // [1,B,1,1]

// masked numerator & denominator
let num = af::sum_all::<f32>(&(&one_minus * &use_cols)).0;
let den = af::sum_all::<f32>(&use_cols).0.max(1e-6);
let l_t = num / den;
loss_acc += l_t;


            // grad wrt r_t, then split into dθ and dV
            let d_u = d_one_minus_cos_du(&r_t_m, &r_des_m); // [N,B]
            let dtheta = 0.5f32 * d_u.clone();
            let dnormV = 0.5f32 * d_u;

            let denom_nv = (v_thr - v_reset).max(1e-6);
            let in_low  = af::ge(&v_t, &af::constant::<f32>(v_reset, v_t.dims()), false);
            let in_high = af::le(&v_t, &af::constant::<f32>(v_thr,   v_t.dims()), false);
            let in_range = af::bitand(&in_low, &in_high, false);
            let dV = (dnormV.clone() / af::constant::<f32>(denom_nv, dnormV.dims()))
                     * to_f32_bool(&in_range);

            // scatter back to time dim
            let mut dQ_t = af::constant::<f32>(0.0, Q.dims());
            let mut dZ_t = af::constant::<f32>(0.0, Z.dims());
let s_all_q0: af::Seq<f32> = af::Seq::default();                   // N
    let s_all_q1: af::Seq<f32> = af::Seq::default();                   // B
    let s_time_q: af::Seq<f32> = af::Seq::new(t as f32, t as f32, 1.0);  // T

    let mut idx_q = af::Indexer::default();
    idx_q.set_index(&s_all_q0, 0, None);
    idx_q.set_index(&s_all_q1, 1, None);
    idx_q.set_index(&s_time_q, 2, None);

    af::assign_gen(
        &mut dQ_t,
        &idx_q,
        &af::moddims(&dtheta, af::Dim4::new(&[n, b, 1, 1])),
    );

 let s_all_z0: af::Seq<f32> = af::Seq::default();                   // N
    let s_all_z1: af::Seq<f32> = af::Seq::default();                   // B
    let s_time_z: af::Seq<f32> = af::Seq::new(t as f32, t as f32, 1.0);  // T

    let mut idx_z = af::Indexer::default();
    idx_z.set_index(&s_all_z0, 0, None);
    idx_z.set_index(&s_all_z1, 1, None);
    idx_z.set_index(&s_time_z, 2, None);

    af::assign_gen(
        &mut dZ_t,
        &idx_z,
        &af::moddims(&dV, af::Dim4::new(&[n, b, 1, 1])),
    );

            dQ_all = dQ_all + dQ_t;
            dZ_all = dZ_all + dZ_t;
        }

        let loss = alpha * (loss_acc / (tmax as f32).max(1.0));
        let dQ_all = af::constant::<f32>(alpha, dQ_all.dims()) * dQ_all;
        let dZ_all = af::constant::<f32>(alpha, dZ_all.dims()) * dZ_all;
        (loss, dQ_all, dZ_all)
    }
}
