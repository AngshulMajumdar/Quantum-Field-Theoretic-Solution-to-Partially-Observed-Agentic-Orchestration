// Port of run_nonlinear_bsd.py
//
// Nonlinear partially observed process Z_{t+1}=0.74Z_t-0.035Z_t^3+A_t+W_t,
// X_t=Z_t+0.22Z_t^2+V_t. Solves direct finite-order (K=2,4,6) Schwinger-
// Dyson closures online against an exact-grid reference filter.

use common::*;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::time::Instant;

const SIG_W: f64 = 0.24;
const SIG_V: f64 = 0.32;
const ACTIONS: [f64; 3] = [-0.4, 0.0, 0.4];
const THETA: f64 = 1.1;
const T: usize = 10;
const NSEED: u64 = 30;
const MAXK: usize = 10;
const NGRID: usize = 301;

fn f(z: f64, u: f64) -> f64 {
    0.74 * z - 0.035 * z * z * z + u
}
fn h(z: f64) -> f64 {
    z + 0.22 * z * z
}
fn hp(z: f64) -> f64 {
    1.0 + 0.44 * z
}

struct Model {
    grid: Vec<f64>,
    dz: f64,
    pow: Vec<Vec<f64>>,           // pow[k][j] = grid[j]^k, k=0..=MAXK
    trans: HashMap<i64, Vec<Vec<f64>>>, // action(as fixed-point key) -> [i][j] matrix
}

fn action_key(u: f64) -> i64 {
    (u * 1e6).round() as i64
}

fn build_model() -> Model {
    let grid = linspace(-3.0, 3.0, NGRID);
    let dz = grid[1] - grid[0];
    let mut pow = vec![vec![0.0; NGRID]; MAXK + 1];
    for k in 0..=MAXK {
        for j in 0..NGRID {
            pow[k][j] = grid[j].powi(k as i32);
        }
    }
    let mut trans = HashMap::new();
    for &u in ACTIONS.iter() {
        let mut m = vec![vec![0.0; NGRID]; NGRID];
        for i in 0..NGRID {
            let mu = f(grid[i], u);
            for j in 0..NGRID {
                m[i][j] = normal_pdf(grid[j], mu, SIG_W);
            }
        }
        trans.insert(action_key(u), m);
    }
    Model { grid, dz, pow, trans }
}

fn normalize(p: &mut [f64], grid: &[f64]) {
    for v in p.iter_mut() {
        *v = v.max(1e-280);
    }
    let s = trapz(p, grid);
    for v in p.iter_mut() {
        *v /= s;
    }
}

fn moments(p: &[f64], kmax: usize, pow: &[Vec<f64>], grid: &[f64]) -> Vec<f64> {
    (0..=kmax)
        .map(|k| {
            let y: Vec<f64> = pow[k].iter().zip(p.iter()).map(|(a, b)| a * b).collect();
            trapz(&y, grid)
        })
        .collect()
}

fn predict(p: &[f64], u: f64, m: &Model) -> Vec<f64> {
    let trans = &m.trans[&action_key(u)];
    let mut out = vec![0.0; NGRID];
    for j in 0..NGRID {
        let col: Vec<f64> = (0..NGRID).map(|i| p[i] * trans[i][j]).collect();
        out[j] = trapz_dx(&col, m.dz);
    }
    normalize(&mut out, &m.grid);
    out
}

fn update(p: &[f64], x: f64, m: &Model) -> Vec<f64> {
    let mut out: Vec<f64> = p
        .iter()
        .zip(m.grid.iter())
        .map(|(&pi, &zj)| pi * normal_pdf(x, h(zj), SIG_V))
        .collect();
    normalize(&mut out, &m.grid);
    out
}

fn gradient_edge2(y: &[f64], dz: f64) -> Vec<f64> {
    let n = y.len();
    let mut g = vec![0.0; n];
    g[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * dz);
    g[n - 1] = (3.0 * y[n - 1] - 4.0 * y[n - 2] + y[n - 3]) / (2.0 * dz);
    for i in 1..n - 1 {
        g[i] = (y[i + 1] - y[i - 1]) / (2.0 * dz);
    }
    g
}

fn scoreprime(p: &[f64], m: &Model) -> Vec<f64> {
    let logp: Vec<f64> = p.iter().map(|&v| v.max(1e-280).ln()).collect();
    gradient_edge2(&logp, m.dz).iter().map(|&v| -v).collect()
}

fn qpoly(lam: &[f64], k: usize, m: &Model) -> Vec<f64> {
    let mut logq = vec![0.0; NGRID];
    for (i, &li) in lam.iter().enumerate().take(k) {
        for j in 0..NGRID {
            logq[j] -= li * m.pow[i + 1][j];
        }
    }
    let mx = logq.iter().cloned().fold(f64::MIN, f64::max);
    let mut q: Vec<f64> = logq.iter().map(|&v| (v - mx).exp()).collect();
    normalize(&mut q, &m.grid);
    q
}

fn initial_lam(target: &[f64], k: usize, m: &Model) -> Vec<f64> {
    let s: Vec<f64> = target.iter().map(|&v| -(v.max(1e-280).ln())).collect();
    let maxtarget = target.iter().cloned().fold(f64::MIN, f64::max);
    let mask: Vec<usize> = (0..NGRID)
        .filter(|&j| target[j] > 1e-5 * maxtarget)
        .collect();
    let masked_max = mask
        .iter()
        .map(|&j| target[j])
        .fold(f64::MIN, f64::max);

    let mut rows = Vec::with_capacity(mask.len());
    let mut y = Vec::with_capacity(mask.len());
    for &j in mask.iter() {
        let w = (target[j] / masked_max).sqrt();
        let mut row = vec![w; k + 1]; // will overwrite below
        row[0] = 1.0 * w;
        for kk in 1..=k {
            row[kk] = m.pow[kk][j] * w;
        }
        rows.push(row);
        y.push(s[j] * w);
    }
    let coef = lstsq_normal_eq(&rows, &y);
    coef[1..].to_vec()
}

fn fit_local_sd(score: &[f64], k: usize, lam_prev: Option<&[f64]>, m: &Model) -> (Vec<f64>, Vec<f64>, f64) {
    let sixth: Vec<f64> = score.iter().map(|&s| s * s / 6.0).collect();
    let scale = (1.0_f64).max(trapz(&sixth, &m.grid).sqrt());

    let mut lam0 = vec![0.0; k];
    match lam_prev {
        None => {
            let idx = if k > 1 { 1 } else { 0 };
            lam0[idx] = 0.5;
        }
        Some(prev) => {
            for i in 0..k.min(prev.len()) {
                lam0[i] = prev[i];
            }
        }
    }

    let residual = |lam: &[f64]| -> Vec<f64> {
        let q = qpoly(lam, k, m);
        let mm = moments(&q, k + 2, &m.pow, &m.grid);
        let mut rr = Vec::with_capacity(2 * k);
        for n in 0..k {
            let y: Vec<f64> = m.pow[n].iter().zip(score.iter()).zip(q.iter())
                .map(|((p, s), qq)| p * s * qq)
                .collect();
            let lhs = trapz(&y, &m.grid);
            let rhs = if n == 0 { 0.0 } else { (n as f64) * mm[n - 1] };
            rr.push((lhs - rhs) / scale);
        }
        for i in 0..k {
            rr.push(2e-5 * lam[i]);
        }
        rr
    };

    let sol = least_squares_lm(residual, &lam0, 2000, 1e-12);
    let lam = sol.x;

    let q = qpoly(&lam, k, m);
    let mm = moments(&q, k + 2, &m.pow, &m.grid);
    let mut rr = Vec::with_capacity(k);
    for n in 0..k {
        let y: Vec<f64> = m.pow[n].iter().zip(score.iter()).zip(q.iter())
            .map(|((p, s), qq)| p * s * qq)
            .collect();
        let lhs = trapz(&y, &m.grid);
        let rhs = if n == 0 { 0.0 } else { (n as f64) * mm[n - 1] };
        rr.push(lhs - rhs);
    }
    let proj = rr.iter().map(|v| v * v).sum::<f64>().sqrt();
    (q, lam, proj)
}

fn fit_initial(p: &[f64], k: usize, m: &Model) -> (Vec<f64>, Vec<f64>, f64) {
    let sp = scoreprime(p, m);
    let lam0 = initial_lam(p, k, m);
    fit_local_sd(&sp, k, Some(&lam0), m)
}

fn local_target_score(q: &[f64], u: f64, x: f64, m: &Model) -> Vec<f64> {
    let trans = &m.trans[&action_key(u)];
    let means: Vec<f64> = m.grid.iter().map(|&z| f(z, u)).collect();

    let mut den = vec![0.0; NGRID];
    let mut num = vec![0.0; NGRID];
    for j in 0..NGRID {
        let mut denj = Vec::with_capacity(NGRID);
        let mut numj = Vec::with_capacity(NGRID);
        for i in 0..NGRID {
            let dlogt = -(m.grid[j] - means[i]) / (SIG_W * SIG_W);
            let tv = q[i] * trans[i][j];
            denj.push(tv);
            numj.push(tv * dlogt);
        }
        den[j] = trapz_dx(&denj, m.dz);
        num[j] = trapz_dx(&numj, m.dz);
    }
    let mut out = vec![0.0; NGRID];
    for j in 0..NGRID {
        let pred_score = num[j] / den[j].max(1e-300);
        let obs_score = (x - h(m.grid[j])) * hp(m.grid[j]) / (SIG_V * SIG_V);
        out[j] = -(pred_score + obs_score);
    }
    out
}

fn vals(p: &[f64], m: &Model) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (i, &u) in ACTIONS.iter().enumerate() {
        let y: Vec<f64> = m.grid.iter().zip(p.iter())
            .map(|(&z, &pj)| (f(z, u).powi(2) + SIG_W * SIG_W) * pj)
            .collect();
        out[i] = trapz(&y, &m.grid) + 0.12 * u * u;
    }
    out
}

fn action_of(p: &[f64], m: &Model) -> f64 {
    let v = vals(p, m);
    let idx = (0..3).min_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap()).unwrap();
    ACTIONS[idx]
}

fn policy(mean_z: f64) -> [f64; 3] {
    let l: [f64; 3] = std::array::from_fn(|i| -THETA * (ACTIONS[i] + 0.45 * mean_z).abs());
    let mx = l.iter().cloned().fold(f64::MIN, f64::max);
    let lse = mx + l.iter().map(|&v| (v - mx).exp()).sum::<f64>().ln();
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = (l[i] - lse).exp();
    }
    out
}

fn grad(p: &[f64], m: &Model) -> f64 {
    let y: Vec<f64> = m.grid.iter().zip(p.iter()).map(|(&z, &pj)| z * pj).collect();
    let mean_z = trapz(&y, &m.grid);
    let pi = policy(mean_z);
    let q = vals(p, m);
    let d: [f64; 3] = std::array::from_fn(|i| -(ACTIONS[i] + 0.45 * mean_z).abs());
    let pidotd: f64 = (0..3).map(|i| pi[i] * d[i]).sum();
    let sc: [f64; 3] = std::array::from_fn(|i| d[i] - pidotd);
    (0..3).map(|i| pi[i] * q[i] * sc[i]).sum()
}

fn tv(p: &[f64], q: &[f64], m: &Model) -> f64 {
    let y: Vec<f64> = p.iter().zip(q.iter()).map(|(&a, &b)| (a - b).abs()).collect();
    0.5 * trapz(&y, &m.grid)
}

fn eta(p: &[f64], q: &[f64]) -> f64 {
    let d: Vec<f64> = p.iter().zip(q.iter())
        .map(|(&a, &b)| -(a.max(1e-280).ln()) + b.max(1e-280).ln())
        .collect();
    let mx = d.iter().cloned().fold(f64::MIN, f64::max);
    let mn = d.iter().cloned().fold(f64::MAX, f64::min);
    let c = 0.5 * (mx + mn);
    d.iter().map(|&v| (v - c).abs()).fold(0.0, f64::max)
}

fn sd_residual_against_reference(q: &[f64], pref: &[f64], nmax: usize, m: &Model) -> f64 {
    let sp = scoreprime(pref, m);
    let mm = moments(q, nmax + 1, &m.pow, &m.grid);
    let mut rr = Vec::with_capacity(nmax);
    for n in 0..nmax {
        let y: Vec<f64> = m.pow[n].iter().zip(sp.iter()).zip(q.iter())
            .map(|((p, s), qq)| p * s * qq)
            .collect();
        let lhs = trapz(&y, &m.grid);
        let rhs = if n == 0 { 0.0 } else { (n as f64) * mm[n - 1] };
        rr.push(lhs - rhs);
    }
    rr.iter().map(|v| v * v).sum::<f64>().sqrt()
}

struct Row {
    values: HashMap<&'static str, f64>,
}

fn simulate(seed: u64, k_opt: Option<usize>, m: &Model) -> (f64, Vec<Row>) {
    let mut rng = rng_from_seed(seed);
    let y0: Vec<f64> = m.grid.iter().map(|&z| {
        0.55 * normal_pdf(z, -0.85, 0.42) + 0.45 * normal_pdf(z, 0.75, 0.48)
    }).collect();
    let mut p0 = y0;
    normalize(&mut p0, &m.grid);
    let mut p = p0.clone();

    let mut pa: Option<Vec<f64>> = None;
    let mut lam: Option<Vec<f64>> = None;
    if let Some(k) = k_opt {
        let (q0, lam0, _) = fit_initial(&p0, k, m);
        pa = Some(q0);
        lam = Some(lam0);
    }

    let mut z = sample_normal(&mut rng, -0.1, 0.9).clamp(-2.5, 2.5);
    let mut cost = 0.0;
    let mut rows = Vec::new();

    for _ in 0..T {
        let pc: &[f64] = match k_opt {
            None => &p,
            Some(_) => pa.as_ref().unwrap(),
        };
        let u = action_of(pc, m);

        z = (f(z, u) + sample_normal(&mut rng, 0.0, SIG_W)).clamp(-2.9, 2.9);
        let x = h(z) + sample_normal(&mut rng, 0.0, SIG_V);
        cost += z * z + 0.12 * u * u;

        p = update(&predict(&p, u, m), x, m);

        if let Some(k) = k_opt {
            let t0 = Instant::now();
            let score = local_target_score(pa.as_ref().unwrap(), u, x, m);
            let (new_pa, new_lam, proj) = fit_local_sd(&score, k, lam.as_deref(), m);
            let ms = t0.elapsed().as_secs_f64() * 1e3;

            let mex = moments(&p, 6, &m.pow, &m.grid);
            let map_ = moments(&new_pa, 6, &m.pow, &m.grid);
            let e = eta(&p, &new_pa);
            let tverr = tv(&p, &new_pa, m);
            let kk6 = k.min(6);
            let moment_err = (1..=kk6)
                .map(|i| (mex[i] - map_[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            let wass = wasserstein_distance(&m.grid, &m.grid, &p, &new_pa);
            let vp = vals(&p, m);
            let vpa = vals(&new_pa, m);
            let q_err = (0..3).map(|i| (vp[i] - vpa[i]).abs()).fold(0.0, f64::max);
            let grad_err = (grad(&p, m) - grad(&new_pa, m)).abs();

            let mut values = HashMap::new();
            values.insert("sd_resid6", sd_residual_against_reference(&new_pa, &p, 6, m));
            values.insert("projection_resid", proj);
            values.insert("moment_err", moment_err);
            values.insert("tv", tverr);
            values.insert("wass", wass);
            values.insert("q_err", q_err);
            values.insert("grad_err", grad_err);
            values.insert("update_ms", ms);
            values.insert("eta", e);
            values.insert("tv_bound", (e / 2.0).tanh());
            values.insert("bound_holds", if tverr <= (e / 2.0).tanh() + 1e-10 { 1.0 } else { 0.0 });
            rows.push(Row { values });

            pa = Some(new_pa);
            lam = Some(new_lam);
        }
    }
    (cost, rows)
}

const ROW_KEYS: [&str; 11] = [
    "sd_resid6", "projection_resid", "moment_err", "tv", "wass",
    "q_err", "grad_err", "update_ms", "eta", "tv_bound", "bound_holds",
];

fn agg(k: usize, m: &Model) -> Value {
    let mut per_seed_costs = Vec::new();
    let mut per_seed_means: Vec<HashMap<&'static str, f64>> = Vec::new();

    for i in 0..NSEED {
        let (c, rows) = simulate(5000 + i, Some(k), m);
        per_seed_costs.push(c);
        let mut means = HashMap::new();
        for &key in ROW_KEYS.iter() {
            let vals: Vec<f64> = rows.iter().map(|r| r.values[key]).collect();
            means.insert(key, mean(&vals));
        }
        per_seed_means.push(means);
    }

    let mut out = Map::new();
    for &key in ROW_KEYS.iter() {
        let vals: Vec<f64> = per_seed_means.iter().map(|m| m[key]).collect();
        out.insert(key.to_string(), serde_json::to_value(mean_std(&vals)).unwrap());
    }
    out.insert("cost".to_string(), serde_json::to_value(mean_std(&per_seed_costs)).unwrap());
    Value::Object(out)
}

fn main() {
    let m = build_model();

    let mut out = Map::new();

    let exact_costs: Vec<f64> = (0..NSEED).map(|i| simulate(5000 + i, None, &m).0).collect();
    out.insert(
        "Exact-grid".to_string(),
        json!({ "cost": mean_std(&exact_costs) }),
    );

    for &k in [2usize, 4, 6].iter() {
        eprintln!("K {}", k);
        out.insert(format!("BSD-{}", k), agg(k, &m));
    }

    let out = Value::Object(out);
    write_json("nonlinear_results.json", &out);
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
