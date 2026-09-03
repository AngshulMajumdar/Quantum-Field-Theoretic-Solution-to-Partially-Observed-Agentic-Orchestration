// Port of run_continuous_baselines.py
//
// EKF and particle-filter baselines on the nonlinear continuous model,
// evaluated against an exact-grid reference filter as an
// accuracy/runtime/memory frontier.

use common::*;
use rand::Rng;
use serde_json::json;
use std::time::Instant;

const SIG_W: f64 = 0.24;
const SIG_V: f64 = 0.32;
const ACTIONS: [f64; 3] = [-0.4, 0.0, 0.4];
const T: usize = 10;
const NSEED: u64 = 30;
const NGRID: usize = 301;
const PF_COUNTS: [usize; 11] = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];

fn f(z: f64, u: f64) -> f64 {
    0.74 * z - 0.035 * z * z * z + u
}
fn fp(z: f64) -> f64 {
    0.74 - 0.105 * z * z
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
    p0: Vec<f64>,
    trans: std::collections::HashMap<i64, Vec<Vec<f64>>>,
}

fn action_key(u: f64) -> i64 {
    (u * 1e6).round() as i64
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

fn build_model() -> Model {
    let grid = linspace(-3.0, 3.0, NGRID);
    let dz = grid[1] - grid[0];
    let mut p0: Vec<f64> = grid
        .iter()
        .map(|&z| 0.55 * normal_pdf(z, -0.85, 0.42) + 0.45 * normal_pdf(z, 0.75, 0.48))
        .collect();
    normalize(&mut p0, &grid);

    let mut trans = std::collections::HashMap::new();
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
    Model { grid, dz, p0, trans }
}

fn predict_grid(p: &[f64], u: f64, m: &Model) -> Vec<f64> {
    let trans = &m.trans[&action_key(u)];
    let mut out = vec![0.0; NGRID];
    for j in 0..NGRID {
        let col: Vec<f64> = (0..NGRID).map(|i| p[i] * trans[i][j]).collect();
        out[j] = trapz_dx(&col, m.dz);
    }
    normalize(&mut out, &m.grid);
    out
}

fn update_grid(p: &[f64], x: f64, m: &Model) -> Vec<f64> {
    let mut out: Vec<f64> = p
        .iter()
        .zip(m.grid.iter())
        .map(|(&pi, &zj)| pi * normal_pdf(x, h(zj), SIG_V))
        .collect();
    normalize(&mut out, &m.grid);
    out
}

fn vals_grid(p: &[f64], m: &Model) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (i, &u) in ACTIONS.iter().enumerate() {
        let y: Vec<f64> = m
            .grid
            .iter()
            .zip(p.iter())
            .map(|(&z, &pj)| (f(z, u).powi(2) + SIG_W * SIG_W) * pj)
            .collect();
        out[i] = trapz(&y, &m.grid) + 0.12 * u * u;
    }
    out
}

fn act_grid(p: &[f64], m: &Model) -> f64 {
    let v = vals_grid(p, m);
    let idx = (0..3).min_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap()).unwrap();
    ACTIONS[idx]
}

fn vals_particles(z: &[f64]) -> [f64; 3] {
    let n = z.len() as f64;
    let mut out = [0.0; 3];
    for (i, &u) in ACTIONS.iter().enumerate() {
        let s: f64 = z.iter().map(|&zi| f(zi, u).powi(2) + SIG_W * SIG_W).sum();
        out[i] = s / n + 0.12 * u * u;
    }
    out
}

fn gauss_points(mean_: f64, p_var: f64) -> ([f64; 3], [f64; 3]) {
    let p_var = p_var.max(1e-10);
    let d = (3.0 * p_var).sqrt();
    ([mean_, mean_ - d, mean_ + d], [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
}

fn vals_gauss(mean_: f64, p_var: f64) -> [f64; 3] {
    let (pts, w) = gauss_points(mean_, p_var);
    let mut out = [0.0; 3];
    for (i, &u) in ACTIONS.iter().enumerate() {
        let s: f64 = (0..3).map(|k| w[k] * (f(pts[k], u).powi(2) + SIG_W * SIG_W)).sum();
        out[i] = s + 0.12 * u * u;
    }
    out
}

fn gauss_density(mean_: f64, p_var: f64, m: &Model) -> Vec<f64> {
    let sd = p_var.max(1e-10).sqrt();
    let mut out: Vec<f64> = m.grid.iter().map(|&z| normal_pdf(z, mean_, sd)).collect();
    normalize(&mut out, &m.grid);
    out
}

fn systematic_resample(w: &[f64], rng: &mut rand::rngs::StdRng) -> Vec<usize> {
    let n = w.len();
    let u0: f64 = sample_uniform(rng, 0.0, 1.0);
    let positions: Vec<f64> = (0..n).map(|i| (u0 + i as f64) / (n as f64)).collect();
    let mut cumsum = vec![0.0; n];
    let mut acc = 0.0;
    for i in 0..n {
        acc += w[i];
        cumsum[i] = acc;
    }
    // searchsorted(cumsum, positions, side='left')
    positions
        .iter()
        .map(|&pos| {
            match cumsum.binary_search_by(|v| v.partial_cmp(&pos).unwrap()) {
                Ok(idx) => idx,
                Err(idx) => idx,
            }
            .min(n - 1)
        })
        .collect()
}

fn sample_p0_particles(rng: &mut rand::rngs::StdRng, n: usize) -> Vec<f64> {
    (0..n)
        .map(|_| {
            let mix: f64 = rng.gen_range(0.0..1.0);
            let z = if mix < 0.55 {
                sample_normal(rng, -0.85, 0.42)
            } else {
                sample_normal(rng, 0.75, 0.48)
            };
            z.clamp(-3.0, 3.0)
        })
        .collect()
}

fn initial_gaussian(m: &Model) -> (f64, f64) {
    let y: Vec<f64> = m.grid.iter().zip(m.p0.iter()).map(|(&z, &p)| z * p).collect();
    let mean_ = trapz(&y, &m.grid);
    let y2: Vec<f64> = m
        .grid
        .iter()
        .zip(m.p0.iter())
        .map(|(&z, &p)| (z - mean_).powi(2) * p)
        .collect();
    let p_var = trapz(&y2, &m.grid);
    (mean_, p_var)
}

enum Kind {
    PF(usize),
    Ekf,
    ExactGrid,
}

fn parse_method(method: &str) -> Kind {
    if let Some(rest) = method.strip_prefix("PF-") {
        Kind::PF(rest.parse().unwrap())
    } else if method == "EKF" {
        Kind::Ekf
    } else {
        Kind::ExactGrid
    }
}

struct RunOut {
    cost: f64,
    update_ms: f64,
    wasserstein: f64,
    q_err: f64,
    retained_scalars: usize,
}

fn one_run(method: &str, seed: u64, m: &Model) -> RunOut {
    let kind = parse_method(method);
    let np_val = match &kind {
        Kind::PF(n) => *n,
        _ => 0,
    };
    let mut rng = rng_from_seed(seed);
    let mut frng = rng_from_seed(seed + 100000 + np_val as u64);

    let mut z = sample_normal(&mut rng, -0.1, 0.9).clamp(-2.5, 2.5);
    let mut pref = m.p0.clone();
    let mut cost = 0.0;
    let mut times = Vec::with_capacity(T);
    let mut wass = Vec::with_capacity(T);
    let mut qerr = Vec::with_capacity(T);

    let mut particles: Vec<f64> = Vec::new();
    let mut ekf_mean = 0.0;
    let mut ekf_var = 0.0;
    let mut pm: Vec<f64> = Vec::new();

    match &kind {
        Kind::PF(n) => particles = sample_p0_particles(&mut frng, *n),
        Kind::Ekf => {
            let (mm, pp) = initial_gaussian(m);
            ekf_mean = mm;
            ekf_var = pp;
        }
        Kind::ExactGrid => pm = m.p0.clone(),
    }

    for _ in 0..T {
        let u = match &kind {
            Kind::PF(_) => {
                let v = vals_particles(&particles);
                ACTIONS[(0..3).min_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap()).unwrap()]
            }
            Kind::Ekf => {
                let v = vals_gauss(ekf_mean, ekf_var);
                ACTIONS[(0..3).min_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap()).unwrap()]
            }
            Kind::ExactGrid => act_grid(&pm, m),
        };

        z = (f(z, u) + sample_normal(&mut rng, 0.0, SIG_W)).clamp(-2.9, 2.9);
        let x = h(z) + sample_normal(&mut rng, 0.0, SIG_V);
        cost += z * z + 0.12 * u * u;

        pref = update_grid(&predict_grid(&pref, u, m), x, m);
        let exact_vals = vals_grid(&pref, m);

        let t0 = Instant::now();
        let vv: [f64; 3];
        match &kind {
            Kind::PF(n) => {
                let n = *n;
                let mut newp: Vec<f64> = particles.iter().map(|&zi| {
                    (f(zi, u) + sample_normal(&mut frng, 0.0, SIG_W)).clamp(-3.0, 3.0)
                }).collect();
                let w: Vec<f64> = newp.iter().map(|&zi| normal_pdf(x, h(zi), SIG_V)).collect();
                let sw: f64 = w.iter().sum();
                let wn: Vec<f64> = if sw > 0.0 {
                    w.iter().map(|&wi| wi / sw).collect()
                } else {
                    vec![1.0 / n as f64; n]
                };
                let idx = systematic_resample(&wn, &mut frng);
                newp = idx.iter().map(|&i| newp[i]).collect();
                vv = vals_particles(&newp);
                let ones = vec![1.0; n];
                wass.push(wasserstein_distance(&m.grid, &newp, &pref, &ones));
                particles = newp;
            }
            Kind::Ekf => {
                let mp = f(ekf_mean, u);
                let pp = fp(ekf_mean).powi(2) * ekf_var + SIG_W * SIG_W;
                let hh = hp(mp);
                let s = hh * hh * pp + SIG_V * SIG_V;
                let kg = pp * hh / s;
                let new_mean = mp + kg * (x - h(mp));
                let new_var = ((1.0 - kg * hh) * pp).max(1e-8);
                let q = gauss_density(new_mean, new_var, m);
                vv = vals_gauss(new_mean, new_var);
                wass.push(wasserstein_distance(&m.grid, &m.grid, &pref, &q));
                ekf_mean = new_mean;
                ekf_var = new_var;
            }
            Kind::ExactGrid => {
                pm = pref.clone();
                vv = vals_grid(&pm, m);
                wass.push(0.0);
            }
        }
        times.push(t0.elapsed().as_secs_f64() * 1e3);
        qerr.push((0..3).map(|i| (exact_vals[i] - vv[i]).abs()).fold(0.0, f64::max));
    }

    let retained_scalars = match &kind {
        Kind::ExactGrid => NGRID,
        Kind::Ekf => 2,
        Kind::PF(n) => *n,
    };

    RunOut {
        cost,
        update_ms: mean(&times),
        wasserstein: mean(&wass),
        q_err: mean(&qerr),
        retained_scalars,
    }
}

fn aggregate(method: &str, m: &Model) -> serde_json::Value {
    let rows: Vec<RunOut> = (0..NSEED).map(|i| one_run(method, 5000 + i, m)).collect();
    let cost: Vec<f64> = rows.iter().map(|r| r.cost).collect();
    let update_ms: Vec<f64> = rows.iter().map(|r| r.update_ms).collect();
    let wasserstein: Vec<f64> = rows.iter().map(|r| r.wasserstein).collect();
    let q_err: Vec<f64> = rows.iter().map(|r| r.q_err).collect();
    json!({
        "cost": mean_std(&cost),
        "update_ms": mean_std(&update_ms),
        "wasserstein": mean_std(&wasserstein),
        "q_err": mean_std(&q_err),
        "retained_scalars": rows[0].retained_scalars,
    })
}

fn main() {
    let m = build_model();
    let mut methods: Vec<String> = vec!["Exact grid".to_string(), "EKF".to_string()];
    for &n in PF_COUNTS.iter() {
        methods.push(format!("PF-{}", n));
    }

    let mut out = serde_json::Map::new();
    for method in methods.iter() {
        eprintln!("method {}", method);
        out.insert(method.clone(), aggregate(method, &m));
    }
    out.insert(
        "metadata".to_string(),
        json!({ "n_seeds": NSEED, "horizon": T, "particle_counts": PF_COUNTS }),
    );

    let out = serde_json::Value::Object(out);
    write_json("continuous_baseline_results.json", &out);
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
