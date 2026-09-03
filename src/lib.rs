// Shared numerical utilities for the Rust port of the QFT partially-observed
// orchestration experiment suite.
//
// Note on reproducibility (see original README, "Reproducibility" section):
// the Python reference uses NumPy's PCG64 bit generator. Bit-exact
// replication of that stream in another language/runtime is explicitly
// *not* the reproducibility target the original authors claim -- they
// state that floating point reduction order, RNG library/version, and
// platform math libraries are all expected to shift exact digits, and that
// wall-clock timings are not a deterministic statistic at all. This port
// follows the same standard: seeded, deterministic, statistically
// equivalent Monte Carlo (same seed count, same sample sizes, same
// distributions), not a bit-identical RNG stream.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::Serialize;

pub fn rng_from_seed(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / ((n - 1) as f64);
    (0..n).map(|i| a + step * (i as f64)).collect()
}

/// Trapezoidal integration against a strictly-increasing uniform grid `x`.
pub fn trapz(y: &[f64], x: &[f64]) -> f64 {
    assert_eq!(y.len(), x.len());
    let mut s = 0.0;
    for i in 0..y.len() - 1 {
        s += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) * 0.5;
    }
    s
}

/// Trapezoidal integration with a fixed step size `dx` (uniform grid).
pub fn trapz_dx(y: &[f64], dx: f64) -> f64 {
    if y.len() < 2 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..y.len() - 1 {
        s += y[i] + y[i + 1];
    }
    s * 0.5 * dx
}

pub fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let q = (x - mu) / sigma;
    (-0.5 * q * q).exp() / ((2.0 * std::f64::consts::PI).sqrt() * sigma)
}

pub fn sample_normal(rng: &mut StdRng, mu: f64, sigma: f64) -> f64 {
    let d = Normal::new(mu, sigma).unwrap();
    d.sample(rng)
}

pub fn sample_uniform(rng: &mut StdRng, lo: f64, hi: f64) -> f64 {
    rng.gen_range(lo..hi)
}

/// Sample a Dirichlet(alpha) vector via independent Gamma(alpha_i,1) draws.
pub fn sample_dirichlet(rng: &mut StdRng, alpha: &[f64]) -> Vec<f64> {
    use rand_distr::Gamma;
    let g: Vec<f64> = alpha
        .iter()
        .map(|&a| Gamma::new(a, 1.0).unwrap().sample(rng))
        .collect();
    let s: f64 = g.iter().sum();
    g.iter().map(|&v| v / s).collect()
}

/// Sample a categorical index in [0, p.len()) with given probabilities.
pub fn sample_categorical(rng: &mut StdRng, p: &[f64]) -> usize {
    let u: f64 = rng.gen_range(0.0..1.0);
    let mut acc = 0.0;
    for (i, &pi) in p.iter().enumerate() {
        acc += pi;
        if u < acc {
            return i;
        }
    }
    p.len() - 1
}

pub fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / (v.len() as f64)
}

/// Sample standard deviation, ddof=1 (matches numpy's std(ddof=1)).
pub fn std_ddof1(v: &[f64]) -> f64 {
    let m = mean(v);
    if v.len() < 2 {
        return 0.0;
    }
    let ss: f64 = v.iter().map(|x| (x - m) * (x - m)).sum();
    (ss / ((v.len() - 1) as f64)).sqrt()
}

#[derive(Serialize, Clone, Copy, Debug)]
pub struct MeanStd {
    pub mean: f64,
    pub std: f64,
}

pub fn mean_std(v: &[f64]) -> MeanStd {
    MeanStd {
        mean: mean(v),
        std: std_ddof1(v),
    }
}

#[derive(Serialize, Clone, Copy, Debug)]
pub struct MeanStdMax {
    pub mean: f64,
    pub std: f64,
    pub max: f64,
}

pub fn mean_std_max(v: &[f64]) -> MeanStdMax {
    MeanStdMax {
        mean: mean(v),
        std: std_ddof1(v),
        max: v.iter().cloned().fold(f64::MIN, f64::max),
    }
}

/// 1-D Wasserstein distance between two weighted empirical distributions
/// supported on point sets `u_vals`/`v_vals`, matching
/// scipy.stats.wasserstein_distance(u_values, v_values, u_weights, v_weights).
pub fn wasserstein_distance(
    u_vals: &[f64],
    v_vals: &[f64],
    u_weights: &[f64],
    v_weights: &[f64],
) -> f64 {
    let mut all: Vec<f64> = u_vals.iter().chain(v_vals.iter()).cloned().collect();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    let mut u_idx: Vec<usize> = (0..u_vals.len()).collect();
    u_idx.sort_by(|&a, &b| u_vals[a].partial_cmp(&u_vals[b]).unwrap());
    let mut v_idx: Vec<usize> = (0..v_vals.len()).collect();
    v_idx.sort_by(|&a, &b| v_vals[a].partial_cmp(&v_vals[b]).unwrap());

    let us: Vec<f64> = u_idx.iter().map(|&i| u_vals[i]).collect();
    let vs: Vec<f64> = v_idx.iter().map(|&i| v_vals[i]).collect();
    let uw: Vec<f64> = u_idx.iter().map(|&i| u_weights[i]).collect();
    let vw: Vec<f64> = v_idx.iter().map(|&i| v_weights[i]).collect();

    let uw_sum: f64 = uw.iter().sum();
    let vw_sum: f64 = vw.iter().sum();

    // all deduped points except the last, matching scipy's implementation
    let all_pts = &all[..all.len().saturating_sub(1)];

    // cumulative weight (CDF*total) evaluated just below each cut point
    let cdf_at = |pts: &[f64], w: &[f64], wsum: f64, cuts: &[f64]| -> Vec<f64> {
        let mut out = Vec::with_capacity(cuts.len());
        let mut cum = 0.0;
        let mut j = 0usize;
        for &c in cuts {
            while j < pts.len() && pts[j] <= c {
                cum += w[j];
                j += 1;
            }
            out.push(cum / wsum);
        }
        out
    };

    let u_cdf = cdf_at(&us, &uw, uw_sum, all_pts);
    let v_cdf = cdf_at(&vs, &vw, vw_sum, all_pts);

    let mut total = 0.0;
    for i in 0..all_pts.len() {
        let dx = all[i + 1] - all[i];
        total += (u_cdf[i] - v_cdf[i]).abs() * dx;
    }
    total
}

/// Minimal Levenberg-Marquardt solver for small dense nonlinear least
/// squares problems, used as a drop-in for scipy.optimize.least_squares in
/// the nonlinear BSD closure experiment. Jacobian is central-difference.
pub struct LmResult {
    pub x: Vec<f64>,
}

pub fn least_squares_lm<F>(residual: F, x0: &[f64], max_nfev: usize, tol: f64) -> LmResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut r = residual(&x);
    let m = r.len();
    let mut cost = r.iter().map(|v| v * v).sum::<f64>();
    let mut lambda = 1e-3;
    let mut nfev = 1usize;

    let jacobian = |x: &[f64], r0: &[f64], nfev: &mut usize| -> Vec<Vec<f64>> {
        let mut jac = vec![vec![0.0; n]; m];
        for j in 0..n {
            let h = (1e-7_f64).max(x[j].abs() * 1e-7);
            let mut xp = x.to_vec();
            xp[j] += h;
            let rp = residual(&xp);
            *nfev += 1;
            for i in 0..m {
                jac[i][j] = (rp[i] - r0[i]) / h;
            }
        }
        jac
    };

    let mut iter = 0usize;
    while nfev < max_nfev && iter < max_nfev {
        iter += 1;
        let jac = jacobian(&x, &r, &mut nfev);
        // Build normal equations J^T J + lambda*diag(J^T J), J^T r
        let mut jtj = vec![vec![0.0; n]; n];
        let mut jtr = vec![0.0; n];
        for i in 0..m {
            for a in 0..n {
                jtr[a] += jac[i][a] * r[i];
                for b in 0..n {
                    jtj[a][b] += jac[i][a] * jac[i][b];
                }
            }
        }

        // try a damped step; backtrack lambda upward on failure
        let mut accepted = false;
        for _ in 0..60 {
            let mut a = jtj.clone();
            for k in 0..n {
                a[k][k] += lambda * jtj[k][k].max(1e-12);
            }
            let rhs: Vec<f64> = jtr.iter().map(|v| -v).collect();
            if let Some(delta) = solve_linear(&a, &rhs) {
                let mut xn = x.clone();
                for k in 0..n {
                    xn[k] += delta[k];
                }
                let rn = residual(&xn);
                nfev += 1;
                let costn = rn.iter().map(|v| v * v).sum::<f64>();
                if costn < cost {
                    let improvement = (cost - costn) / cost.max(1e-300);
                    x = xn;
                    r = rn;
                    cost = costn;
                    lambda *= 0.5;
                    accepted = true;
                    if improvement < tol {
                        return LmResult { x };
                    }
                    break;
                } else {
                    lambda *= 3.0;
                }
            } else {
                lambda *= 3.0;
            }
            if nfev >= max_nfev {
                break;
            }
        }
        if !accepted {
            break;
        }
    }
    LmResult { x }
}

/// Solve a (possibly rectangular, tall) linear least squares problem
/// `rows * c ~= y` via the normal equations, matching the intercept-plus-
/// polynomial fits used to seed the BSD closure. `rows[i]` is one design
/// row (already weighted, if applicable).
pub fn lstsq_normal_eq(rows: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let ncols = rows[0].len();
    let mut ata = vec![vec![0.0; ncols]; ncols];
    let mut aty = vec![0.0; ncols];
    for (row, &yi) in rows.iter().zip(y.iter()) {
        for a in 0..ncols {
            aty[a] += row[a] * yi;
            for b in 0..ncols {
                ata[a][b] += row[a] * row[b];
            }
        }
    }
    solve_linear(&ata, &aty).unwrap_or_else(|| vec![0.0; ncols])
}

/// Solve A x = b via Gaussian elimination with partial pivoting.
pub fn solve_linear(a_in: &[Vec<f64>], b_in: &[f64]) -> Option<Vec<f64>> {
    let n = b_in.len();
    let mut a: Vec<Vec<f64>> = a_in.to_vec();
    let mut b: Vec<f64> = b_in.to_vec();
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                piv = r;
            }
        }
        if best < 1e-14 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in (col + 1)..n {
            let f = a[r][col] / a[col][col];
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// Write a serde_json::Value to disk pretty-printed, matching Python's
/// json.dump(..., indent=2).
pub fn write_json(path: &str, value: &serde_json::Value) {
    let s = serde_json::to_string_pretty(value).unwrap();
    std::fs::write(path, s).expect("failed to write output json");
}
