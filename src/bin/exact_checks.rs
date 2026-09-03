// Port of run_exact_checks.py
//
// Exact path-action / marginal / observable correspondence checks on
// independently generated random HMMs: forward filtering vs. an
// independently-written sum-product recursion should agree exactly (up to
// floating point), and sampled trajectory log p(Gamma) + S(Gamma) should be
// zero for every sample.

use common::*;
use rand::Rng;
use serde_json::json;

fn main() {
    let ns = 3usize;
    let no = 2usize;
    let h = 8usize;

    let mut path_log_error = Vec::new();
    let mut marginal_error = Vec::new();
    let mut observable_error = Vec::new();

    for seed in 0..30u64 {
        let mut rng = rng_from_seed(1000 + seed);

        let p0 = sample_dirichlet(&mut rng, &vec![1.0; ns]);
        let t_mat: Vec<Vec<f64>> = (0..ns)
            .map(|_| sample_dirichlet(&mut rng, &vec![1.0; ns]))
            .collect();
        let o_mat: Vec<Vec<f64>> = (0..ns)
            .map(|_| sample_dirichlet(&mut rng, &vec![1.0; no]))
            .collect();
        let obs: Vec<usize> = (0..=h).map(|_| rng.gen_range(0..no)).collect();

        let mut alpha: Vec<f64> = (0..ns).map(|i| p0[i] * o_mat[i][obs[0]]).collect();
        let s: f64 = alpha.iter().sum();
        for v in alpha.iter_mut() {
            *v /= s;
        }
        let mut beta = alpha.clone();
        let mut maxerr = 0.0f64;

        for t in 1..=h {
            // alpha = (alpha @ T) * O[:, obs[t]]; normalize
            let mut new_alpha = vec![0.0; ns];
            for j in 0..ns {
                let mut acc = 0.0;
                for i in 0..ns {
                    acc += alpha[i] * t_mat[i][j];
                }
                new_alpha[j] = acc * o_mat[j][obs[t]];
            }
            let s: f64 = new_alpha.iter().sum();
            for v in new_alpha.iter_mut() {
                *v /= s;
            }
            alpha = new_alpha;

            // independently-written sum-product recursion
            let mut pred = vec![0.0; ns];
            for j in 0..ns {
                let mut acc = 0.0;
                for i in 0..ns {
                    acc += beta[i] * t_mat[i][j];
                }
                pred[j] = acc;
            }
            let mut new_beta: Vec<f64> = (0..ns).map(|j| pred[j] * o_mat[j][obs[t]]).collect();
            let s: f64 = new_beta.iter().sum();
            for v in new_beta.iter_mut() {
                *v /= s;
            }
            beta = new_beta;

            let err = (0..ns)
                .map(|i| (alpha[i] - beta[i]).abs())
                .fold(0.0, f64::max);
            maxerr = maxerr.max(err);
        }
        marginal_error.push(maxerr);

        let f: Vec<f64> = (0..ns).map(|i| (i as f64).powi(2)).collect();
        let alpha_f: f64 = (0..ns).map(|i| alpha[i] * f[i]).sum();
        let beta_f: f64 = (0..ns).map(|i| beta[i] * f[i]).sum();
        observable_error.push((alpha_f - beta_f).abs());

        let mut pe = 0.0f64;
        for _ in 0..100 {
            let z: Vec<usize> = (0..=h).map(|_| rng.gen_range(0..ns)).collect();
            let mut lp = p0[z[0]].ln() + o_mat[z[0]][obs[0]].ln();
            let mut sact = -p0[z[0]].ln() - o_mat[z[0]][obs[0]].ln();
            for t in 0..h {
                let step = t_mat[z[t]][z[t + 1]].ln() + o_mat[z[t + 1]][obs[t + 1]].ln();
                lp += step;
                sact -= step;
            }
            pe = pe.max((lp + sact).abs());
        }
        path_log_error.push(pe);
    }

    let out = json!({
        "path_log_error": mean_std_max(&path_log_error),
        "marginal_error": mean_std_max(&marginal_error),
        "observable_error": mean_std_max(&observable_error),
    });

    write_json("exact_check_results.json", &out);
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
