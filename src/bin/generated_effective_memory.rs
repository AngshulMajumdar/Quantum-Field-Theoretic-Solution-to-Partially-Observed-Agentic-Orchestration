// Port of run_generated_effective_memory.py
//
// Integrates out obsolete latent history in a finite two-state partially
// observed model, extracts the induced history-dependent effective coupling
// g(x0,x1), truncates it at threshold kappa, and measures the resulting
// decision loss against the theoretical certificate 2*tanh(kappa/2).

use common::*;
use serde_json::{json, Value};

const NSEED: u64 = 30;
const KAPPAS: [f64; 3] = [0.20, 0.40, 0.70];
const Z: [i32; 2] = [-1, 1];
const X: [i32; 2] = [-1, 1];

fn trans_prob(zp: i32, z: i32, stay: f64) -> f64 {
    if zp == z {
        stay
    } else {
        1.0 - stay
    }
}

fn obs_prob(x: i32, z: i32, acc: f64) -> f64 {
    if x == z {
        acc
    } else {
        1.0 - acc
    }
}

struct Record {
    ph: f64,
    g: f64,
    pplus: f64,
    exact_a: i32,
    exact_cost: f64,
}

struct ModelOut {
    stay: f64,
    acc: f64,
    prior_plus: f64,
    max_fit_err: f64,
    kappa_out: Vec<(f64, f64, f64)>, // (policy_loss, retained_history_mass, max_eta) per kappa
}

fn one_model(seed: u64) -> ModelOut {
    let mut rng = rng_from_seed(seed);
    let stay = sample_uniform(&mut rng, 0.72, 0.92);
    let acc = sample_uniform(&mut rng, 0.68, 0.90);
    let prior_plus = sample_uniform(&mut rng, 0.35, 0.65);
    let prior = |z: i32| -> f64 {
        if z == -1 {
            1.0 - prior_plus
        } else {
            prior_plus
        }
    };

    let mut records = Vec::new();
    let mut max_fit_err = 0.0f64;

    for &x0 in X.iter() {
        for &x1 in X.iter() {
            let mut joint = std::collections::HashMap::new();
            for &z2 in Z.iter() {
                let mut s = 0.0;
                for &z0 in Z.iter() {
                    for &z1 in Z.iter() {
                        s += prior(z0)
                            * obs_prob(x0, z0, acc)
                            * trans_prob(z1, z0, stay)
                            * obs_prob(x1, z1, acc)
                            * trans_prob(z2, z1, stay);
                    }
                }
                joint.insert(z2, s);
            }
            let ph = joint[&-1] + joint[&1];
            let pminus = joint[&-1] / ph;
            let pplus = joint[&1] / ph;
            let g = 0.5 * (pplus / pminus).ln();
            let sminus = -pminus.ln();
            let splus = -pplus.ln();
            let c = 0.5 * (sminus + splus);
            let fit_err = (sminus - (c + g)).abs().max((splus - (c - g)).abs());
            max_fit_err = max_fit_err.max(fit_err);
            let exact_a = if g >= 0.0 { 1 } else { -1 };
            let exact_cost = if exact_a == 1 { pminus } else { pplus };
            records.push(Record {
                ph,
                g,
                pplus,
                exact_a,
                exact_cost,
            });
        }
    }

    let mut kappa_out = Vec::new();
    for &kappa in KAPPAS.iter() {
        let mut loss = 0.0;
        let mut retained_mass = 0.0;
        let mut max_eta = 0.0f64;
        for r in records.iter() {
            let g = r.g;
            let (ghat, eta) = if g.abs() > kappa {
                retained_mass += r.ph;
                (g, 0.0)
            } else {
                (0.0, g.abs())
            };
            max_eta = max_eta.max(eta);
            let trunc_a = if ghat >= 0.0 { 1 } else { -1 };
            let trunc_cost = if trunc_a == 1 {
                1.0 - r.pplus
            } else {
                r.pplus
            };
            loss += r.ph * (trunc_cost - r.exact_cost);
        }
        kappa_out.push((loss, retained_mass, max_eta));
    }

    ModelOut {
        stay,
        acc,
        prior_plus,
        max_fit_err,
        kappa_out,
    }
}

fn main() {
    let models: Vec<ModelOut> = (0..NSEED).map(|i| one_model(9100 + i)).collect();

    let fit: Vec<f64> = models.iter().map(|m| m.max_fit_err).collect();
    let mut summary = json!({
        "nseed": NSEED,
        "kappas": KAPPAS,
        "effective_action_fit_error": mean_std_max(&fit),
    });

    for (ki, &kappa) in KAPPAS.iter().enumerate() {
        let losses: Vec<f64> = models.iter().map(|m| m.kappa_out[ki].0).collect();
        let retained: Vec<f64> = models.iter().map(|m| m.kappa_out[ki].1).collect();
        let etas: Vec<f64> = models.iter().map(|m| m.kappa_out[ki].2).collect();
        let certificate = 2.0 * (kappa / 2.0).tanh();
        let certificates = vec![certificate; models.len()];
        let all_certified = losses
            .iter()
            .zip(certificates.iter())
            .all(|(l, c)| *l <= *c + 1e-12);

        let key = format!("kappa_{:.2}", kappa);
        summary[key] = json!({
            "policy_loss": mean_std(&losses),
            "retained_history_mass": mean_std(&retained),
            "max_eta": mean_std(&etas),
            "certificate": mean_std(&certificates),
            "all_certified": all_certified,
        });
    }

    let models_json: Vec<Value> = models
        .iter()
        .enumerate()
        .map(|(_, m)| {
            let mut v = json!({
                "stay": m.stay,
                "acc": m.acc,
                "prior_plus": m.prior_plus,
                "max_fit_err": m.max_fit_err,
            });
            for (ki, &kappa) in KAPPAS.iter().enumerate() {
                let (loss, retained, eta) = m.kappa_out[ki];
                let certificate = 2.0 * (kappa / 2.0).tanh();
                v[format!("kappa_{:.2}", kappa)] = json!({
                    "policy_loss": loss,
                    "retained_history_mass": retained,
                    "max_eta": eta,
                    "certificate": certificate,
                });
            }
            v
        })
        .collect();

    let out = json!({ "summary": summary, "models": models_json });
    write_json("generated_effective_memory_results.json", &out);
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
}
