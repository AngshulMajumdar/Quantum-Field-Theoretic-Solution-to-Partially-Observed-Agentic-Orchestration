// Port of run_agentic_routing.py
//
// Three-class/three-specialist partially observed routing problem. The
// orchestrator receives noisy task-class evidence, invokes a specialist,
// observes success/failure, and routes subsequent invocations using either
// a persistent categorical BSD state or various finite-memory/memoryless
// comparators.

use common::*;
use rand::Rng;
use serde_json::json;
use std::time::Instant;

const NSEED: u64 = 30;
const NEP: usize = 200;
const H: usize = 20;
const N: usize = 3;
const OBS_ACC: f64 = 0.64;

fn success_matrix() -> [[f64; 3]; 3] {
    [
        [0.92, 0.50, 0.45],
        [0.48, 0.90, 0.52],
        [0.50, 0.46, 0.91],
    ]
}
fn call_cost() -> [f64; 3] {
    [0.05, 0.06, 0.04]
}
fn prior() -> [f64; 3] {
    [0.34, 0.33, 0.33]
}
fn obs_matrix() -> [[f64; 3]; 3] {
    let off = (1.0 - OBS_ACC) / ((N - 1) as f64);
    let mut m = [[off; 3]; 3];
    for i in 0..3 {
        m[i][i] = OBS_ACC;
    }
    m
}

fn norm(b: &mut [f64; 3]) {
    for v in b.iter_mut() {
        *v = v.max(1e-300);
    }
    let s: f64 = b.iter().sum();
    for v in b.iter_mut() {
        *v /= s;
    }
}

fn cue_update(b: &[f64; 3], x: usize, obs: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut nb = [0.0; 3];
    for i in 0..3 {
        nb[i] = b[i] * obs[i][x];
    }
    norm(&mut nb);
    nb
}

fn outcome_update(b: &[f64; 3], a: usize, ok: bool, success: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut nb = [0.0; 3];
    for i in 0..3 {
        let like = if ok { success[i][a] } else { 1.0 - success[i][a] };
        nb[i] = b[i] * like;
    }
    norm(&mut nb);
    nb
}

fn belief_from_records(
    records: &[(usize, usize, bool)],
    current_cue: usize,
    obs: &[[f64; 3]; 3],
    success: &[[f64; 3]; 3],
) -> [f64; 3] {
    let mut b = prior();
    for &(x, a, ok) in records.iter() {
        b = cue_update(&b, x, obs);
        b = outcome_update(&b, a, ok, success);
    }
    cue_update(&b, current_cue, obs)
}

fn route(b: &[f64; 3], success: &[[f64; 3]; 3], call: &[f64; 3]) -> usize {
    let mut best = 0usize;
    let mut best_val = f64::INFINITY;
    for a in 0..3 {
        let mut expected = call[a];
        for z in 0..3 {
            expected += b[z] * (1.0 - success[z][a]);
        }
        if expected < best_val {
            best_val = expected;
            best = a;
        }
    }
    best
}

fn run(seed: u64, mode: &str) -> (f64, f64, f64) {
    let obs = obs_matrix();
    let success = success_matrix();
    let call = call_cost();
    let pr = prior();
    let oracle: [usize; 3] = {
        let mut o = [0usize; 3];
        for z in 0..3 {
            let mut best = 0usize;
            let mut best_val = f64::INFINITY;
            for a in 0..3 {
                let v = (1.0 - success[z][a]) + call[a];
                if v < best_val {
                    best_val = v;
                    best = a;
                }
            }
            o[z] = best;
        }
        o
    };

    let mut rng = rng_from_seed(seed);
    let mut costs = Vec::with_capacity(NEP * H);
    let mut correct = Vec::with_capacity(NEP * H);
    let mut runt = Vec::with_capacity(NEP * H);

    for _ in 0..NEP {
        let z = sample_categorical(&mut rng, &pr);
        let mut b = pr;
        let mut records: Vec<(usize, usize, bool)> = Vec::new();

        for _ in 0..H {
            let x = sample_categorical(&mut rng, &obs[z]);
            let t0 = Instant::now();
            let b_for_action = match mode {
                "BSD-categorical" => {
                    b = cue_update(&b, x, &obs);
                    b
                }
                "window-2" => {
                    let start = records.len().saturating_sub(2);
                    belief_from_records(&records[start..], x, &obs, &success)
                }
                "window-1" => {
                    let start = records.len().saturating_sub(1);
                    belief_from_records(&records[start..], x, &obs, &success)
                }
                "memoryless-current-cue" => cue_update(&pr, x, &obs),
                _ => panic!("unknown mode"),
            };
            let a = route(&b_for_action, &success, &call);
            runt.push(t0.elapsed().as_secs_f64() * 1e6);
            let ok = rng.gen_range(0.0..1.0) < success[z][a];
            costs.push((if ok { 0.0 } else { 1.0 }) + call[a]);
            correct.push(if a == oracle[z] { 1.0 } else { 0.0 });
            if mode == "BSD-categorical" {
                b = outcome_update(&b, a, ok, &success);
            }
            records.push((x, a, ok));
        }
    }

    (mean(&costs), mean(&correct), mean(&runt))
}

fn main() {
    let modes = ["BSD-categorical", "window-2", "window-1", "memoryless-current-cue"];
    let obs = obs_matrix();
    let success = success_matrix();
    let call = call_cost();
    let pr = prior();
    let oracle: [usize; 3] = {
        let mut o = [0usize; 3];
        for z in 0..3 {
            let mut best = 0usize;
            let mut best_val = f64::INFINITY;
            for a in 0..3 {
                let v = (1.0 - success[z][a]) + call[a];
                if v < best_val {
                    best_val = v;
                    best = a;
                }
            }
            o[z] = best;
        }
        o
    };

    let mut summary = serde_json::Map::new();
    let mut raw = serde_json::Map::new();

    for &mode in modes.iter() {
        let rows: Vec<(f64, f64, f64)> = (0..NSEED).map(|i| run(12000 + i, mode)).collect();
        let cost_v: Vec<f64> = rows.iter().map(|r| r.0).collect();
        let acc_v: Vec<f64> = rows.iter().map(|r| r.1).collect();
        let us_v: Vec<f64> = rows.iter().map(|r| r.2).collect();
        summary.insert(
            mode.to_string(),
            json!({
                "cost_per_step": mean_std(&cost_v),
                "routing_accuracy": mean_std(&acc_v),
                "update_us": mean_std(&us_v),
            }),
        );
        raw.insert(mode.to_string(), json!(rows));
    }

    let out = json!({
        "parameters": {
            "cue_accuracy": OBS_ACC,
            "cue_matrix": obs,
            "success_matrix": success,
            "invocation_costs": call,
            "prior": pr,
            "oracle_agent_by_task_class": oracle,
            "success_failure_used_as_next_belief_evidence": true,
            "routing_accuracy_definition": "fraction of routing decisions selecting argmin_a[1-P(success|true task class,a)+invocation_cost[a]]"
        },
        "summary": summary,
        "raw": raw,
    });

    write_json("agentic_routing_results.json", &out);
    println!("{}", serde_json::to_string_pretty(&out["parameters"]).unwrap());
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
}
