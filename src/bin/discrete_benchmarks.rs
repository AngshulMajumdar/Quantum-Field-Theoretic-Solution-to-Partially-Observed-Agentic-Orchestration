// Port of run_discrete_benchmarks.py
//
// Tiger and RockSample(4,4) benchmark families, comparing exact-belief
// (BSD-family) planners against finite-window and QMDP-style heuristic
// comparators.

use common::*;
use rand::Rng;
use serde_json::json;
use std::time::Instant;

const GAMMA: f64 = 0.95;

// ---------------- Tiger ----------------

const PC: f64 = 0.85;
const NGRID: usize = 12001;

fn interp(x: f64, grid: &[f64], v: &[f64]) -> f64 {
    let n = grid.len();
    let lo = grid[0];
    let hi = grid[n - 1];
    if x <= lo {
        return v[0];
    }
    if x >= hi {
        return v[n - 1];
    }
    let dz = (hi - lo) / ((n - 1) as f64);
    let pos = (x - lo) / dz;
    let i0 = pos.floor() as usize;
    let i0 = i0.min(n - 2);
    let frac = pos - i0 as f64;
    v[i0] * (1.0 - frac) + v[i0 + 1] * frac
}

fn bu(b: f64, o: usize) -> f64 {
    let num = (if o == 0 { PC } else { 1.0 - PC }) * b;
    let den = num + (if o == 1 { PC } else { 1.0 - PC }) * (1.0 - b);
    num / den
}

fn rew(b: f64, a: usize) -> f64 {
    match a {
        0 => -1.0,
        1 => 10.0 * (1.0 - b) - 100.0 * b,
        _ => 10.0 * b - 100.0 * (1.0 - b),
    }
}

fn solve_tiger_value() -> (Vec<f64>, Vec<f64>) {
    let grid = linspace(0.0, 1.0, NGRID);
    let mut v = vec![0.0; NGRID];
    for _it in 0..4000 {
        let cont = GAMMA * interp(0.5, &grid, &v);
        let mut vn = vec![0.0; NGRID];
        let mut maxdiff = 0.0f64;
        for j in 0..NGRID {
            let g = grid[j];
            let ql = 10.0 * (1.0 - g) - 100.0 * g + cont;
            let qr = 10.0 * g - 100.0 * (1.0 - g) + cont;
            let po = PC * g + (1.0 - PC) * (1.0 - g);
            let b0 = PC * g / po;
            let po1 = (1.0 - PC) * g + PC * (1.0 - g);
            let b1 = (1.0 - PC) * g / po1;
            let q0 = -1.0 + GAMMA * (po * interp(b0, &grid, &v) + po1 * interp(b1, &grid, &v));
            let val = q0.max(ql).max(qr);
            vn[j] = val;
            maxdiff = maxdiff.max((val - v[j]).abs());
        }
        v = vn;
        if maxdiff < 1e-11 {
            break;
        }
    }
    (grid, v)
}

fn tiger_exact(b: f64, grid: &[f64], v: &[f64]) -> usize {
    let cont = GAMMA * interp(0.5, grid, v);
    let po = PC * b + (1.0 - PC) * (1.0 - b);
    let q0 = -1.0 + GAMMA * (po * interp(bu(b, 0), grid, v) + (1.0 - po) * interp(bu(b, 1), grid, v));
    let q1 = rew(b, 1) + cont;
    let q2 = rew(b, 2) + cont;
    let qs = [q0, q1, q2];
    (0..3).max_by(|&a, &b2| qs[a].partial_cmp(&qs[b2]).unwrap()).unwrap()
}

fn tiger_qmdp(b: f64) -> usize {
    if rew(b, 1) >= rew(b, 2) {
        1
    } else {
        2
    }
}

fn tiger_window(hist: &[usize], w: usize, grid: &[f64], v: &[f64]) -> usize {
    let mut b = 0.5;
    let start = hist.len().saturating_sub(w);
    for &o in &hist[start..] {
        b = bu(b, o);
    }
    tiger_exact(b, grid, v)
}

fn tiger_ep(kind: &str, rng: &mut rand::rngs::StdRng, grid: &[f64], v: &[f64], h: usize) -> f64 {
    let mut s = rng.gen_range(0..2usize);
    let mut b = 0.5;
    let mut hist: Vec<usize> = Vec::new();
    let mut r = 0.0;
    let mut d = 1.0;
    for _ in 0..h {
        let a = match kind {
            "BSD (exact-belief family)" => tiger_exact(b, grid, v),
            "Finite window W=2" => tiger_window(&hist, 2, grid, v),
            "Finite window W=1" => tiger_window(&hist, 1, grid, v),
            "QMDP" => tiger_qmdp(b),
            _ => panic!("unknown kind"),
        };
        r += d * rew(if s == 0 { 1.0 } else { 0.0 }, a);
        d *= GAMMA;
        if a == 0 {
            let o = if rng.gen_range(0.0..1.0) < PC { s } else { 1 - s };
            hist.push(o);
            b = bu(b, o);
        } else {
            s = rng.gen_range(0..2usize);
            b = 0.5;
            hist.clear();
        }
    }
    r
}

// ---------------- RockSample(4,4) ----------------

const SIZE: i32 = 4;
const ROCKS: [(i32, i32); 4] = [(0, 1), (1, 3), (2, 0), (3, 2)];
const KROCK: usize = 4;
const D0: f64 = 3.0;

fn moves(dir: char) -> (i32, i32) {
    match dir {
        'N' => (0, 1),
        'S' => (0, -1),
        'E' => (1, 0),
        _ => (-1, 0),
    }
}

fn sensor_acc(pos: (i32, i32), i: usize) -> f64 {
    let (rx, ry) = ROCKS[i];
    let d = (((pos.0 - rx) as f64).powi(2) + ((pos.1 - ry) as f64).powi(2)).sqrt();
    let eta = 2f64.powf(-d / D0);
    0.5 * (1.0 + eta)
}

fn update_rock(p: f64, o: usize, acc: f64) -> f64 {
    let likeg = if o == 1 { acc } else { 1.0 - acc };
    let likeb = if o == 1 { 1.0 - acc } else { acc };
    let den = likeg * p + likeb * (1.0 - p);
    if den > 0.0 {
        likeg * p / den
    } else {
        p
    }
}

#[derive(Clone, Copy, Debug)]
enum Act {
    Move(char),
    Sample(usize),
    Check(usize),
}

fn rs_policy(pos: (i32, i32), p: &[f64; KROCK], sampled: &[bool; KROCK], qmdp: bool) -> Act {
    for i in 0..KROCK {
        if pos == ROCKS[i] && !sampled[i] && p[i] >= 0.57 {
            return Act::Sample(i);
        }
    }
    let mut candidates: Vec<(f64, usize, i32)> = Vec::new();
    for i in 0..KROCK {
        if sampled[i] || p[i] < 0.32 {
            continue;
        }
        let dist = (pos.0 - ROCKS[i].0).abs() + (pos.1 - ROCKS[i].1).abs();
        let score = 10.0 * p[i] - 0.12 * (dist as f64);
        candidates.push((score, i, dist));
    }
    if candidates.is_empty() {
        return Act::Move('E');
    }
    // Python max() on tuples: compares score, then i, then dist.
    let &(_, i, _dist) = candidates
        .iter()
        .max_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
        })
        .unwrap();
    if !qmdp && p[i] > 0.35 && p[i] < 0.68 && sensor_acc(pos, i) > 0.60 {
        return Act::Check(i);
    }
    let (tx, ty) = ROCKS[i];
    let (x, y) = pos;
    if x < tx {
        return Act::Move('E');
    }
    if x > tx {
        return Act::Move('W');
    }
    if y < ty {
        return Act::Move('N');
    }
    if y > ty {
        return Act::Move('S');
    }
    Act::Sample(i)
}

fn rs_ep(kind: &str, rng: &mut rand::rngs::StdRng, h: usize, np_particles: usize, w: usize) -> f64 {
    let mut true_: [i32; KROCK] = std::array::from_fn(|_| rng.gen_range(0..2));
    let mut sampled = [false; KROCK];
    let mut pos: (i32, i32) = (0, 0);
    let mut p: [f64; KROCK] = [0.5; KROCK];
    let mut hist: [Vec<(usize, f64)>; KROCK] = Default::default();

    let is_particle = kind.starts_with("Particle");
    let is_window = kind.starts_with("Finite window");
    let mut particles: Vec<[i32; KROCK]> = if is_particle {
        (0..np_particles)
            .map(|_| std::array::from_fn(|_| rng.gen_range(0..2)))
            .collect()
    } else {
        Vec::new()
    };

    let mut r_total = 0.0;
    let mut d = 1.0;

    for _ in 0..h {
        let pp: [f64; KROCK] = if is_particle {
            let mut m = [0.0; KROCK];
            for i in 0..KROCK {
                let s: i32 = particles.iter().map(|pt| pt[i]).sum();
                m[i] = s as f64 / particles.len() as f64;
            }
            m
        } else if is_window {
            let mut m = [0.5; KROCK];
            for i in 0..KROCK {
                let start = hist[i].len().saturating_sub(w);
                for &(o, acc) in &hist[i][start..] {
                    m[i] = update_rock(m[i], o, acc);
                }
            }
            m
        } else {
            p
        };

        let act = rs_policy(pos, &pp, &sampled, kind == "QMDP");
        let mut r = 0.0;
        match act {
            Act::Move(dir) => {
                let (dx, dy) = moves(dir);
                let (nx, ny) = (pos.0 + dx, pos.1 + dy);
                if dir == 'E' && nx >= SIZE {
                    r = 10.0;
                    r_total += d * r;
                    break;
                }
                if nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE {
                    pos = (nx, ny);
                }
            }
            Act::Sample(i) => {
                if pos == ROCKS[i] && !sampled[i] {
                    r = if true_[i] != 0 { 10.0 } else { -10.0 };
                    true_[i] = 0;
                    sampled[i] = true;
                    p[i] = 0.0;
                    if is_particle {
                        for pt in particles.iter_mut() {
                            pt[i] = 0;
                        }
                    }
                }
            }
            Act::Check(i) => {
                let acc = sensor_acc(pos, i);
                let o: usize = if rng.gen_range(0.0..1.0) < acc {
                    true_[i] as usize
                } else {
                    1 - true_[i] as usize
                };
                hist[i].push((o, acc));
                p[i] = update_rock(p[i], o, acc);
                if is_particle {
                    let w_: Vec<f64> = particles
                        .iter()
                        .map(|pt| if pt[i] as usize == o { acc } else { 1.0 - acc })
                        .collect();
                    let sw: f64 = w_.iter().sum();
                    let wn: Vec<f64> = w_.iter().map(|&v| v / sw).collect();
                    let n = particles.len();
                    let idx: Vec<usize> = (0..n).map(|_| sample_categorical_slice(rng, &wn)).collect();
                    particles = idx.iter().map(|&k| particles[k]).collect();
                }
            }
        }
        r_total += d * r;
        d *= GAMMA;
    }
    r_total
}

fn sample_categorical_slice(rng: &mut rand::rngs::StdRng, p: &[f64]) -> usize {
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

struct Eval {
    return_mean: f64,
    return_std: f64,
    ms_episode_mean: f64,
    ms_episode_std: f64,
}

fn evaluate<F>(epfun: F, episodes_per_seed: usize) -> Eval
where
    F: Fn(&mut rand::rngs::StdRng) -> f64,
{
    let mut vals = Vec::with_capacity(30);
    let mut times = Vec::with_capacity(30);
    for seed in 0..30u64 {
        let mut rng = rng_from_seed(7000 + seed);
        let t0 = Instant::now();
        let rr: Vec<f64> = (0..episodes_per_seed).map(|_| epfun(&mut rng)).collect();
        times.push(t0.elapsed().as_secs_f64() / (episodes_per_seed as f64) * 1e3);
        vals.push(mean(&rr));
    }
    Eval {
        return_mean: mean(&vals),
        return_std: std_ddof1(&vals),
        ms_episode_mean: mean(&times),
        ms_episode_std: std_ddof1(&times),
    }
}

fn eval_to_json(e: &Eval) -> serde_json::Value {
    json!({
        "return_mean": e.return_mean,
        "return_std": e.return_std,
        "ms_episode_mean": e.ms_episode_mean,
        "ms_episode_std": e.ms_episode_std,
    })
}

fn main() {
    let (grid, v) = solve_tiger_value();

    let mut tiger_out = serde_json::Map::new();
    for kind in [
        "BSD (exact-belief family)",
        "Finite window W=2",
        "Finite window W=1",
        "QMDP",
    ] {
        let e = evaluate(|rng| tiger_ep(kind, rng, &grid, &v, 100), 100);
        tiger_out.insert(kind.to_string(), eval_to_json(&e));
    }

    let mut rs_out = serde_json::Map::new();
    for kind in [
        "BSD (exact factor family)",
        "Particle belief N=64",
        "Finite window W=2",
        "Finite window W=1",
        "QMDP-style no-check",
    ] {
        let e = evaluate(
            |rng| match kind {
                "BSD (exact factor family)" => rs_ep("BSD", rng, 60, 64, 2),
                "Particle belief N=64" => rs_ep("Particle N=64", rng, 60, 64, 2),
                "Finite window W=2" => rs_ep("Finite window W=2", rng, 60, 64, 2),
                "Finite window W=1" => rs_ep("Finite window W=1", rng, 60, 64, 1),
                _ => rs_ep("QMDP", rng, 60, 64, 2),
            },
            80,
        );
        rs_out.insert(kind.to_string(), eval_to_json(&e));
    }

    let out = json!({ "Tiger": tiger_out, "RockSample4x4": rs_out });
    write_json("discrete_results.json", &out);
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
