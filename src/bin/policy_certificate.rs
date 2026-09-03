// Port of run_policy_certificate.py
//
// Enumerates the finite-horizon (H=3) observation-history-conditioned
// policy class over a small 2-state POMDP and measures the full chain:
// discarded effective interaction -> d_infty -> TV -> value error ->
// policy loss, against the closed-form certificates.

use common::*;
use serde_json::json;

const H: usize = 3;
const NSEED: u64 = 30;
const GVALS: [f64; 4] = [0.15, 0.35, 0.60, 0.90];

fn npol() -> usize {
    1usize << ((1usize << H) - 1)
}

struct Model {
    p0: [f64; 2],
    t: [[[f64; 2]; 2]; 2], // T[a][z] -> [p(z'=0), p(z'=1)]
    o: [[f64; 2]; 2],      // O[z][x]
    c: [[f64; 2]; 2],      // c[z][a]
}

fn model(seed: u64) -> Model {
    let mut rng = rng_from_seed(seed);
    let mut p0 = [
        0.56 + sample_normal(&mut rng, 0.0, 0.012),
        0.44 + sample_normal(&mut rng, 0.0, 0.012),
    ];
    p0[0] = p0[0].max(0.05);
    p0[1] = p0[1].max(0.05);
    let s: f64 = p0[0] + p0[1];
    p0[0] /= s;
    p0[1] /= s;

    let base = [
        [[0.90, 0.10], [0.66, 0.34]],
        [[0.34, 0.66], [0.10, 0.90]],
    ];
    let mut t = [[[0.0f64; 2]; 2]; 2];
    for a in 0..2 {
        for z in 0..2 {
            let mut row = [
                base[a][z][0] + sample_normal(&mut rng, 0.0, 0.012),
                base[a][z][1] + sample_normal(&mut rng, 0.0, 0.012),
            ];
            row[0] = row[0].max(0.02);
            row[1] = row[1].max(0.02);
            let s: f64 = row[0] + row[1];
            t[a][z] = [row[0] / s, row[1] / s];
        }
    }

    let mut acc = 0.81 + sample_normal(&mut rng, 0.0, 0.015);
    acc = acc.max(0.72).min(0.90);
    let o = [[acc, 1.0 - acc], [1.0 - acc, acc]];

    // c = [[0,1],[1,0]] + 0.08*[[0,1],[0,1]]
    let c = [[0.0, 1.0 + 0.08], [1.0, 0.0 + 0.08]];

    Model { p0, t, o, c }
}

fn policy_action(code: usize, t: usize, hbits: usize) -> usize {
    let idx = (1usize << t) - 1 + hbits;
    (code >> idx) & 1
}

fn eval_policy(code: usize, m: &Model, g: f64) -> f64 {
    // branches: (hbits, M[2][2], C[2][2])
    let mut branches: Vec<(usize, [[f64; 2]; 2], [[f64; 2]; 2])> = Vec::new();
    let mut m0 = [[0.0f64; 2]; 2];
    m0[0][0] = m.p0[0];
    m0[1][1] = m.p0[1];
    branches.push((0, m0, [[0.0; 2]; 2]));

    for t in 0..H {
        let mut nxt = Vec::new();
        for (hbits, mm, cc) in branches.into_iter() {
            let a = policy_action(code, t, hbits);
            // C0[z0][zt] = C[z0][zt] + M[z0][zt]*c[zt][a]
            let mut c0 = [[0.0f64; 2]; 2];
            for z0 in 0..2 {
                for zt in 0..2 {
                    c0[z0][zt] = cc[z0][zt] + mm[z0][zt] * m.c[zt][a];
                }
            }
            // Mp = M @ T[a]  (Mp[z0][zt1] = sum_zt M[z0][zt]*T[a][zt][zt1])
            let mut mp = [[0.0f64; 2]; 2];
            let mut cp = [[0.0f64; 2]; 2];
            for z0 in 0..2 {
                for zt1 in 0..2 {
                    let mut sm = 0.0;
                    let mut sc = 0.0;
                    for zt in 0..2 {
                        sm += mm[z0][zt] * m.t[a][zt][zt1];
                        sc += c0[z0][zt] * m.t[a][zt][zt1];
                    }
                    mp[z0][zt1] = sm;
                    cp[z0][zt1] = sc;
                }
            }
            if t == H - 1 {
                nxt.push((hbits, mp, cp));
            } else {
                for x in 0..2 {
                    let mut mpw = [[0.0f64; 2]; 2];
                    let mut cpw = [[0.0f64; 2]; 2];
                    for z0 in 0..2 {
                        for zt1 in 0..2 {
                            let w = m.o[zt1][x];
                            mpw[z0][zt1] = mp[z0][zt1] * w;
                            cpw[z0][zt1] = cp[z0][zt1] * w;
                        }
                    }
                    nxt.push(((hbits << 1) | x, mpw, cpw));
                }
            }
        }
        branches = nxt;
    }

    // W[z0][zH] = exp(-g*OP[z0][zH]); OP = V outer V, V=[-1,1] => OP[i][j] = V[i]*V[j]
    let v = [-1.0f64, 1.0];
    let mut w = [[0.0f64; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            w[i][j] = (-g * v[i] * v[j]).exp();
        }
    }

    let mut z = 0.0;
    let mut num = 0.0;
    for (_, mm, cc) in branches.iter() {
        for i in 0..2 {
            for j in 0..2 {
                z += mm[i][j] * w[i][j];
                num += cc[i][j] * w[i][j];
            }
        }
    }
    num / z
}

fn main() {
    let np = npol();
    let mut allrows: Vec<std::collections::HashMap<String, [f64; 8]>> = Vec::new();
    // fields per g: actual_loss, max_value_error, tv_certificate, value_certificate,
    // policy_certificate, value_bound_holds, policy_bound_holds  (7 fields)

    let start = std::time::Instant::now();

    for i in 0..NSEED {
        let m = model(9100 + i);
        let base: Vec<f64> = (0..np).map(|code| eval_policy(code, &m, 0.0)).collect();
        let kcode = base
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let bbound = (H as f64) * m.c.iter().flatten().cloned().fold(f64::MIN, f64::max);

        let mut rows = std::collections::HashMap::new();
        for &g in GVALS.iter() {
            let exact: Vec<f64> = (0..np).map(|code| eval_policy(code, &m, g)).collect();
            let ecode = exact
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            let actual = exact[kcode] - exact[ecode];
            let max_value = (0..np)
                .map(|i| (exact[i] - base[i]).abs())
                .fold(0.0f64, f64::max);
            let tv_cert = (g.abs() / 2.0).tanh();
            let value_cert = 2.0 * bbound * tv_cert;
            let policy_cert = 4.0 * bbound * tv_cert;
            let value_holds = if max_value <= value_cert + 1e-12 {
                1.0
            } else {
                0.0
            };
            let policy_holds = if actual <= policy_cert + 1e-12 {
                1.0
            } else {
                0.0
            };
            rows.insert(
                format!("{}", g),
                [
                    actual,
                    max_value,
                    tv_cert,
                    value_cert,
                    policy_cert,
                    value_holds,
                    policy_holds,
                    0.0,
                ],
            );
        }
        allrows.push(rows);
    }

    let keys = [
        "actual_loss",
        "max_value_error",
        "tv_certificate",
        "value_certificate",
        "policy_certificate",
        "value_bound_holds",
        "policy_bound_holds",
    ];

    let mut out = serde_json::Map::new();
    for &g in GVALS.iter() {
        let gkey = format!("{}", g);
        let mut gobj = serde_json::Map::new();
        for (fi, &field) in keys.iter().enumerate() {
            let vals: Vec<f64> = allrows.iter().map(|r| r[&gkey][fi]).collect();
            gobj.insert(field.to_string(), serde_json::to_value(mean_std(&vals)).unwrap());
        }
        out.insert(gkey, serde_json::Value::Object(gobj));
    }
    out.insert(
        "metadata".to_string(),
        json!({
            "H": H,
            "n_seeds": NSEED,
            "n_policies": np,
            "operator_osc_norm": 1.0,
            "elapsed_s": start.elapsed().as_secs_f64(),
        }),
    );

    let out = serde_json::Value::Object(out);
    write_json("policy_certificate_results.json", &out);
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
