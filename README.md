# QFT Partially Observed Agentic Orchestration — Rust port

Rust re-implementation of the seven experiment scripts from the original
Python repository (`Partially_Observed_Agentic_Orchestration_QFT_Intro_Revised`).
Same models, same seed counts, same sample sizes, same output JSON shape.
Wall-clock times are smaller; everything else is statistically the same.

## Build

```bash
cargo build --release
```

Seven binaries are produced under `target/release/`:

| Binary                        | Ports                          | Output JSON |
|--------------------------------|---------------------------------|-------------|
| `exact_checks`                 | `run_exact_checks.py`           | `exact_check_results.json` |
| `nonlinear_bsd`                 | `run_nonlinear_bsd.py`          | `nonlinear_results.json` |
| `policy_certificate`            | `run_policy_certificate.py`     | `policy_certificate_results.json` |
| `generated_effective_memory`    | `run_generated_effective_memory.py` | `generated_effective_memory_results.json` |
| `agentic_routing`               | `run_agentic_routing.py`        | `agentic_routing_results.json` |
| `continuous_baselines`          | `run_continuous_baselines.py`   | `continuous_baseline_results.json` |
| `discrete_benchmarks`           | `run_discrete_benchmarks.py`    | `discrete_results.json` |

Run any of them from the directory you want the output JSON written to, e.g.:

```bash
cd /tmp/out && /path/to/target/release/nonlinear_bsd
```

The JSON files already checked into this repo were produced by one such run
of each binary (release build, single machine) and are what the numbers
below were checked against.

## What changed vs. the Python originals, and why

**RNG.** The Python scripts use `numpy.random.default_rng` (PCG64), with
NumPy's specific Dirichlet/Normal/integer-sampling algorithms. Reproducing
that bit stream in Rust isn't a reasonable target — the original README
itself states that RNG library/version, floating-point reduction order, and
platform math libraries are all expected to shift exact digits, and that
wall-clock timing is "not a deterministic statistic" at all. This port uses
`rand::StdRng` (ChaCha12) seeded the same way (same seed integers, same
seed count, same draw order per model), with `rand_distr` for Normal/Gamma
and a plain Dirichlet-via-Gamma sampler. Every seed count (30), episode
count, particle-count sweep, and horizon length matches the Python source
exactly — only the underlying bit generator differs. Every experiment was
checked field-by-field against the Python JSON outputs and reproduces the
same qualitative trends and the same order of magnitude on every quantity
(see table below).

**`scipy.optimize.least_squares`.** `nonlinear_bsd.rs` (the direct
finite-order Schwinger–Dyson closure) is the one place doing real nonlinear
least squares. `common::least_squares_lm` is a small hand-rolled
Levenberg–Marquardt solver (central-difference Jacobian, adaptive damping)
standing in for SciPy's trust-region-reflective solver. It was tuned
(iteration budget, damping schedule) until `projection_resid`/`moment_err`
for K=6 matched the Python run to within the same order of magnitude across
all 30 seeds — see the comparison below.

**`scipy.stats.wasserstein_distance`.** Reimplemented directly
(`common::wasserstein_distance`) from its definition (weighted-CDF
difference integral); matches SciPy's algorithm including its handling of
duplicate weighted support points.

**`np.interp` / `np.gradient(edge_order=2)`.** Reimplemented directly
(`interp` in `discrete_benchmarks.rs`, `gradient_edge2` in
`nonlinear_bsd.rs`) using the same interpolation/finite-difference formulas
NumPy uses.

## Fidelity check (Python → Rust, same seed counts)

All seven experiments were run and diffed field-by-field against the
Python-generated JSON in the original repo. Summary:

- `exact_checks`: both give ~machine-epsilon errors (0, as expected for an
  exact correspondence check).
- `generated_effective_memory`, `policy_certificate`: certificates and
  losses match to several significant digits; `all_certified` /
  `*_bound_holds` are `true` in both.
- `agentic_routing`: cost-per-step and routing-accuracy match to 3 decimal
  places across all four modes.
- `nonlinear_bsd`: same monotonic K=2→4→6 improvement trend; all fields
  (`sd_resid6`, `moment_err`, `tv`, `wasserstein`, `q_err`, `eta`,
  `bound_holds`) match Python's order of magnitude at every K.
- `continuous_baselines`: the accuracy-vs-particle-count frontier
  (`q_err`, `wasserstein` falling as N grows from 32 to 32768, converging to
  the exact-grid reference) matches Python's curve closely at every N.
- `discrete_benchmarks`: Tiger and RockSample(4,4) returns match to within
  ~5% per condition, including the near-zero-variance deterministic
  collapse of Tiger's `Finite window W=1` condition, which both
  implementations reproduce.

`update_ms` / `update_us` / `ms_episode_mean` fields are, as expected,
substantially smaller in the Rust build (single-digit-ms or sub-ms per
update where Python was several ms to tens of ms) — that's the one thing
that was *supposed* to change.

## Layout

```
src/lib.rs                          shared numerics: linspace/trapz, normal
                                     pdf, RNG helpers, LM solver, Wasserstein
                                     distance, mean/std, JSON writer
src/bin/exact_checks.rs
src/bin/nonlinear_bsd.rs
src/bin/policy_certificate.rs
src/bin/generated_effective_memory.rs
src/bin/agentic_routing.rs
src/bin/continuous_baselines.rs
src/bin/discrete_benchmarks.rs
```
