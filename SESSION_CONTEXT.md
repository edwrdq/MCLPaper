# Session Context (Non-Verbatim Chat Summary)

This file is a structured summary of the session context, code changes, experiments, and conclusions.

Note: This is not an exact transcript export of the chat UI. It is a comprehensive working record of what was built and decided.

## Project State Summary

The project was migrated from an initial Octave MCL prototype to a NumPy-based 2D Monte Carlo Localization implementation with:

- Raycast (lidar-like) sensor model in a square/rectangular room
- RMSE tracking against ground truth
- Benchmark and live visualization modes
- Stochastic control dynamics (velocity + acceleration + jerk)
- Safety-margin box constraints and wall-aware motion shaping
- Adaptive hooks:
  - adaptive motion noise (`sigma_t`)
  - adaptive particle count (ESS-based)
- Research benchmarking script for multi-condition experiments

## Major Evolution During Session

### 1. Octave Baseline (initial)

- Built modular Octave MCL:
  - motion model
  - sensor model
  - systematic resampling
  - RMSE tracking
- Added plotting and params
- Later removed `.m` files after migrating to Python

### 2. Python + NumPy Rewrite (core implementation)

- Added `uv` project setup (`pyproject.toml`, `uv.lock`)
- Implemented vectorized MCL in Python:
  - `main.py`
  - `motion_model.py`
  - `resample_particles.py`
  - `compute_rmse.py`
  - `state_utils.py`
- Added `rich` terminal summary output
- Added plotting and CLI flags

### 3. Runtime Modes + Dynamics

- Added `--mode visualize` and `--mode benchmark`
- Added simulated control dynamics:
  - maintain `v` and `w`
  - bounded acceleration random walk
  - jerk-driven acceleration updates
- Added control/accel/jerk logging

### 4. Raycast Sensor Model (replaced landmark model)

- Added `raycast_sensor.py`
- Switched MCL confirmation/measurement update to wall-distance raycasting
- Removed landmark sensor path (`sensor_model.py` deleted)
- System now assumes square/rectangular room boundaries as the map

### 5. Visualizer Improvements

- Live display now shows:
  - particle cloud
  - true pose + heading arrow
  - estimated pose + heading arrow
  - room walls
  - ray beams (true + estimated)
  - RMSE subplot
- Added speed controls for visualization:
  - sim-follow timing
  - time scaling
  - optional no-pause
  - optional no-ray drawing (visual perf)

### 6. Stability / Realism Fixes

- Added safety-margin inset box to keep truth + particles away from walls
- Added wall-aware control shaping to reduce unrealistic wall crashes/spin
- Added boundary clipping + heading reflection for particles and truth
- Added dashed safety-margin box in visualizer

### 7. Particle Init + Performance Tuning

- Particle init modes:
  - `local_radius` (default, 2 m radius)
  - `global`
  - `local` (Gaussian legacy/debug)
- Added `--weight-beam-stride` to speed particle likelihood updates by using every Nth beam

### 8. Adaptive Hooks (for research benchmarking)

- Adaptive noise (optional):
  - `sigma_t = sigma_0 * (1 + alpha*|a| + beta*|j|)`
- Adaptive particle count (optional):
  - ESS-based growth/shrink around base particle count
- Added logging of:
  - `sigma_t_history`
  - `ess_history`
  - `particle_count_history`
  - `accel_history`, `jerk_history`
  - `accel_mag_history`, `jerk_mag_history`

### 9. Adaptive Noise Smoothing / Damping

- Added optional EMA-style smoothing for `sigma_t`:
  - `sigma_t = gamma * sigma_prev + (1 - gamma) * sigma_raw`
- Disabled by default (baseline-compatible)

### 10. Research Benchmark Suite

- Added `research_aggressive_benchmark.py`
- Runs 4 conditions across many trials:
  - `baseline`
  - `adaptive_noise`
  - `adaptive_particles`
  - `adaptive_both`
- Default requested experiment configuration:
  - `20x20 m` room
  - `2000` particles
  - `10000` steps
  - `aggression = high`
- Added process-based parallel trials via `--workers`

## Key Bugs / Pitfalls Found and Fixed

### A. Huge RMSE in large research runs (fixed)

Two causes were identified:

1. `local_radius` init was accidentally sampling headings uniformly in `[-pi, pi]`
- This defeated the local prior and caused poor tracking
- Fixed to sample heading around `x0_true[2]` using `init_theta_std`

2. Research benchmark default start pose was exactly at square-room center
- In a square wall-only map, the center is a symmetry trap for raycast-only localization
- Changed default start to off-center in the research script

Result: RMSE returned to sane values in long aggressive runs.

## Current High-Level File Map

- `main.py`
  - core MCL simulation
  - visualize / benchmark modes
  - raycast-only sensor usage
  - adaptive hooks
  - CLI
- `raycast_sensor.py`
  - vectorized raycasting + ray-based likelihood
- `motion_model.py`
  - vectorized particle motion update
- `dynamics.py`
  - jerk/acceleration/velocity process
- `resample_particles.py`
  - systematic resampling (supports output particle resizing)
- `state_utils.py`
  - truth propagation + state estimation
- `compute_rmse.py`
  - RMSE utility
- `research_aggressive_benchmark.py`
  - multi-condition research runs (parallelizable)

## Current Important Defaults (Behavior)

- Sensor model: raycast-only (square room walls)
- Particle init: `local_radius` (2 m around start)
- Safety margin enabled (inset box)
- Visualizer can show beams (expensive; can disable)
- Aggression presets available (`low|medium|high`)

## Performance / Optimization Notes (Current)

### Already implemented

- Vectorized motion model
- Vectorized ray likelihoods
- Trial-level parallelization in research suite (`--workers`)
- Beam subsampling for particle weighting (`--weight-beam-stride`)
- Optional adaptive particles (ESS-based)
- Optional visual ray disabling (`--no-viz-rays`)

### Main compute bottleneck

- Particle measurement update (raycast likelihood over particles x beams)

### Best speed knobs (without core rewrite)

1. `--workers` (research script)
2. `--weight-beam-stride 2` or `3`
3. Fewer `--ray-beams`
4. Lower `--max-particles` (if adaptive particles enabled)

### Possible future optimization

- Optional Numba JIT for raycasting / likelihood evaluation (not implemented in this session)

## Adaptive Noise Tuning Guidance (Session Conclusions)

- Untuned adaptive noise performed poorly at first
- Adding damping/smoothing improved it significantly
- Adaptive noise remains more tuning-sensitive than adaptive particle count
- Adaptive particle count is generally easier to tune safely

Practical tuning strategy discussed:

- Tune adaptive noise alone first (disable adaptive particles)
- Use conservative gains:
  - `alpha` ~ `0.02` to `0.10`
  - `beta` ~ `0.005` to `0.05`
- Add damping (`gamma`) around `0.7` to `0.9`
- Then test combined mode

## User-Provided Benchmark Results (Latest Shared)

```csv
condition,trials,mean_of_mean_rmse,std_of_mean_rmse,mean_final_rmse,avg_runtime_s,total_runtime_s
baseline,20.0,0.013868089838279824,0.0009337479358187164,0.014605673081520462,48.16619469870002,963.3238939740004
adaptive_noise,20.0,0.014113039737802233,0.0011230232978372012,0.015770279592475102,42.60227697520013,852.0455395040026
adaptive_particles,20.0,0.013907423128125054,0.0009477780237348815,0.01681231862286114,41.98394259964989,839.6788519929978
adaptive_both,20.0,0.014652995124468039,0.0010110357864386453,0.014637484092595535,42.062911539749805,841.258230794996
```

Interpretation discussed:

- Baseline still best on mean RMSE
- Adaptive methods are now close and much faster
- `adaptive_both` gets near-baseline final RMSE with reduced runtime
- This supports a strong paper narrative about tuning sensitivity + runtime/accuracy tradeoffs

## Suggested Paper-Level Summary (Discussed)

- Untuned adaptive noise can underperform a fixed-noise baseline.
- Properly tuned adaptive noise (with damping/smoothing) can become competitive.
- Adaptive particle count tends to be easier to tune and offers useful runtime savings.
- Combined adaptive methods can achieve near-baseline terminal accuracy with improved runtime.

## Useful Commands (Current)

### Main visualize mode (fast)

```bash
uv run python main.py --mode visualize \
  --no-viz-follow-sim --viz-pause 0 \
  --viz-stride 10 \
  --no-viz-rays
```

### Main benchmark mode (adaptive examples)

```bash
uv run python main.py --mode benchmark \
  --adaptive-noise --adaptive-noise-alpha 0.05 --adaptive-noise-beta 0.01 \
  --adaptive-noise-smoothing --adaptive-noise-damping 0.8
```

```bash
uv run python main.py --mode benchmark \
  --adaptive-particles --ess-threshold 0.4 --base-particles 300 --max-particles 700
```

### Research suite (full-scale defaults)

```bash
uv run python research_aggressive_benchmark.py --workers 8
```

### Research suite (speed/compute tuning)

```bash
uv run python research_aggressive_benchmark.py \
  --workers 8 \
  --weight-beam-stride 2
```

## Git / Commit Notes (Discussed)

Suggested Conventional Commit message for the session:

```text
feat(mcl): add raycast MCL modes, adaptive hooks, and research benchmark suite
```

## Tracking / Ignore Notes

`.gitignore` was updated to ignore generated outputs (including benchmark directories).

Temporary smoke-test directories may still exist locally if not yet ignored in all variants:

- `tmp_research_smoke/`
- `tmp_research_smoke_parallel/`
- `tmp_research_postfix/`

If desired, add all `tmp_research_*` directories to `.gitignore`.

## Final Context Snapshot (What Is “True” Now)

- The codebase is raycast-only for localization measurement updates.
- Local-radius particle init is now correctly local in both position and heading.
- Research suite supports parallel trial execution.
- Adaptive noise has optional damping/smoothing and is suitable for tuning studies.
- Adaptive particle count is available and benchmarked.
- Reported recent results indicate real improvement and credible paper-ready tradeoffs.
