# Spline flows: spline vs. waypoint motion diffusion on the G1

## Overview

The model aims to predict the next 32 frames (about 1 s) of G1 walking from the last 8 frames. We implemented two diffusion models that are identical except that one produces spline parameters (464 numbers) and the other produces waypoints (928 numbers). Then, we tested recovery from perturbed start poses with both time-indexed tracking and the spline flow field.

## Results

**Imitation**, on 3,791 test windows (held-out subject 2) with 10 samples each:

- Accuracy tie: 0.179 vs. 0.181 rad (mean of 10)
- Spline 2.3× smoother: jerk 478 vs. 1,077 (real motion 669)
- Same generation time: 267 ms per window
- Both beat the freeze baseline (0.235 rad)
- Only true with the loss measured after decoding (see [What didn't work](#what-didnt-work))

**Recovery** (the model's spline held fixed, start pose perturbed):

- Every setup gets back to 10% of the perturbation in 0.37 s
- Flow keeps its new phase; tracking keeps the clock
- At 1.6–3.2 rad, flow lands on the wrong part of the gait in 16–44% of cases

Figures are in `figures/`, videos in `figures/videos/`.

## How to reproduce

On MIT ORCD Engaging (GPU jobs on an L40S):

```bash
module load miniforge && conda activate spline
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install einops diffusers mujoco imageio imageio-ffmpeg matplotlib
hf download wty-yy/LAFAN1_g1 --repo-type dataset --local-dir ~/data/LAFAN1_g1
git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie
```

Then, from `~/spline_flows`:

1. `01_data_prep.ipynb`: checks, windows, split
2. `02_spline_fit.ipynb`: spline error vs. K
3. `03_model_setup.ipynb`: UNet check and spline targets
4. Train both models:
   ```bash
   mkdir -p runs
   sbatch -J spline train.sbatch spline
   sbatch -J waypoint train.sbatch waypoint
   ```
5. Evaluate: `sbatch -J eval eval.sbatch`, then `04_eval.ipynb`
6. Recovery: `05_recovery_initial.ipynb` (main experiment); `05_recovery.ipynb` (extra: mid-motion pushes)
7. Videos: `sbatch -J render render.sbatch`

## Files

| File | What it does |
|---|---|
| `common.py` | Shared data split, network, noise schedule, spline decoding, sampling and metrics, so both models use identical code |
| `train.py`, `train.sbatch` | Trains one model (`--target spline` or `waypoint`) and keeps the checkpoint with the lowest validation loss |
| `eval.py`, `eval.sbatch` | Generates 10 samples per test window for both models and times generation |
| `recovery.py` | Kinematic perturbation simulator: tracking and flow controllers, perturbations, metrics |
| `render.py`, `render.sbatch` | MuJoCo videos of the G1 |
| `conditional_unet1d.py`, `conv1d_components.py`, `positional_embedding.py` | Diffusion network, from the spline_policy repo |
| `quadratic_spline.py` | Spline fitting, from the spline_policy repo |
| `runs/` | Final logs and results. Earlier versions: `runs_v1_overfit`, `runs_v2_eps`, `runs_v3_sample` |

## Key decisions

- **Walking only, joints only:** one periodic behavior keeps the comparison focused; both models predict the 29 joint angles (no root motion).
- **Split by person before cutting windows:** test = subject 2, validation = subject 4 (used only to pick the best checkpoint), so overlapping windows never leak between sets.
- **K = 14, C1:** 16 numbers per joint, half the waypoint size; 18% relative velocity error (50% at K = 6); K + 2 must be divisible by 4 for the UNet.
- **Noise prediction, with the noise error measured after decoding:** weights each spline parameter by its effect on the motion; for waypoints, decoding does nothing, so the baseline is standard.
- **Same network and settings for both:** the repo's lowdim config (18,006,429 parameters, AdamW, 500 epochs, EMA, 100 DDPM steps); sampling clip range ±2 because spline targets reach ±1.72.
- **Flow field:** the paper's version (attraction toward the closest point plus progression along the curve), gain k = 6 per second; the waypoint baseline uses time-indexed tracking, and both rules run on both representations.

## What didn't work

1. **Validating on the test subject:** validation loss rose from about epoch 90 while training loss kept falling. Switched to a separate validation subject.
2. **Wrong PyTorch build:** the default build silently fell back to CPU on the cluster's CUDA 12.4 driver. Fixed with the cu124 build.
3. **Loss on the raw spline parameters:** the spline model zig-zagged (jerk 2009). In the C1 chain, each middle point is 2 × previous end − previous middle, so any parameter error passes to every later segment with alternating sign. A small simulation reproduced it.
4. **Predicting the clean motion instead:** all 10 samples came out identical. The loss is dominated by high noise levels, where the best guess is the average, so early stopping kept half-trained checkpoints.
5. **The fix:** noise prediction with the error measured after decoding. Spline jerk 2009 → 478, with diversity kept.

## Limitations

- Since the gait almost loops, some large perturbations land on the wrong part of the curve (16–44% at 1.6–3.2 rad).
- Kinematic only (no dynamics, balance or joint limits); joints only, no root motion.
- Small data (~10 walking files) leads to early overfitting.
- Most of the error is in prediction, not the representation (0.18 vs. 0.005 rad fit floor).

## Next steps

- Search for the closest point only near the current phase.
- Use a B-spline basis so parameter errors stay local.
- Pin the first control point to the current pose.
- Longer history and more data (more files, left/right mirroring).
- SONIC tracking in physics, to separate spline-field recovery from controller recovery (needs Isaac Lab via a container on the cluster).

## Time and compute

**Time:** about 12 hours in total.

| Day | Hours | Work |
|---|---|---|
| Wed 11/18 | 1.5 | Cluster setup, reading the paper and code |
| Thu 11/19 | 2.5 | Data preparation (step 1) |
| Sat 11/21 | 2 | Spline fit and training setup (steps 2–3) |
| Sun 11/22 | 3 | Training, evaluation, loss debugging, recovery (steps 3–5) |
| Mon 11/23 | 3 | Recovery to spec, videos, slides, README |

**Compute:** MIT ORCD Engaging.
- About 51 GPU-minutes on one NVIDIA L40S across all versions: training runs of 6–9 min each, evaluation runs of about 1 min each, plus a few minutes of video rendering.
- The final pipeline alone is about 14 GPU-minutes: waypoint model 6:23, spline model 6:52, evaluation 1:00.
- CPU Jupyter sessions (4 cores, 8 GB) for data preparation, analysis and the recovery simulations.