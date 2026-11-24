import os
import sys

import numpy as np
import torch
from diffusers import DDPMScheduler

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)
from conditional_unet1d import ConditionalUnet1D  # noqa: E402
from quadratic_spline import QuadraticSpline  # noqa: E402

HIST, H, J = 8, 32, 29
K = 14
P = K + 2
FPS = 30
DATA = os.path.join(PROJECT, "walk_windows_history8_horizon32_stride4.npz")
SPLINE = os.path.join(PROJECT, "spline_targets_K14.npz")
VAL_SUBJECT = "subject4"


def normalize(x, J_min, J_max):
    return 2 * (x - J_min) / (J_max - J_min + 1e-6) - 1


def unnormalize(z, J_min, J_max):
    return (z + 1) / 2 * (J_max - J_min + 1e-6) + J_min


def window_file_ids(X):
    same_file = np.all(X[1:, :4] == X[:-1, 4:], axis=(1, 2))
    return np.concatenate([[0], np.cumsum(~same_file)])


def split_train_val(d):
    names = [os.path.splitext(os.path.basename(str(f)))[0] for f in d["train_files"]]
    file_id = window_file_ids(d["X_train"])
    n_found = file_id.max() + 1
    assert n_found == len(names), f"found {n_found} runs of consecutive windows but {len(names)} training files"
    window_names = np.array(names)[file_id]
    is_val = np.char.find(window_names.astype(str), VAL_SUBJECT) >= 0
    assert is_val.any(), f"no training file contains '{VAL_SUBJECT}': {names}"
    return is_val, names, file_id


def load_data(target):
    d = np.load(DATA)
    J_min, J_max = d["J_min"], d["J_max"]
    is_val, _, _ = split_train_val(d)

    def cond(X):
        return normalize(X, J_min, J_max).reshape(len(X), -1)

    if target == "spline":
        s = np.load(SPLINE)
        tgt_all, tgt_test = s["W_train"], s["W_test"]
    else:
        tgt_all = normalize(d["Y_train"], J_min, J_max)
        tgt_test = normalize(d["Y_test"], J_min, J_max)

    return {
        "J_min": J_min, "J_max": J_max,
        "cond_train": cond(d["X_train"][~is_val]), "target_train": tgt_all[~is_val],
        "cond_val": cond(d["X_train"][is_val]), "target_val": tgt_all[is_val],
        "X_val": d["X_train"][is_val], "Y_val": d["Y_train"][is_val],
        "cond_test": cond(d["X_test"]), "target_test": tgt_test,
        "X_test": d["X_test"], "Y_test": d["Y_test"],
    }


def make_model():
    return ConditionalUnet1D(
        input_dim=J,
        global_cond_dim=HIST * J,
        diffusion_step_embed_dim=128,
        down_dims=[128, 256, 512],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    )


def make_scheduler():
    return DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        clip_sample_range=2.0,
        prediction_type="epsilon",
    )


_spline = QuadraticSpline(nbSeg=K, nbDim=J, device="cpu")
_Psi_cpu, _, _ = _spline.computePsiList1D(torch.linspace(0, 1, H))
_Psi = {}


def decode_spline(W):
    if W.device not in _Psi:
        _Psi[W.device] = _Psi_cpu.to(W.device)
    return (W.reshape(len(W), -1) @ _Psi[W.device].T).reshape(len(W), H, J)


def to_trajectory(x, target):
    return decode_spline(x) if target == "spline" else x


@torch.no_grad()
def sample(model, scheduler, cond, length, num_inference_steps=100, generator=None):
    x = torch.randn(len(cond), length, J, device=cond.device, generator=generator)
    scheduler.set_timesteps(num_inference_steps)
    for t in scheduler.timesteps:
        out = model(x, t, global_cond=cond)
        x = scheduler.step(out, t, x, generator=generator).prev_sample
    return x


def errors(pred, true):
    pos = np.sqrt(np.mean((pred - true) ** 2))
    vel = np.sqrt(np.mean((np.diff(pred, axis=1) * FPS - np.diff(true, axis=1) * FPS) ** 2))
    return float(pos), float(vel)
