import numpy as np
import torch

import common as C

DT = 1 / C.FPS
T_REF = (C.H - 1) / C.FPS
FINE = 10
N_FINE = (C.H - 1) * FINE + 1


def spline_reference(W_norm, J_min, J_max):
    spline = C.QuadraticSpline(nbSeg=C.K, nbDim=C.J, device="cpu")
    Psi, dPsi, _ = spline.computePsiList1D(torch.linspace(0, 1, N_FINE))
    W = torch.tensor(W_norm, dtype=torch.float32).reshape(len(W_norm), -1)
    pos = (W @ Psi.T).reshape(len(W), N_FINE, C.J).numpy()
    vel = (W @ dPsi.T).reshape(len(W), N_FINE, C.J).numpy() / T_REF
    return C.unnormalize(pos, J_min, J_max), vel * (J_max - J_min + 1e-6) / 2


def spline_reference_from_traj(traj, J_min, J_max):
    spline = C.QuadraticSpline(nbSeg=C.K, nbDim=C.J, device="cpu")
    Yn = torch.tensor(C.normalize(traj, J_min, J_max), dtype=torch.float32)
    w, _ = spline.encode_trajectory(Yn)
    return spline_reference(w.reshape(len(traj), C.P, C.J).numpy(), J_min, J_max)


def waypoint_reference(Y):
    t_fine = np.linspace(0, C.H - 1, N_FINE)
    seg = np.minimum(np.floor(t_fine).astype(int), C.H - 2)
    a = (t_fine - seg)[None, :, None]
    pos = (1 - a) * Y[:, seg] + a * Y[:, seg + 1]
    vel = (Y[:, seg + 1] - Y[:, seg]) * C.FPS
    return pos, vel


def closest(ref_pos, x):
    d2 = np.sum((ref_pos - x[:, None]) ** 2, axis=-1)
    i = d2.argmin(axis=1)
    return i, np.sqrt(d2[np.arange(len(x)), i])


def simulate(ref_pos, controller, push, t_push=0.2, k=6.0, t_end=T_REF + 0.5):
    B = len(ref_pos)
    rows = np.arange(B)
    push_step = int(round(t_push / DT))
    x = ref_pos[:, 0].copy()
    xs, dist, phase = [], [], []
    for n in range(int(round(t_end / DT)) + 1):
        if n == push_step:
            x = x + push
        i, d = closest(ref_pos, x)
        xs.append(x.copy())
        dist.append(d)
        phase.append(i / (FINE * C.FPS))
        if controller == "tracking":
            j = np.full(B, min(n * FINE, N_FINE - 1))
        else:
            j = i
        j_next = np.minimum(j + FINE, N_FINE - 1)
        progress = ref_pos[rows, j_next] - ref_pos[rows, j]
        x = x + progress + k * DT * (ref_pos[rows, j] - x)
    return np.stack(xs, 1), np.stack(dist, 1), np.stack(phase, 1)


def push_vectors(ref_pos, ref_vel, kind, magnitude, t_push=0.2, joint=3, seed=0):
    B = len(ref_pos)
    j = int(round(t_push * C.FPS * FINE))
    tangent = ref_vel[:, j] / (np.linalg.norm(ref_vel[:, j], axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed)
    if kind == "random":
        u = rng.standard_normal((B, C.J))
    elif kind == "perpendicular":
        u = rng.standard_normal((B, C.J))
        u -= np.sum(u * tangent, axis=1, keepdims=True) * tangent
    elif kind == "forward":
        u = tangent.copy()
    elif kind == "backward":
        u = -tangent
    elif kind == "one joint":
        u = np.zeros((B, C.J))
        u[:, joint] = 1.0
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return magnitude * u


def metrics(xs, dist, phase, ref_pos, magnitude, t_push=0.2, t_check=None):
    t_check = t_push + 0.5 if t_check is None else t_check
    p = int(round(t_push / DT))
    c = int(round(t_check / DT))
    expected = phase[:, p - 1] + DT if p > 0 else 0.0
    t = np.arange(dist.shape[1]) * DT
    recovered = dist[:, p:] <= 0.1 * magnitude
    first = np.where(recovered.any(1), recovered.argmax(1), -1)
    rec_time = np.where(first >= 0, first * DT, np.nan)
    j_clock = min(int(round(t_check * C.FPS * FINE)), N_FINE - 1)
    jerk = np.diff(xs[:, p + 1:], n=3, axis=1) * C.FPS ** 3
    return {
        "recovery_time_s": rec_time,
        "phase_jump_s": phase[:, p] - expected,
        "phase_shift_s": phase[:, c] - t[c],
        "clock_error_rad": np.linalg.norm(xs[:, c] - ref_pos[:, j_clock], axis=1),
        "rms_jerk": np.sqrt(np.mean(jerk ** 2, axis=(1, 2))),
    }
