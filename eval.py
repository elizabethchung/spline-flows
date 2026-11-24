import argparse
import json
import os
import time

import numpy as np
import torch

import common as C

p = argparse.ArgumentParser()
p.add_argument("--n_samples", type=int, default=10)
p.add_argument("--batch", type=int, default=2048)
p.add_argument("--max_windows", type=int, default=None)
p.add_argument("--runs", default=os.path.join(C.PROJECT, "runs"))
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu" and args.max_windows is None:
    raise SystemExit("No GPU visible to PyTorch, refusing a full evaluation on CPU.")


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


def load_model(target):
    model = C.make_model().to(device).eval()
    ckpt = torch.load(os.path.join(args.runs, target, "ckpt_best.pt"), map_location=device)
    model.load_state_dict(ckpt["ema"])
    return model, ckpt["epoch"]


def generate(model, target, cond, generator=None):
    length = C.P if target == "spline" else C.H
    x = C.sample(model, C.make_scheduler(), cond, length, generator=generator)
    return C.to_trajectory(x, target)


def time_call(model, target, cond, batch_size, repeats=5):
    c = cond[:1].repeat(batch_size, 1)
    generate(model, target, c)
    times = []
    for _ in range(repeats):
        sync()
        t = time.time()
        generate(model, target, c)
        sync()
        times.append(time.time() - t)
    return float(np.median(times))


out, timing = {}, {}
for target in ["spline", "waypoint"]:
    data = C.load_data(target)
    cond = torch.tensor(data["cond_test"], dtype=torch.float32, device=device)
    if args.max_windows:
        cond = cond[:args.max_windows]
    W, N = len(cond), args.n_samples
    model, best_epoch = load_model(target)
    print(f"{target}: checkpoint from epoch {best_epoch}, {W} windows x {N} samples", flush=True)

    g = torch.Generator(device=device).manual_seed(0)
    cond_rep = cond.repeat_interleave(N, dim=0)
    chunks = []
    for i in range(0, len(cond_rep), args.batch):
        chunks.append(generate(model, target, cond_rep[i:i + args.batch], g).cpu())
    traj = torch.cat(chunks).numpy().reshape(W, N, C.H, C.J)
    out[target] = C.unnormalize(traj, data["J_min"], data["J_max"]).astype(np.float32)

    t1 = time_call(model, target, cond, 1)
    t256 = time_call(model, target, cond, 256)
    timing[target] = {
        "best_epoch": best_epoch,
        "ms_one_window": 1000 * t1,
        "ms_per_window_in_batch_of_256": 1000 * t256 / 256,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    print(f"  one window {1000 * t1:.1f} ms, batch of 256 {1000 * t256 / 256:.2f} ms per window", flush=True)

n = len(out["spline"])
np.savez(os.path.join(args.runs, "eval_test.npz"),
         spline=out["spline"], waypoint=out["waypoint"],
         X_test=data["X_test"][:n].astype(np.float32), Y_test=data["Y_test"][:n].astype(np.float32))
with open(os.path.join(args.runs, "eval_timing.json"), "w") as f:
    json.dump(timing, f, indent=1)
print("saved", os.path.join(args.runs, "eval_test.npz"))
