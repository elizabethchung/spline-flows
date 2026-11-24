import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler

import common as C

p = argparse.ArgumentParser()
p.add_argument("--target", choices=["spline", "waypoint"], required=True)
p.add_argument("--epochs", type=int, default=500)
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--val_every", type=int, default=5)
p.add_argument("--max_steps", type=int, default=None)
p.add_argument("--out", default=os.path.join(C.PROJECT, "runs"))
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu" and args.max_steps is None:
    raise SystemExit("No GPU visible to PyTorch, refusing a full run on CPU.")
out_dir = os.path.join(args.out, args.target)
os.makedirs(out_dir, exist_ok=True)

data = C.load_data(args.target)


def to_tensor(a):
    return torch.tensor(a, dtype=torch.float32, device=device)


cond_tr, tgt_tr = to_tensor(data["cond_train"]), to_tensor(data["target_train"])
cond_va, tgt_va = to_tensor(data["cond_val"]), to_tensor(data["target_val"])
length = tgt_tr.shape[1]
print(f"target={args.target}  train {tuple(tgt_tr.shape)}  val ({C.VAL_SUBJECT}) {tuple(tgt_va.shape)}  "
      f"device={device}")

model = C.make_model().to(device)
ema = copy.deepcopy(model).eval()
for q in ema.parameters():
    q.requires_grad_(False)
noise_sched = C.make_scheduler()

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
steps_per_epoch = int(np.ceil(len(tgt_tr) / args.batch_size))
total_steps = args.max_steps or args.epochs * steps_per_epoch
lr_sched = get_scheduler(
    "cosine", optimizer=opt,
    num_warmup_steps=min(500, total_steps // 10),
    num_training_steps=total_steps,
)
n_params = sum(q.numel() for q in model.parameters())
print(f"params {n_params:,}  steps/epoch {steps_per_epoch}  total steps {total_steps}")


def diffusion_loss(net, x0, cond, generator=None):
    noise = torch.randn(x0.shape, device=device, generator=generator)
    t = torch.randint(0, noise_sched.config.num_train_timesteps, (len(x0),),
                      device=device, generator=generator)
    noisy = noise_sched.add_noise(x0, noise, t)
    pred_noise = net(noisy, t, global_cond=cond)
    return F.mse_loss(C.to_trajectory(pred_noise, args.target), C.to_trajectory(noise, args.target))


@torch.no_grad()
def val_loss(net):
    g = torch.Generator(device=device).manual_seed(0)
    total = 0.0
    for i in range(0, len(tgt_va), 512):
        x0, c = tgt_va[i:i + 512], cond_va[i:i + 512]
        total += diffusion_loss(net, x0, c, g).item() * len(x0)
    return total / len(tgt_va)


def save_checkpoint(name, epoch, step, val):
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "args": vars(args), "epoch": epoch, "step": step, "val_loss_ema": val},
               os.path.join(out_dir, name))


log, step, t_start = [], 0, time.time()
best = {"val_loss_ema": float("inf"), "epoch": None}
for epoch in range(1, args.epochs + 1):
    model.train()
    perm = torch.randperm(len(tgt_tr), device=device)
    epoch_loss = torch.zeros((), device=device)
    n_seen = 0
    for i in range(0, len(perm), args.batch_size):
        idx = perm[i:i + args.batch_size]
        loss = diffusion_loss(model, tgt_tr[idx], cond_tr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        lr_sched.step()

        decay = min(0.9999, 1 - (1 + step) ** -0.75)
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(decay).add_(pm, alpha=1 - decay)

        epoch_loss += loss.detach() * len(idx)
        n_seen += len(idx)
        step += 1
        if step >= total_steps:
            break

    done = step >= total_steps or epoch == args.epochs
    if epoch % args.val_every == 0 or done or epoch == 1:
        row = {"epoch": epoch, "step": step,
               "train_loss": epoch_loss.item() / n_seen,
               "val_loss_ema": val_loss(ema),
               "minutes": (time.time() - t_start) / 60}
        log.append(row)
        is_best = row["val_loss_ema"] < best["val_loss_ema"]
        if is_best:
            best = {"val_loss_ema": row["val_loss_ema"], "epoch": epoch}
            save_checkpoint("ckpt_best.pt", epoch, step, row["val_loss_ema"])
        print(f"epoch {epoch:4d}  step {step:6d}  train {row['train_loss']:.4f}  "
              f"val(ema) {row['val_loss_ema']:.4f}  {row['minutes']:.1f} min"
              f"{'  *best' if is_best else ''}", flush=True)
        with open(os.path.join(out_dir, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
    if done:
        save_checkpoint("ckpt_last.pt", epoch, step, log[-1]["val_loss_ema"])
        break

train_minutes = (time.time() - t_start) / 60
print(f"best validation loss {best['val_loss_ema']:.4f} at epoch {best['epoch']}")

ema.load_state_dict(torch.load(os.path.join(out_dir, "ckpt_best.pt"), map_location=device)["ema"])
n = min(256, len(tgt_va))
idx = np.random.default_rng(0).choice(len(tgt_va), size=n, replace=False)
g = torch.Generator(device=device).manual_seed(0)
if device == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
x = C.to_trajectory(C.sample(ema, noise_sched, cond_va[idx], length, generator=g), args.target)
if device == "cuda":
    torch.cuda.synchronize()
gen_seconds = time.time() - t0
pred = C.unnormalize(x.cpu().numpy(), data["J_min"], data["J_max"])
true = data["Y_val"][idx]
freeze = np.repeat(data["X_val"][idx][:, -1:, :], C.H, axis=1)
pos, vel = C.errors(pred, true)
pos_f, vel_f = C.errors(freeze, true)
print(f"quick check on {n} validation windows (best checkpoint):\n"
      f"  model : position {pos:.4f} rad, velocity {vel:.3f} rad/s\n"
      f"  freeze: position {pos_f:.4f} rad, velocity {vel_f:.3f} rad/s\n"
      f"  sampling {gen_seconds:.2f} s for the batch")

summary = {
    "target": args.target, "params": n_params, "val_subject": C.VAL_SUBJECT,
    "train_windows": len(tgt_tr), "val_windows": len(tgt_va),
    "epochs_run": epoch, "steps": step, "train_minutes": round(train_minutes, 2),
    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    "best_epoch": best["epoch"], "best_val_loss_ema": best["val_loss_ema"],
    "quick_check_val": {"windows": n, "pos_rmse_rad": pos, "vel_rmse_rad_s": vel,
                        "freeze_pos_rmse_rad": pos_f, "freeze_vel_rmse_rad_s": vel_f,
                        "batch_sampling_seconds": gen_seconds},
    "args": vars(args),
}
with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("saved to", out_dir)
