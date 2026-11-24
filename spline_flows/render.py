import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import common as C
import recovery as R

p = argparse.ArgumentParser()
p.add_argument("--scene", default=os.path.expanduser("~/mujoco_menagerie/unitree_g1/scene.xml"))
p.add_argument("--out", default=os.path.join(C.PROJECT, "figures", "videos"))
p.add_argument("--fps", type=int, default=10)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

model = mujoco.MjModel.from_xml_path(args.scene)
data, ghost_data = mujoco.MjData(model), mujoco.MjData(model)
H_PX, W_PX, BAR = 520, 440, 70
renderer = mujoco.Renderer(model, H_PX - BAR, W_PX)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.0, 0.0, 0.75]
cam.distance, cam.azimuth, cam.elevation = 2.6, 120, -10
opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
ROOT = model.key_qpos[0][:7].copy() if model.nkey else np.array([0, 0, 0.79, 1, 0, 0, 0])
GHOST_RGBA = [1.0, 0.5, 0.15, 0.25]
try:
    BIG, SMALL = ImageFont.load_default(size=22), ImageFont.load_default(size=17)
except TypeError:
    BIG = SMALL = ImageFont.load_default()


def set_pose(d, joints):
    d.qpos[:7] = ROOT
    d.qpos[7:] = joints
    mujoco.mj_forward(model, d)


def render_panel(joints, title, note, ghost=None, alert=False):
    set_pose(data, joints)
    renderer.update_scene(data, camera=cam)
    if ghost is not None:
        set_pose(ghost_data, ghost)
        n0 = renderer.scene.ngeom
        mujoco.mjv_addGeoms(model, ghost_data, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, renderer.scene)
        for i in range(n0, renderer.scene.ngeom):
            renderer.scene.geoms[i].rgba[:] = GHOST_RGBA
    img = Image.new("RGB", (W_PX, H_PX), (255, 255, 255))
    img.paste(Image.fromarray(renderer.render()), (0, BAR))
    draw = ImageDraw.Draw(img)
    draw.text((12, 8), title, fill=(0, 0, 0), font=BIG)
    draw.text((12, 40), note, fill=(200, 30, 30) if alert else (90, 90, 90), font=SMALL)
    return np.asarray(img)


def write(name, panels, notes, alerts, footer):
    frames = []
    for t in range(len(notes)):
        row = np.concatenate([render_panel(traj[t], title, notes[t], None if g is None else g[t], alerts[t])
                              for title, traj, g in panels], axis=1)
        bar = Image.new("RGB", (row.shape[1], 36), (255, 255, 255))
        ImageDraw.Draw(bar).text((12, 8), footer, fill=(90, 90, 90), font=SMALL)
        frames.append(np.concatenate([row, np.asarray(bar)], axis=0))
    path = os.path.join(args.out, name)
    imageio.mimwrite(path, frames, fps=args.fps, quality=8, macro_block_size=4)
    print("saved", path)


e = np.load(os.path.join(C.PROJECT, "runs", "eval_test.npz"))
X, Y = e["X_test"], e["Y_test"]
models = {"Spline": e["spline"], "Waypoint": e["waypoint"]}
score = sum(np.mean((M.mean(1) - Y) ** 2, axis=(1, 2)) for M in models.values()) / 2
speed = np.sqrt(np.mean((np.diff(Y, axis=1) * C.FPS) ** 2, axis=(1, 2)))
walking = np.where(speed >= np.median(speed))[0]
picks = {"success": walking[np.argmin(score[walking])], "failure": int(np.argmax(score))}
for label, w in picks.items():
    real = np.concatenate([X[w], Y[w]])
    panels = [("Real motion", real, None)]
    for name, M in models.items():
        panels.append((f"{name} model, sample 1 of 10", np.concatenate([X[w], M[w, 0]]), real))
    n_hist = len(X[w])
    notes = [f"history (model input)   t = {(t - n_hist + 1) / C.FPS:+.2f} s" if t < n_hist
             else f"prediction   t = {(t - n_hist + 1) / C.FPS:+.2f} s" for t in range(len(real))]
    write(f"demo_vs_generated_{label}_window{w}.mp4", panels, notes, [False] * len(real),
          "Orange ghost = the real motion. Same 8 history frames in every panel, then each model predicts the next 32. "
          "Pelvis held still; 1/3 speed.")

d = np.load(C.DATA)
idx = np.random.default_rng(0).choice(len(X), size=min(300, len(X)), replace=False)
pos, vel = R.spline_reference_from_traj(models["Spline"][idx, 0], d["J_min"], d["J_max"])


def recovery_video(name, w, kind, m):
    push = R.push_vectors(pos, vel, kind, m, t_push=0.0)[w:w + 1]
    xs = {ctrl: R.simulate(pos[w:w + 1], ctrl, push, t_push=0.0)[0][0] for ctrl in ["tracking", "flow"]}
    n = xs["flow"].shape[0]
    ref = pos[w, np.minimum(np.arange(n) * R.FINE, R.N_FINE - 1)]
    notes = [f"t = {i * R.DT:.2f} s" + ("   PERTURBED START" if i < 4 else "") for i in range(n)]
    alerts = [i < 4 for i in range(n)]
    panels = [("Generated spline (no push)", ref, None),
              ("Time-indexed tracking", xs["tracking"], ref),
              ("Distance-field flow", xs["flow"], ref)]
    write(f"{name}_window{idx[w]}.mp4", panels, notes, alerts,
          f"Spline from the model, held fixed. Start pose perturbed by {m} rad ({kind}). "
          "Orange ghost = the spline at the same moment. Pelvis held still; 1/3 speed.")


recovery_video("recovery_initial_success_random_0.8rad", 0, "random", 0.8)
push = R.push_vectors(pos, vel, "random", 1.6, t_push=0.0)
_, _, phase = R.simulate(pos, "flow", push, t_push=0.0)
jump = np.abs(phase[:, 0])
recovery_video(f"recovery_initial_failure_random_1.6rad_start{jump.max():.2f}s", int(jump.argmax()), "random", 1.6)
