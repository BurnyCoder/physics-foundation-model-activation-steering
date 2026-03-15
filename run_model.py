"""
Run GPhyT models on real physics data and generate comparison outputs.

Usage:
    python run_model.py                                    # turbulent_radiative (default), all sizes
    python run_model.py --dataset shear_flow               # shear flow dataset
    python run_model.py --sizes M --steps 10               # just GPT_M, 10 steps
    python run_model.py --dataset shear_flow --traj 2      # shear flow, trajectory 2
"""

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from gphyt.models.transformer.model import get_model

# ── Constants ───────────────────────────────────────────────────────────────

N_STEPS_INPUT = 4
OUT_SHAPE = (256, 128)
INPUT_CHANNELS = 5  # pressure, density, temperature, vel_x, vel_y
HF_REPO = "flwi/Physics-Foundation-Model"

DATASETS = {
    "heat": {
        "title": "Heat Diffusion",
        "synthetic": True,
        "active_channels": (2,),
        "fields": [
            (2, "Temperature", "hot"),
        ],
        "default_start": 0,
    },
    "advection": {
        "title": "Advection (blob in uniform flow)",
        "synthetic": True,
        "active_channels": (2, 3, 4),
        "fields": [
            (2, "Temperature", "hot"),
            (3, "Velocity X", "coolwarm"),
            (4, "Velocity Y", "coolwarm"),
        ],
        "default_start": 0,
    },
    "vortex": {
        "title": "Decaying Vortex",
        "synthetic": True,
        "active_channels": (0, 3, 4),
        "fields": [
            (0, "Pressure", "inferno"),
            (3, "Velocity X", "coolwarm"),
            (4, "Velocity Y", "coolwarm"),
            (-1, "|Velocity|", "magma"),
        ],
        "default_start": 0,
    },
    "euler": {
        "npz": Path("data/euler_sample.npz"),
        "title": "Euler Compressible Flow",
        "active_channels": (0, 1, 3, 4),  # pressure, density, momentum_x, momentum_y
        "fields": [
            (0, "Pressure", "inferno"),
            (1, "Density", "viridis"),
            (3, "Momentum X", "coolwarm"),
            (4, "Momentum Y", "coolwarm"),
            (-1, "|Momentum|", "magma"),
        ],
        "default_start": 5,
    },
    "turbulent_radiative": {
        "hdf5": Path("data/datasets/turbulent_radiative_layer_2D/data/test/"
                     "turbulent_radiative_layer_tcool_0.32.hdf5"),
        "title": "Turbulent Radiative Layer 2D",
        "active_channels": (0, 1, 3, 4),
        "fields": [
            (0, "Pressure", "inferno"),
            (1, "Density", "viridis"),
            (3, "Velocity X", "coolwarm"),
            (4, "Velocity Y", "coolwarm"),
            (-1, "|Velocity|", "magma"),
        ],
        "default_start": 10,
        "t0_fields": ["pressure", "density"],
    },
    "shear_flow": {
        "hdf5": Path("data/datasets/shear_flow/data/test/"
                     "shear_flow_Reynolds_1e5_Schmidt_2e0.hdf5"),
        "title": "Shear Flow",
        "active_channels": (0, 3, 4),
        "fields": [
            (0, "Pressure", "inferno"),
            (3, "Velocity X", "coolwarm"),
            (4, "Velocity Y", "coolwarm"),
            (-1, "|Velocity|", "magma"),
        ],
        "default_start": 50,
        "t0_fields": ["pressure"],
    },
    "rayleigh_benard": {
        "hdf5": Path("data/datasets/rayleigh_benard/data/test/"
                     "rayleigh_benard_Rayleigh_1e9_Prandtl_5.hdf5"),
        "title": "Rayleigh-Bénard Convection",
        "active_channels": (0, 2, 3, 4),  # pressure, buoyancy→temp, vel_x, vel_y
        "fields": [
            (0, "Pressure", "inferno"),
            (2, "Temperature", "hot"),
            (3, "Velocity X", "coolwarm"),
            (4, "Velocity Y", "coolwarm"),
            (-1, "|Velocity|", "magma"),
        ],
        "default_start": 10,
        "t0_fields": ["pressure", "buoyancy"],
        "buoyancy_as_temperature": True,
    },
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_model_config(size: str) -> dict:
    return {
        "img_size": (N_STEPS_INPUT, *OUT_SHAPE),
        "transformer": {
            "input_channels": INPUT_CHANNELS,
            "model_size": f"GPT_{size}",
            "att_mode": "full",
            "dropout": 0.0,
            "pos_enc_mode": "absolute",
            "patch_size": [1, 16, 16],
            "stochastic_depth_rate": 0.0,
            "use_derivatives": True,
            "integrator": "Euler",
        },
        "tokenizer": {
            "tokenizer_mode": "linear",
            "detokenizer_mode": "linear",
            "tokenizer_overlap": 0,
            "detokenizer_overlap": 0,
        },
    }


def generate_heat_diffusion(num_steps: int, alpha: float = 0.25) -> np.ndarray:
    """Simulate 2D heat equation: dT/dt = alpha * laplacian(T).
    Gaussian hot spot diffusing on a 256x128 grid.
    Returns (num_steps, H, W, 5) with temperature in channel 2."""
    H, W = OUT_SHAPE
    total = N_STEPS_INPUT + num_steps
    data = np.zeros((total, H, W, INPUT_CHANNELS), dtype=np.float32)

    # Initial condition: Gaussian hot spot
    y, x = np.mgrid[:H, :W].astype(np.float32)
    T = np.exp(-((x - W / 2)**2 + (y - H / 2)**2) / (2 * 20**2))

    for t in range(total):
        data[t, :, :, 2] = T
        # Finite difference Laplacian with periodic BC
        lap = (np.roll(T, 1, 0) + np.roll(T, -1, 0) +
               np.roll(T, 1, 1) + np.roll(T, -1, 1) - 4 * T)
        T = T + alpha * lap

    return data


def generate_advection(num_steps: int) -> np.ndarray:
    """Advect a Gaussian blob with constant uniform velocity.
    dT/dt + u*dT/dx + v*dT/dy = 0, solved with upwind finite differences.
    Sub-steps to satisfy CFL condition."""
    H, W = OUT_SHAPE
    total = N_STEPS_INPUT + num_steps
    u, v = 0.4, 0.15
    n_sub = 10

    data = np.zeros((total, H, W, INPUT_CHANNELS), dtype=np.float32)

    y, x = np.mgrid[:H, :W].astype(np.float32)
    T = np.exp(-((x - W * 0.25)**2 + (y - H * 0.5)**2) / (2 * 12**2))

    for t in range(total):
        data[t, :, :, 2] = T
        data[t, :, :, 3] = u * n_sub
        data[t, :, :, 4] = v * n_sub
        for _ in range(n_sub):
            dTdx = T - np.roll(T, 1, axis=1)
            dTdy = T - np.roll(T, 1, axis=0)
            T = T - u * dTdx - v * dTdy

    return data


def generate_decaying_vortex(num_steps: int) -> np.ndarray:
    """Taylor-Green decaying vortex — exact Navier-Stokes solution.
    u =  sin(x)*cos(y)*exp(-2*nu*t)
    v = -cos(x)*sin(y)*exp(-2*nu*t)
    p = (cos(2x)+cos(2y))/4 * exp(-4*nu*t)
    This is textbook incompressible flow — very close to training data."""
    H, W = OUT_SHAPE
    total = N_STEPS_INPUT + num_steps
    nu = 0.01  # viscosity → controls decay rate

    data = np.zeros((total, H, W, INPUT_CHANNELS), dtype=np.float32)

    # Physical domain [0, 2pi] x [0, 2pi]
    yy = np.linspace(0, 2 * np.pi, H, endpoint=False, dtype=np.float32)
    xx = np.linspace(0, 2 * np.pi, W, endpoint=False, dtype=np.float32)
    x, y = np.meshgrid(xx, yy)  # (H, W)

    for t_idx in range(total):
        t = t_idx * 0.5  # time
        decay = np.exp(-2 * nu * t)
        u = np.sin(y) * np.cos(x) * decay
        v = -np.cos(y) * np.sin(x) * decay
        p = 0.25 * (np.cos(2 * x) + np.cos(2 * y)) * decay**2

        data[t_idx, :, :, 0] = p
        data[t_idx, :, :, 3] = u
        data[t_idx, :, :, 4] = v

    return data


def load_data(dataset_name: str, traj_idx: int, start_t: int,
              num_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load dataset, normalize, resize. Returns (input, gt)."""
    ds = DATASETS[dataset_name]

    if ds.get("synthetic"):
        generators = {
            "heat": generate_heat_diffusion,
            "advection": generate_advection,
            "vortex": generate_decaying_vortex,
        }
        data = generators[dataset_name](num_steps)
        # Normalize
        window = data[:N_STEPS_INPUT]
        for c in ds["active_channels"]:
            m, s = window[..., c].mean(), window[..., c].std() + 1e-6
            data[..., c] = (data[..., c] - m) / s
        t = torch.from_numpy(data)
        return t[:N_STEPS_INPUT].unsqueeze(0), t[N_STEPS_INPUT:]

    # Euler dataset: loaded from pre-downloaded npz (streamed from remote HDF5)
    if "npz" in ds:
        npz = np.load(ds["npz"])
        pressure, density = npz["pressure"], npz["density"]
        momentum = npz["momentum"]
        n_t, H, W = pressure.shape
        data = np.zeros((n_t, H, W, INPUT_CHANNELS), dtype=np.float32)
        data[..., 0] = pressure
        data[..., 1] = density
        data[..., 3] = momentum[..., 0]
        data[..., 4] = momentum[..., 1]
    else:
        with h5py.File(ds["hdf5"], "r") as f:
            pressure = f["t0_fields/pressure"][traj_idx]
            density = (f["t0_fields/density"][traj_idx]
                       if "density" in ds.get("t0_fields", []) else None)
            buoyancy = (f["t0_fields/buoyancy"][traj_idx]
                        if "buoyancy" in ds.get("t0_fields", []) else None)
            velocity = f["t1_fields/velocity"][traj_idx]

        n_t, H, W = pressure.shape
        data = np.zeros((n_t, H, W, INPUT_CHANNELS), dtype=np.float32)
        data[..., 0] = pressure
        if density is not None:
            data[..., 1] = density
        if buoyancy is not None:
            data[..., 2] = buoyancy  # buoyancy → temperature channel
        data[..., 3] = velocity[..., 0]
        data[..., 4] = velocity[..., 1]

    # Instance normalize from input window
    window = data[start_t : start_t + N_STEPS_INPUT]
    for c in ds["active_channels"]:
        m, s = window[..., c].mean(), window[..., c].std() + 1e-6
        data[..., c] = (data[..., c] - m) / s

    t = torch.from_numpy(data).permute(0, 3, 1, 2)
    t = F.interpolate(t, size=OUT_SHAPE, mode="bilinear", align_corners=False)
    t = t.permute(0, 2, 3, 1)

    end = start_t + N_STEPS_INPUT + num_steps
    return t[start_t : start_t + N_STEPS_INPUT].unsqueeze(0), t[start_t + N_STEPS_INPUT : end]


def load_model(size: str, device: torch.device) -> torch.nn.Module:
    path = hf_hub_download(repo_id=HF_REPO, filename=f"gphyt-{size}.pth")
    model = get_model(make_model_config(size))
    sd = torch.load(path, weights_only=False, map_location=device)
    consume_prefix_in_state_dict_if_present(sd, "module.")
    consume_prefix_in_state_dict_if_present(sd, "_orig_mod.")
    sd.pop("_metadata", None)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def rollout(model, x, num_steps, device):
    outputs = []
    cur = x.to(device).clone()
    for _ in range(num_steps):
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred = model(cur)
        outputs.append(pred.cpu().float())
        cur = torch.cat([cur[:, 1:], pred], dim=1)
    return torch.cat(outputs, dim=1)[0].numpy()


def field_data(data, ch):
    if ch == -1:
        return np.sqrt(data[..., 3]**2 + data[..., 4]**2)
    return data[..., ch]


def color_range(arrays, ch):
    vals = np.concatenate([field_data(a, ch).ravel() for a in arrays])
    return tuple(np.nanpercentile(vals, [2, 98]))


def render_frame(t, gt, rollouts, ranges, fields, sizes, title, num_steps):
    cols = ["Ground Truth"] + [f"GPT-{s}" for s in sizes]
    fig, axes = plt.subplots(len(fields), len(cols),
                             figsize=(2.8 * len(cols), 2.5 * len(fields)))
    if axes.ndim == 1:
        axes = axes[:, np.newaxis] if len(cols) == 1 else axes[np.newaxis, :]

    for row, (ch, name, cmap) in enumerate(fields):
        vmin, vmax = ranges[ch]
        axes[row, 0].imshow(field_data(gt, ch)[t].T, cmap=cmap,
                            vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        axes[row, 0].set_ylabel(name, fontsize=9, fontweight="bold")
        for ci, s in enumerate(sizes, 1):
            axes[row, ci].imshow(field_data(rollouts[s], ch)[t].T, cmap=cmap,
                                 vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    for ci, label in enumerate(cols):
        axes[0, ci].set_title(label, fontsize=10, fontweight="bold")

    fig.suptitle(f"{title} — t+{t+1}/{num_steps}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return frame


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run GPhyT on physics data")
    parser.add_argument("--dataset", default="turbulent_radiative",
                        choices=list(DATASETS.keys()),
                        help="Dataset to use (default: turbulent_radiative)")
    parser.add_argument("--sizes", nargs="+", default=["S", "M", "L", "XL"],
                        choices=["S", "M", "L", "XL"])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--start", type=int, default=None,
                        help="Start timestep (default: dataset-specific)")
    parser.add_argument("--traj", type=int, default=0,
                        help="Trajectory index (default: 0)")
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    start_t = args.start if args.start is not None else ds["default_start"]
    fields = ds["fields"]
    title = ds["title"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("output") / f"{args.dataset}_{ts}"
    run_dir.mkdir(parents=True)
    print(f"Output: {run_dir}/\n")

    # Data
    print(f"Loading {title} (traj={args.traj}, start={start_t})...")
    x, gt = load_data(args.dataset, args.traj, start_t, args.steps)
    gt_np = gt.numpy()
    print(f"  Input: t={start_t}-{start_t + N_STEPS_INPUT - 1}, "
          f"GT: t={start_t + N_STEPS_INPUT}-{start_t + N_STEPS_INPUT + args.steps - 1}\n")

    # Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollouts = {}
    for size in args.sizes:
        print(f"GPT-{size}...", end=" ", flush=True)
        model = load_model(size, device)
        n = sum(p.numel() for p in model.parameters())
        rollouts[size] = rollout(model, x, args.steps, device)
        print(f"{n:,} params, done")
        del model; torch.cuda.empty_cache()

    # Color ranges
    all_data = [gt_np] + list(rollouts.values())
    ranges = {ch: color_range(all_data, ch) for ch, _, _ in fields}

    # GIF
    print("\nGenerating GIF...")
    frames = [render_frame(t, gt_np, rollouts, ranges, fields, args.sizes, title, args.steps)
              for t in range(args.steps)]
    gif_path = run_dir / "comparison.gif"
    iio.imwrite(gif_path, frames, loop=0, duration=int(1000 / args.fps))
    print(f"  {gif_path}")

    # Static PNGs per field
    key_steps = [i for i in [0, 4, 9, 19, 29] if i < args.steps]
    cols = ["Ground Truth"] + [f"GPT-{s}" for s in args.sizes]

    for ch, name, cmap in fields:
        vmin, vmax = ranges[ch]
        fig, axes = plt.subplots(len(key_steps), len(cols),
                                 figsize=(2.5 * len(cols), 2.2 * len(key_steps)), squeeze=False)
        for ri, t in enumerate(key_steps):
            axes[ri][0].imshow(field_data(gt_np, ch)[t].T, cmap=cmap,
                               vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
            axes[ri][0].set_ylabel(f"t+{t+1}", fontsize=9, fontweight="bold")
            for ci, s in enumerate(args.sizes, 1):
                axes[ri][ci].imshow(field_data(rollouts[s], ch)[t].T, cmap=cmap,
                                    vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
            for ax in axes[ri]:
                ax.set_xticks([]); ax.set_yticks([])
        for ci, label in enumerate(cols):
            axes[0][ci].set_title(label, fontsize=9, fontweight="bold")

        safe = name.replace("|", "").replace(" ", "_").lower()
        fig.suptitle(f"{name} — All Model Sizes", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(run_dir / f"{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {run_dir}/{safe}.png")

    # Particle tracking GIF — only when velocity fields exist
    has_velocity = any(ch in (3, 4, -1) for ch, _, _ in fields)
    if has_velocity:
        print("\nGenerating particle tracking GIF...")
        H, W = gt_np.shape[1], gt_np.shape[2]
        n_particles = 200
        rng = np.random.default_rng(42)
        seed_pos = np.stack([rng.uniform(0, H - 1, n_particles),
                             rng.uniform(0, W - 1, n_particles)], axis=1)

        def advect_particles(vel_data, positions, dt=1.0):
            trajectories = [positions.copy()]
            pos = positions.copy()
            for t in range(vel_data.shape[0]):
                vx, vy = vel_data[t, :, :, 3], vel_data[t, :, :, 4]
                pi = np.clip(pos[:, 0], 0, H - 1).astype(int)
                pj = np.clip(pos[:, 1], 0, W - 1).astype(int)
                pos[:, 0] += vx[pi, pj] * dt
                pos[:, 1] += vy[pi, pj] * dt
                pos[:, 0] = np.clip(pos[:, 0], 0, H - 1)
                pos[:, 1] = np.clip(pos[:, 1], 0, W - 1)
                trajectories.append(pos.copy())
            return trajectories

        gt_traj = advect_particles(gt_np, seed_pos)
        pred_trajs = {s: advect_particles(r, seed_pos) for s, r in rollouts.items()}

        particle_frames = []
        cols_p = ["GT"] + [f"GPT-{s}" for s in args.sizes]
        all_trajs = [gt_traj] + [pred_trajs[s] for s in args.sizes]
        colors = plt.cm.tab10(np.linspace(0, 1, n_particles))
        # Background field: density if available, else pressure
        bg_ch = 1 if any(ch == 1 for ch, _, _ in fields) else 0

        for t in range(args.steps):
            fig, axes_p = plt.subplots(1, len(cols_p),
                                       figsize=(3.5 * len(cols_p), 3.5), squeeze=False)
            for ci, (label, traj) in enumerate(zip(cols_p, all_trajs)):
                ax = axes_p[0][ci]
                bg = gt_np if ci == 0 else list(rollouts.values())[ci - 1]
                ax.imshow(field_data(bg, bg_ch)[t].T,
                          cmap="gray_r", alpha=0.4, aspect="auto", origin="lower")
                trail_start = max(0, t - 4)
                for p in range(n_particles):
                    trail_x = [traj[s][p, 0] for s in range(trail_start, t + 2)]
                    trail_y = [traj[s][p, 1] for s in range(trail_start, t + 2)]
                    ax.plot(trail_x, trail_y, color=colors[p], linewidth=0.5, alpha=0.5)
                ax.scatter(traj[t + 1][:, 0], traj[t + 1][:, 1],
                           c=colors, s=4, zorder=5)
                ax.set_xlim(0, H); ax.set_ylim(0, W)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(label, fontsize=10, fontweight="bold")

            fig.suptitle(f"{title} — Particle Tracking  t+{t+1}/{args.steps}",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.canvas.draw()
            w_px, h_px = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)[..., :3].copy()
            particle_frames.append(frame)
            plt.close(fig)

        particle_gif = run_dir / "particles.gif"
        iio.imwrite(particle_gif, particle_frames, loop=0, duration=int(1000 / args.fps))
        print(f"  {particle_gif}")

    # Save tensors
    pt_path = run_dir / "results.pt"
    torch.save({
        "input": x.cpu().float(),
        "ground_truth": gt.cpu().float(),
        "rollouts": {s: torch.from_numpy(r) for s, r in rollouts.items()},
        "args": vars(args),
        "dataset": args.dataset,
    }, pt_path)
    print(f"  {pt_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
