"""CPU example of safety-social guidance on a controlled crossing scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

from sage.data.mid_adapter import build_scene_batch_from_tensors
from sage.guidance.safety_social import SafetySocialGuidance, SafetySocialGuidanceConfig


def run(output: Path, iterations: int = 80, seed: int = 7) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    dt, horizon, history = 0.2, 24, 8
    robot = torch.tensor([-2.8, -0.08])
    entities = torch.tensor([[0.0, 0.95], [0.8, -0.9]])
    entity_velocity = torch.tensor([[0.0, -0.35], [0.0, 0.0]])
    past = torch.arange(-history + 1, 1) * dt
    robot_history = robot[None] + past[:, None] * torch.tensor([1.15, 0.0])
    entity_history = entities[:, None] + past[None, :, None] * entity_velocity[:, None]
    batch = build_scene_batch_from_tensors(
        robot_history=robot_history[None],
        robot_future=torch.tensor([1.15, 0.0]).expand(1, horizon, 2).clone(),
        neighbor_history=entity_history[None],
        neighbor_future=entity_velocity[None, :, None].expand(1, 2, horizon, 2).clone(),
        neighbor_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    batch.dt = dt
    batch.goals = torch.zeros(1, 3, 2)
    batch.goals[0, 0] = torch.tensor([2.8, -0.08])
    batch.goal_mask = batch.robot_mask.clone()
    guide = SafetySocialGuidance(SafetySocialGuidanceConfig(
        dt=dt, w_collision=3.0, w_social=0.1, w_kinematic=0.02,
        w_goal=10.0, max_lateral_speed=1.2, max_angular_speed=4.0,
    ))
    velocity = batch.future.clone()
    snapshots, energies = [velocity.clone()], [float(guide.energy(velocity, batch))]
    for step in range(iterations):
        velocity = guide.project_robot_velocities(
            velocity - 0.08 * guide.robot_only_gradient(velocity, batch), batch
        ).detach()
        if (step + 1) % 4 == 0 or step + 1 == iterations:
            snapshots.append(velocity.clone())
            energies.append(float(guide.energy(velocity, batch)))
    assert torch.equal(velocity[:, 1:], batch.future[:, 1:])
    trajectories = [(batch.current_pos[:, :, None] + torch.cumsum(v * dt, dim=2))[0].numpy() for v in snapshots]
    diagnostics = guide.diagnostics(velocity, batch)
    summary = {
        "example": "controlled-crossing-guidance", "seed": seed,
        "scope": "Training-free trajectory refinement on a synthetic scene.",
        "iterations": iterations, "energy_before": energies[0], "energy_after": energies[-1],
        "energies": energies, "robot_trajectory": trajectories[-1][0].tolist(),
        "entity_predictions_fixed": True,
        "diagnostics": {key: value for key, value in diagnostics.items() if np.isfinite(value)},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, energy_ax) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.65, 1]})
    fig.patch.set_facecolor("#f8fafc")
    ax.plot(trajectories[0][0, :, 0], trajectories[0][0, :, 1], "--", color="#94a3b8", label="Initial robot plan")
    ax.plot(trajectories[0][1, :, 0], trajectories[0][1, :, 1], color="#3b82f6", lw=2, label="Pedestrian prediction")
    ax.scatter(*entities[1].numpy(), s=180, marker="s", color="#64748b", label="Static entity")
    ax.scatter(*robot.numpy(), s=70, color="#ea580c", zorder=4)
    ax.scatter(2.8, -0.08, marker="*", s=150, color="#16a34a", label="Robot goal")
    line, = ax.plot([], [], color="#ea580c", lw=3, label="Refined robot plan")
    all_points = np.concatenate([item.reshape(-1, 2) for item in trajectories])
    low, high = all_points.min(axis=0) - 0.5, all_points.max(axis=0) + 0.5
    ax.set(xlim=(min(-3.2, low[0]), max(3.3, high[0])), ylim=(min(-1.5, low[1]), max(1.5, high[1])), xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", fontsize=8)
    energy_line, = energy_ax.plot([], [], color="#ea580c", lw=2)
    energy_ax.set(xlim=(0, max(1, len(energies) - 1)), ylim=(min(energies) * 0.9, max(energies) * 1.05), xlabel="Refinement snapshot", ylabel="Safety-social energy")
    energy_ax.grid(alpha=0.15)
    title = fig.suptitle("SAGE | safety-social trajectory refinement", fontweight="bold")
    fig.text(0.5, 0.015, "Controlled synthetic example · fixed entity futures · robot-only updates", ha="center", color="#475569", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))

    def frame(index):
        line.set_data(trajectories[index][0, :, 0], trajectories[index][0, :, 1])
        energy_line.set_data(np.arange(index + 1), energies[:index + 1])
        title.set_text(f"SAGE | trajectory refinement {index + 1}/{len(trajectories)}")
        return line, energy_line, title

    animation = FuncAnimation(fig, frame, frames=len(trajectories), interval=120)
    animation.save(output / "sage_demo.gif", writer=PillowWriter(fps=8))
    frame(len(trajectories) - 1)
    fig.savefig(output / "sage_demo.png", dpi=150)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = run(args.output, args.iterations, args.seed)
    print(f"energy: {result['energy_before']:.4f} -> {result['energy_after']:.4f}")
    print(f"Animation and summary: {args.output.resolve()}")


if __name__ == "__main__":
    main()
