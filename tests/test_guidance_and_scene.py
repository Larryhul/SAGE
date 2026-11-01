import pytest
import torch

from sage.data.heterogeneous_scene import integrate_velocities
from sage.experiments.smoke_test import make_synthetic_batch
from sage.guidance.safety_social import SafetySocialGuidance, SafetySocialGuidanceConfig
from sage.models.social_force import social_force_baseline


def test_integrate_velocities_is_cumulative_sum():
    vel = torch.ones(1, 1, 4, 2)
    pos = integrate_velocities(vel, torch.zeros(1, 1, 2), dt=0.5)
    assert torch.allclose(pos[0, 0, :, 0], torch.tensor([0.5, 1.0, 1.5, 2.0]))


def test_integrate_velocities_rejects_wrong_rank():
    with pytest.raises(ValueError):
        integrate_velocities(torch.ones(1, 4, 2), torch.zeros(1, 1, 2), dt=0.4)


def test_synthetic_batch_is_valid():
    batch = make_synthetic_batch()
    batch.validate()
    assert batch.num_robots >= 1 and batch.num_entities >= 1


def test_validate_rejects_bad_goal_shape():
    batch = make_synthetic_batch()
    batch.goals = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError):
        batch.validate()


def test_guidance_gradient_only_moves_robots():
    batch = make_synthetic_batch()
    guidance = SafetySocialGuidance(SafetySocialGuidanceConfig())
    grad = guidance.robot_only_gradient(batch.future.clone(), batch)
    assert torch.isfinite(grad).all()
    entity = ~batch.robot_mask[:, :, None, None]
    assert torch.count_nonzero(grad * entity.to(grad.dtype)) == 0


def test_guidance_energy_is_finite_per_scene():
    batch = make_synthetic_batch()
    guidance = SafetySocialGuidance(SafetySocialGuidanceConfig())
    energy = guidance.energy(batch.future, batch)
    assert torch.isfinite(energy).all()


def test_velocity_projection_bounds_robot_speed():
    batch = make_synthetic_batch()
    cfg = SafetySocialGuidanceConfig()
    guidance = SafetySocialGuidance(cfg)
    fast = batch.future * 100.0
    out = guidance.project_robot_velocities(fast, batch)
    speed = torch.linalg.norm(out, dim=-1)
    limit = cfg.max_speed * cfg.guidance_velocity_clip_factor
    robot = batch.robot_mask[:, :, None].expand_as(speed)
    assert (speed[robot] <= limit + 1e-5).all()


def test_social_force_baseline_shape_and_determinism():
    batch = make_synthetic_batch()
    a = social_force_baseline(batch)
    b = social_force_baseline(batch)
    assert a.shape == batch.future.shape
    assert torch.isfinite(a).all()
    assert torch.equal(a, b)
