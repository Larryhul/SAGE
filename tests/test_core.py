import pytest
import torch

from sage.evaluation.metrics import ade, fde
from sage.experiments.smoke_test import make_synthetic_batch
from sage.models.sage_model import SAGE, SAGEConfig


def _model(batch):
    cfg = SAGEConfig()
    cfg.encoder.use_goal_conditioning = True
    cfg.guidance.w_goal = 0.3
    cfg.diffusion.horizon = batch.horizon
    return SAGE(cfg)


def test_ade_fde_known_values():
    gt = torch.zeros(1, 2, 3, 2)
    pred = gt.clone()
    pred[:, 0, :, 0] = torch.tensor([0.0, 1.0, 2.0])
    pred[:, 1, :, 0] = 3.0
    mask = torch.tensor([[True, False]])
    assert ade(pred, gt, mask).item() == pytest.approx(1.0)
    assert fde(pred, gt, mask).item() == pytest.approx(2.0)


def test_ade_fde_all_masked_is_zero():
    gt = torch.zeros(1, 2, 3, 2)
    pred = torch.ones_like(gt)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    assert ade(pred, gt, mask).item() == 0.0
    assert fde(pred, gt, mask).item() == 0.0


def test_forward_backward_gives_finite_loss_and_gradients():
    torch.manual_seed(0)
    batch = make_synthetic_batch()
    model = _model(batch)
    out = model(batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("guided", [False, True])
def test_generate_shape_and_finiteness(guided):
    torch.manual_seed(0)
    batch = make_synthetic_batch()
    model = _model(batch).eval()
    with torch.no_grad():
        sample = model.generate(batch, guided=guided, step_stride=10)
    assert sample["future_vel"].shape == batch.future.shape
    assert torch.isfinite(sample["future_vel"]).all()
