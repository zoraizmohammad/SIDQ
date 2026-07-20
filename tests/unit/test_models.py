"""Tests for model components."""

import torch

from sidq.models.losses import ConsistencyLoss, FocalLoss, MultiViewLoss
from sidq.models.raw.raw_specialist import RawSpecialist
from sidq.models.ssl.layer_mixing import LearnedLayerMixing


class TestLearnedLayerMixing:
    def test_forward(self):
        mixer = LearnedLayerMixing(num_layers=13)
        states = tuple(torch.randn(2, 50, 768) for _ in range(13))
        out = mixer(states)
        assert out.shape == (2, 50, 768)

    def test_weights_sum_to_one(self):
        mixer = LearnedLayerMixing(num_layers=5, normalize=True)
        w = torch.softmax(mixer.weights, dim=0)
        assert abs(w.sum().item() - 1.0) < 1e-5


class TestRawSpecialist:
    def test_forward(self):
        model = RawSpecialist(sinc_channels=20, channels=[32, 32])
        wav = torch.randn(2, 16000)
        logits = model(wav)
        assert logits.shape == (2, 2)

    def test_backward(self):
        model = RawSpecialist(sinc_channels=20, channels=[16, 16])
        wav = torch.randn(1, 8000, requires_grad=True)
        logits = model(wav)
        logits.sum().backward()
        assert wav.grad is not None


class TestFocalLoss:
    def test_reduces(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_gamma_zero_equals_ce(self):
        loss_fn = FocalLoss(gamma=0.0)
        ce_fn = torch.nn.CrossEntropyLoss()
        logits = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        focal = loss_fn(logits, targets)
        ce = ce_fn(logits, targets)
        assert abs(focal.item() - ce.item()) < 1e-4


class TestConsistencyLoss:
    def test_mse(self):
        loss_fn = ConsistencyLoss(method="mse", weight=1.0)
        a = torch.randn(4, 2)
        b = a + 0.1 * torch.randn(4, 2)
        loss = loss_fn(a, b)
        assert loss.item() > 0

    def test_identical_inputs(self):
        loss_fn = ConsistencyLoss(method="mse", weight=1.0)
        a = torch.randn(4, 2)
        loss = loss_fn(a, a.clone())
        assert loss.item() < 1e-6


class TestMultiViewLoss:
    def test_forward(self):
        loss_fn = MultiViewLoss()
        clean = torch.randn(8, 2)
        corrupted = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        result = loss_fn(clean, corrupted, targets)
        assert "total" in result
        assert "clean" in result
        assert "consistency" in result
        assert result["total"].item() > 0
