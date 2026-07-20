"""Tests for inference scoring pipeline."""

import numpy as np
import torch

from sidq.inference.scorer import collate_variable_length, score_batch


class MockModel(torch.nn.Module):
    """Mock model that returns fixed scores for testing."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Linear(1, 1)

    def forward(self, waveform, lengths=None):
        batch_size = waveform.shape[0]
        # Return logits: [spoofed_score, bonafide_score]
        spoofed = torch.zeros(batch_size, 1)
        bonafide = torch.ones(batch_size, 1) * 2.0
        return torch.cat([spoofed, bonafide], dim=-1)


class TestCollateVariableLength:
    def test_pads_to_max_length(self):
        batch = [
            {"audio_id": "a", "waveform": torch.randn(8000)},
            {"audio_id": "b", "waveform": torch.randn(16000)},
        ]
        result = collate_variable_length(batch)
        assert result["waveforms"].shape == (2, 16000)
        assert result["lengths"].tolist() == [8000, 16000]
        assert result["audio_ids"] == ["a", "b"]

    def test_equal_lengths(self):
        batch = [
            {"audio_id": "x", "waveform": torch.randn(10000)},
            {"audio_id": "y", "waveform": torch.randn(10000)},
        ]
        result = collate_variable_length(batch)
        assert result["waveforms"].shape == (2, 10000)


class TestScoreBatch:
    def test_returns_bonafide_scores(self):
        model = MockModel()
        waveforms = torch.randn(4, 16000)
        scores = score_batch(model, waveforms, device=torch.device("cpu"))
        assert scores.shape == (4,)
        assert np.allclose(scores, 2.0)

    def test_score_direction(self):
        """Verify that returned score is the bonafide class (index 1)."""
        model = MockModel()
        waveforms = torch.randn(2, 16000)
        scores = score_batch(model, waveforms, device=torch.device("cpu"))
        # MockModel returns bonafide=2.0, spoofed=0.0
        # Higher score means more bonafide — correct direction
        assert all(s > 0 for s in scores)
