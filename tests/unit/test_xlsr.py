"""Tests for XLS-R frontend (without downloading full model)."""

import pytest
import torch

from sidq.models.ssl.xlsr import XLSRFrontend


class TestXLSRFrontend:
    def test_init_defaults(self):
        frontend = XLSRFrontend()
        assert frontend.freeze is True
        assert frontend.output_dim == 1024
        assert not frontend.is_loaded

    def test_init_with_layer_weights(self):
        frontend = XLSRFrontend(layer_weights=True)
        assert frontend.use_layer_weights is True
        assert frontend.layer_weights.shape[0] == 25  # 24 layers + 1

    def test_forward_without_loading_raises(self):
        frontend = XLSRFrontend()
        with pytest.raises(RuntimeError, match="not loaded"):
            frontend(torch.randn(1, 16000))

    def test_load_invalid_model_raises(self):
        frontend = XLSRFrontend(model_name="nonexistent/model-xyz-fake")
        with pytest.raises(RuntimeError, match="Failed to load"):
            frontend.load_pretrained()

    def test_output_dim(self):
        frontend = XLSRFrontend()
        assert frontend.output_dim == 1024

    def test_layer_weights_gradient(self):
        frontend = XLSRFrontend(layer_weights=True, freeze=True)
        assert frontend.layer_weights.requires_grad is True
