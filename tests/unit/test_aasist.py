"""Tests for AASIST backend."""

import torch

from sidq.models.backends.aasist import AASISTBackend, GraphAttentionLayer


class TestGraphAttentionLayer:
    def test_forward_shape(self):
        layer = GraphAttentionLayer(64, 32)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 32)

    def test_with_adjacency(self):
        layer = GraphAttentionLayer(64, 32)
        x = torch.randn(2, 4, 64)
        adj = torch.ones(2, 4, 4)
        out = layer(x, adj=adj)
        assert out.shape == (2, 4, 32)


class TestAASISTBackend:
    def test_forward_shape(self):
        backend = AASISTBackend(input_dim=1024, hidden_dim=160, num_classes=2)
        features = torch.randn(4, 50, 1024)
        logits = backend(features)
        assert logits.shape == (4, 2)

    def test_with_lengths(self):
        backend = AASISTBackend(input_dim=256, hidden_dim=64, num_classes=2)
        features = torch.randn(3, 40, 256)
        lengths = torch.tensor([40, 30, 20])
        logits = backend(features, lengths=lengths)
        assert logits.shape == (3, 2)

    def test_backward(self):
        backend = AASISTBackend(input_dim=128, hidden_dim=32, num_classes=2)
        features = torch.randn(2, 20, 128, requires_grad=True)
        logits = backend(features)
        loss = logits.sum()
        loss.backward()
        assert features.grad is not None

    def test_small_config(self):
        backend = AASISTBackend(
            input_dim=64,
            hidden_dim=16,
            num_graph_layers=1,
            num_spec_nodes=2,
            num_temp_nodes=2,
            num_classes=2,
        )
        features = torch.randn(1, 10, 64)
        logits = backend(features)
        assert logits.shape == (1, 2)
