"""Tests for training engine."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from sidq.training.engine import Trainer


class SimpleModel(torch.nn.Module):
    """Tiny model for training smoke tests."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(100, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

    def forward(self, waveform, lengths=None):
        # Fake: just use first 100 samples as features
        x = waveform[:, :100]
        if x.shape[1] < 100:
            x = torch.nn.functional.pad(x, (0, 100 - x.shape[1]))
        return self.net(x)


def _make_loader(n: int = 64, seq_len: int = 1000) -> DataLoader:
    waveforms = torch.randn(n, seq_len)
    labels = torch.randint(0, 2, (n,))
    dataset = TensorDataset(waveforms, labels)

    def collate(batch):
        ws = torch.stack([b[0] for b in batch])
        ls = torch.stack([b[1] for b in batch])
        return {"waveforms": ws, "labels": ls}

    return DataLoader(dataset, batch_size=16, collate_fn=collate)


class TestTrainer:
    def test_train_epoch(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            mixed_precision=False,
        )
        loader = _make_loader()
        loss = trainer.train_epoch(loader)
        assert loss > 0
        assert trainer.state.global_step > 0

    def test_validate(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            mixed_precision=False,
        )
        loader = _make_loader()
        metrics = trainer.validate(loader)
        assert "val_eer" in metrics
        assert 0 <= metrics["val_eer"] <= 1.0

    def test_fit_short(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            mixed_precision=False,
            early_stopping_patience=2,
        )
        train_loader = _make_loader()
        val_loader = _make_loader(n=32)
        result = trainer.fit(train_loader, val_loader, epochs=3)
        assert result.total_epochs <= 3
        assert result.best_eer >= 0

    def test_checkpoint_save_load(self, tmp_path):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            mixed_precision=False,
            checkpoint_dir=tmp_path,
        )
        trainer.state.best_eer = 0.15
        trainer._save_checkpoint("test.pt")
        assert (tmp_path / "test.pt").exists()

        # Load into fresh trainer
        model2 = SimpleModel()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        trainer2 = Trainer(
            model=model2,
            optimizer=optimizer2,
            device=torch.device("cpu"),
            mixed_precision=False,
        )
        trainer2.load_checkpoint(tmp_path / "test.pt")
        assert trainer2.state.best_eer == 0.15
