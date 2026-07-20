"""Tests for deterministic validation splitting."""

import pytest

from sidq.data.schemas import ManifestEntry, SplitManifest
from sidq.data.splitting import SplitConfig, compute_split_hash, create_validation_split


def _make_manifest(n: int = 100) -> SplitManifest:
    entries = [
        ManifestEntry(
            audio_id=f"sample_{i:04d}",
            relative_path=f"sample_{i:04d}.flac",
            label=1 if i % 2 == 0 else 0,
            duration_sec=3.0 + (i % 10) * 0.5,
        )
        for i in range(n)
    ]
    return SplitManifest(split_name="train", entries=entries)


class TestCreateValidationSplit:
    def test_no_overlap(self):
        manifest = _make_manifest(200)
        train, val = create_validation_split(manifest)
        train_ids = {e.audio_id for e in train.entries}
        val_ids = {e.audio_id for e in val.entries}
        assert len(train_ids & val_ids) == 0

    def test_covers_all_samples(self):
        manifest = _make_manifest(200)
        train, val = create_validation_split(manifest)
        assert train.num_samples + val.num_samples == 200

    def test_deterministic(self):
        manifest = _make_manifest(200)
        config = SplitConfig(seed=42)
        t1, v1 = create_validation_split(manifest, config)
        t2, v2 = create_validation_split(manifest, config)
        assert {e.audio_id for e in t1.entries} == {e.audio_id for e in t2.entries}
        assert {e.audio_id for e in v1.entries} == {e.audio_id for e in v2.entries}

    def test_different_seed_different_split(self):
        manifest = _make_manifest(200)
        _, v1 = create_validation_split(manifest, SplitConfig(seed=42))
        _, v2 = create_validation_split(manifest, SplitConfig(seed=99))
        ids1 = {e.audio_id for e in v1.entries}
        ids2 = {e.audio_id for e in v2.entries}
        assert ids1 != ids2

    def test_stratified_preserves_proportions(self):
        manifest = _make_manifest(1000)
        config = SplitConfig(val_fraction=0.2, stratify_by_label=True)
        train, val = create_validation_split(manifest, config)

        train_pos = sum(1 for e in train.entries if e.label == 1)
        val_pos = sum(1 for e in val.entries if e.label == 1)

        train_ratio = train_pos / train.num_samples
        val_ratio = val_pos / val.num_samples
        assert abs(train_ratio - val_ratio) < 0.05

    def test_unlabeled_raises(self):
        entries = [
            ManifestEntry(audio_id=f"x_{i}", relative_path=f"x_{i}.flac", label=None)
            for i in range(50)
        ]
        manifest = SplitManifest(split_name="test", entries=entries)
        with pytest.raises(ValueError, match="no labeled"):
            create_validation_split(manifest)


class TestSplitHash:
    def test_deterministic_hash(self):
        manifest = _make_manifest(100)
        t1, v1 = create_validation_split(manifest)
        t2, v2 = create_validation_split(manifest)
        assert compute_split_hash(t1, v1) == compute_split_hash(t2, v2)

    def test_different_splits_different_hash(self):
        manifest = _make_manifest(100)
        t1, v1 = create_validation_split(manifest, SplitConfig(seed=1))
        t2, v2 = create_validation_split(manifest, SplitConfig(seed=2))
        assert compute_split_hash(t1, v1) != compute_split_hash(t2, v2)
