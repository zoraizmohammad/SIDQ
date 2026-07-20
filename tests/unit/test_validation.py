"""Tests for dataset audit and validation."""

from sidq.data.schemas import ManifestEntry, SplitManifest
from sidq.data.validation import audit_manifest


class TestAuditManifest:
    def _entries(self, n: int, labels: list[int | None] | None = None) -> list[ManifestEntry]:
        if labels is None:
            labels = [1 if i % 2 == 0 else 0 for i in range(n)]
        return [
            ManifestEntry(
                audio_id=f"test_{i:04d}",
                relative_path=f"test_{i:04d}.flac",
                label=labels[i] if i < len(labels) else None,
                duration_sec=float(2 + i % 5),
            )
            for i in range(n)
        ]

    def test_clean_manifest(self):
        entries = self._entries(100)
        manifest = SplitManifest(split_name="train", entries=entries)
        report = audit_manifest(manifest)
        assert report.is_clean
        assert report.num_samples == 100
        assert 0 in report.label_counts
        assert 1 in report.label_counts

    def test_duration_stats(self):
        entries = self._entries(10)
        manifest = SplitManifest(split_name="dev", entries=entries)
        report = audit_manifest(manifest)
        assert "mean" in report.duration_stats
        assert "median" in report.duration_stats
        assert report.duration_stats["min"] >= 2.0

    def test_missing_labels_detected(self):
        entries = [
            ManifestEntry(audio_id="a", relative_path="a.flac", label=1),
            ManifestEntry(audio_id="b", relative_path="b.flac", label=None),
            ManifestEntry(audio_id="c", relative_path="c.flac", label=0),
        ]
        manifest = SplitManifest(split_name="train", entries=entries)
        report = audit_manifest(manifest)
        assert not report.is_clean
        assert "b" in report.missing_labels

    def test_duplicates_detected(self):
        entries = [
            ManifestEntry(audio_id="x", relative_path="x.flac", label=1),
            ManifestEntry(audio_id="x", relative_path="x2.flac", label=0),
        ]
        manifest = SplitManifest(split_name="train", entries=entries)
        report = audit_manifest(manifest)
        assert not report.is_clean
        assert "x" in report.duplicate_ids
