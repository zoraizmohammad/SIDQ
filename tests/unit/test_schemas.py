"""Tests for data schemas and manifests."""

import pytest
from pydantic import ValidationError

from sidq.data.schemas import AudioSample, ManifestEntry, SplitManifest


class TestAudioSample:
    def test_valid_bonafide(self):
        s = AudioSample(audio_id="test_001", label=1)
        assert s.label == 1

    def test_valid_spoofed(self):
        s = AudioSample(audio_id="test_002", label=0)
        assert s.label == 0

    def test_invalid_label(self):
        with pytest.raises(ValidationError):
            AudioSample(audio_id="test_003", label=2)

    def test_negative_label(self):
        with pytest.raises(ValidationError):
            AudioSample(audio_id="test_004", label=-1)

    def test_none_label_allowed(self):
        s = AudioSample(audio_id="test_005", label=None)
        assert s.label is None

    def test_empty_id_rejected(self):
        with pytest.raises(ValidationError):
            AudioSample(audio_id="", label=1)

    def test_whitespace_id_rejected(self):
        with pytest.raises(ValidationError):
            AudioSample(audio_id="   ", label=1)


class TestSplitManifest:
    def _make_entries(self, ids: list[str]) -> list[ManifestEntry]:
        return [ManifestEntry(audio_id=aid, relative_path=f"{aid}.flac") for aid in ids]

    def test_no_duplicates(self):
        entries = self._make_entries(["a", "b", "c"])
        manifest = SplitManifest(split_name="train", entries=entries)
        assert manifest.validate_no_duplicates() == []

    def test_duplicates_detected(self):
        entries = self._make_entries(["a", "b", "a"])
        manifest = SplitManifest(split_name="train", entries=entries)
        assert "a" in manifest.validate_no_duplicates()

    def test_num_samples(self):
        entries = self._make_entries(["x", "y", "z"])
        manifest = SplitManifest(split_name="dev", entries=entries)
        assert manifest.num_samples == 3

    def test_audio_ids(self):
        entries = self._make_entries(["foo", "bar"])
        manifest = SplitManifest(split_name="test", entries=entries)
        assert manifest.audio_ids == {"foo", "bar"}

    def test_missing_labels_in_labeled_split(self):
        entries = [
            ManifestEntry(audio_id="a", relative_path="a.flac", label=1),
            ManifestEntry(audio_id="b", relative_path="b.flac", label=None),
        ]
        manifest = SplitManifest(split_name="train", entries=entries)
        missing = manifest.validate_labels()
        assert "b" in missing
        assert "a" not in missing
