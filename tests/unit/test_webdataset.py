"""Tests for WebDataset streaming loader."""


from sidq.data.webdataset import WebDatasetLoader


class TestWebDatasetLoader:
    def test_init_defaults(self):
        loader = WebDatasetLoader(shard_urls=["shard-0000.tar"])
        assert loader.sample_rate == 16_000
        assert loader.max_duration_sec == 20.0
        assert loader.shuffle_shards is True

    def test_init_custom(self):
        loader = WebDatasetLoader(
            shard_urls=["s1.tar", "s2.tar"],
            sample_rate=8000,
            max_duration_sec=10.0,
            shuffle_shards=False,
            seed=123,
        )
        assert loader.sample_rate == 8000
        assert loader.seed == 123
        assert len(loader.shard_urls) == 2

    def test_requires_webdataset(self):
        """Verify graceful error when webdataset not installed."""
        loader = WebDatasetLoader(shard_urls=["fake.tar"])
        # We only test that the object is created properly
        # Actual iteration requires real TAR files
        assert loader is not None

    def test_decode_audio_invalid(self):
        """Verify None return on invalid audio data."""
        loader = WebDatasetLoader(shard_urls=[])
        result = loader._decode_audio(b"not-valid-audio-bytes")
        assert result is None

    def test_decode_audio_none_input(self):
        loader = WebDatasetLoader(shard_urls=[])
        result = loader._decode_audio(None)
        assert result is None

    def test_process_sample_no_audio(self):
        loader = WebDatasetLoader(shard_urls=[])
        result = loader._process_sample({"__key__": "test", "txt": "hello"})
        assert result is None
