import pytest

from gen_training import get_folder_id, load_folder_ids, get_text_stream
from cipher_generation.config import DatasetConfig


class TestEnvironmentVariables:
    def test_get_folder_id_success(self, monkeypatch):
        """Test that get_folder_id correctly retrieves an existing environment variable."""
        monkeypatch.setenv("TEST_CIPHER_ENV_VAR", "mock_folder_123")

        result = get_folder_id("TEST_CIPHER_ENV_VAR")

        assert result == "mock_folder_123"

    def test_get_folder_id_raises_error(self, monkeypatch):
        """Test that get_folder_id raises an OSError when the variable is missing."""
        monkeypatch.delenv("MISSING_ENV_VAR", raising=False)

        with pytest.raises(
            OSError, match="Environment variable MISSING_ENV_VAR not set"
        ):
            get_folder_id("MISSING_ENV_VAR")

    def test_load_folder_ids_success(self, monkeypatch):
        """Test that load_folder_ids successfully fetches all required IDs and returns them in order."""
        monkeypatch.setenv("FOLDER_ID_TRAIN", "train_xyz")
        monkeypatch.setenv("FOLDER_ID_VAL", "val_xyz")
        monkeypatch.setenv("FOLDER_ID_TEST", "test_xyz")
        monkeypatch.setenv("FOLDER_ID_METADATA", "meta_xyz")

        train, val, test, meta = load_folder_ids()

        assert train == "train_xyz"
        assert val == "val_xyz"
        assert test == "test_xyz"
        assert meta == "meta_xyz"


class TestGetTextStream:
    @pytest.fixture
    def mock_dependencies(self, mocker):
        """Provides mocked instances of the internal utilities called by get_text_stream."""
        return {
            "extractor_cls": mocker.patch("gen_training.DatasetExtractor"),
            "load_genres": mocker.patch("gen_training.load_existing_genre_map"),
            "randomize": mocker.patch("gen_training.randomize_stream"),
            "sampler_cls": mocker.patch("gen_training.CorpusSampler"),
        }

    @pytest.fixture
    def dummy_config(self):
        """Provides a minimal DatasetConfig for routing tests."""
        return DatasetConfig(training_num=10, validation_num=2, test_matrix={350: [0]})

    def test_get_text_stream_default_extractor(self, mock_dependencies, dummy_config):
        """Test that get_text_stream initializes default dependencies and routes data correctly."""
        mock_extractor_instance = mock_dependencies["extractor_cls"].return_value
        mock_sampler_instance = mock_dependencies["sampler_cls"].return_value

        result = get_text_stream(config=dummy_config)

        mock_dependencies["extractor_cls"].assert_called_once()
        mock_dependencies["load_genres"].assert_called_once()

        mock_dependencies["randomize"].assert_called_once_with(
            mock_extractor_instance.get_full_stream.return_value
        )

        mock_dependencies["sampler_cls"].assert_called_once_with(
            dummy_config, mock_dependencies["load_genres"].return_value
        )

        mock_sampler_instance.generate_stream.assert_called_once_with(
            mock_dependencies["randomize"].return_value
        )

        assert result == (
            mock_sampler_instance,
            mock_sampler_instance.generate_stream.return_value,
        )

    def test_get_text_stream_custom_extractor(
        self, mocker, mock_dependencies, dummy_config
    ):
        """Test that passing a custom extractor bypasses the default initialization."""
        custom_extractor = mocker.Mock()
        mock_sampler_instance = mock_dependencies["sampler_cls"].return_value

        result = get_text_stream(config=dummy_config, extractor=custom_extractor)

        mock_dependencies["extractor_cls"].assert_not_called()

        mock_dependencies["randomize"].assert_called_once_with(
            custom_extractor.get_full_stream.return_value
        )

        mock_dependencies["sampler_cls"].assert_called_once_with(
            dummy_config, mock_dependencies["load_genres"].return_value
        )

        assert result == (
            mock_sampler_instance,
            mock_sampler_instance.generate_stream.return_value,
        )
