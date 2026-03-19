import pytest
from cipher_generation.config import DatasetConfig, CipherConfig


class TestDatasetConfig:
    """Test suite for the DatasetConfig dataclass."""

    def test_default_initialization(self):
        """Verify default values initialize successfully and pass validation."""
        config = DatasetConfig()
        assert config.training_num == 1_000_000
        assert config.validation_num == 10_000
        assert config.foundation_pct == 0.10
        assert config.transition_pct == 0.20
        assert config.frontier_pct == 0.70

    @pytest.mark.parametrize(
        "foundation, transition, frontier",
        [
            (0.50, 0.25, 0.25),
            (0.0, 0.0, 1.0),
            (0.33333, 0.33333, 0.33334),
        ],
    )
    def test_valid_percentages(self, foundation, transition, frontier):
        """Verify that various percentage combinations summing to 1.0 pass validation."""
        config = DatasetConfig(
            foundation_pct=foundation,
            transition_pct=transition,
            frontier_pct=frontier,
        )
        total = config.foundation_pct + config.transition_pct + config.frontier_pct
        assert round(total, 5) == 1.0

    @pytest.mark.parametrize(
        "foundation, transition, frontier",
        [
            (0.50, 0.50, 0.50),
            (0.10, 0.10, 0.10),
            (0.99, 0.0, 0.0),
        ],
    )
    def test_invalid_percentages_raise_error(self, foundation, transition, frontier):
        """Verify that percentages not summing to 1.0 raise a ValueError."""
        with pytest.raises(
            ValueError, match="Training distribution percentages must sum to 1.0"
        ):
            DatasetConfig(
                foundation_pct=foundation,
                transition_pct=transition,
                frontier_pct=frontier,
            )

    def test_test_matrix_isolation(self):
        """Verify the default factory creates a new dictionary for each instance."""
        config1 = DatasetConfig()
        config2 = DatasetConfig()

        config1.test_matrix[9999] = [1, 2, 3]

        assert 9999 in config1.test_matrix
        assert 9999 not in config2.test_matrix

    def test_num_test_ciphers_default(self):
        """Verify the default test matrix and ciphers_per_bin calculate correctly."""
        config = DatasetConfig()

        assert config.num_test_ciphers == 8100

    def test_num_test_ciphers_custom(self):
        """Verify the property dynamically calculates based on custom inputs."""
        config = DatasetConfig(
            test_matrix={10: [1, 2], 20: [3]},
            ciphers_per_bin=50,
        )
        assert config.num_test_ciphers == 150

    def test_num_test_ciphers_empty_matrix(self):
        """Verify the property safely handles an empty test matrix."""
        config = DatasetConfig(
            test_matrix={},
            ciphers_per_bin=100,
        )
        assert config.num_test_ciphers == 0


class TestCipherConfig:
    """Test suite for the CipherConfig dataclass."""

    def test_initialization_with_defaults(self):
        """Verify CipherConfig initializes properly with required parameters."""
        config = CipherConfig(
            train_folder="train_id",
            val_folder="val_id",
            test_folder="test_id",
            metadata_folder="meta_id",
        )

        assert config.train_folder == "train_id"
        assert config.batch_size == 10_000
        assert config.num_workers is None
        assert isinstance(config.dataset_config, DatasetConfig)

    def test_dataset_config_isolation(self):
        """Verify the default factory creates a new DatasetConfig for each CipherConfig."""
        config1 = CipherConfig("t", "v", "te", "m")
        config2 = CipherConfig("t", "v", "te", "m")

        config1.dataset_config.training_num = 99

        assert config1.dataset_config.training_num == 99
        assert config2.dataset_config.training_num == 1_000_000
