from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Configuration for the cipher dataset generation.

    A redundancy value of 0 denotes the monoalphabetic baseline for that length.

    Attributes:
        foundation_pct (float): The percentage of ciphers to use as the foundation.
        transition_pct (float): The percentage of ciphers to use as transitions.
        frontier_pct (float): The percentage of ciphers to use as the frontier.
        test_matrix (Dict[int, List[int]]): A mapping of length to redundancy levels.

    """

    training_num: int = 1_000_000
    validation_num: int = 10_000

    foundation_pct: float = 0.10
    foundation_range: tuple[int, int] = (350, 1000)
    transition_pct: float = 0.20
    transition_range: tuple[int, int] = (1000, 4000)
    frontier_pct: float = 0.70
    frontier_range: tuple[int, int] = (4000, 10000)

    test_matrix: dict[int, list[int]] = field(
        default_factory=lambda: {
            350: [5, 10, 15, 0],
            400: [5, 10, 15, 0],
            450: [5, 10, 15, 20, 0],
            600: [5, 10, 15, 20, 0],
            800: [5, 10, 15, 20, 25, 30, 0],
            1000: [5, 10, 15, 20, 25, 30, 0],
            2000: [5, 10, 15, 20, 25, 30, 50, 0],
            4000: [5, 10, 15, 20, 25, 30, 50, 100, 0],
            6000: [5, 10, 15, 20, 25, 30, 50, 100, 150, 0],
            8000: [5, 10, 15, 20, 25, 30, 50, 100, 200, 300, 0],
            10000: [5, 10, 15, 20, 25, 30, 50, 100, 200, 300, 0],
        },
    )
    ciphers_per_bin: int = 100

    def __post_init__(self) -> None:
        """Validate the distribution splits immediately after initialization.

        Raises:
            ValueError: If the distribution percentages do not sum to 1.0.

        """
        total = self.foundation_pct + self.transition_pct + self.frontier_pct
        if round(total, 5) != 1.0:
            raise ValueError("Training distribution percentages must sum to 1.0")

    @property
    def num_test_ciphers(self) -> int:
        """Get the total number of test ciphers across all bins."""
        return (
            sum(len(diffs) for diffs in self.test_matrix.values())
            * self.ciphers_per_bin
        )


@dataclass
class CipherConfig:
    """Configuration for the cipher generation process.

    Attributes:
        train_folder (str): The folder ID for the training split.
        val_folder (str): The folder ID for the validation split.
        test_folder (str): The folder ID for the test split.
        metadata_folder (str): The folder ID for the metadata split.
        num_workers (int): The number of workers to use for parallel processing.
        batch_size (int): The number of ciphers to generate in a single batch.
        dataset_config (DatasetConfig): The dataset configuration.

    """

    train_folder: str
    val_folder: str
    test_folder: str
    metadata_folder: str
    num_workers: int | None = None
    batch_size: int = 10_000
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
