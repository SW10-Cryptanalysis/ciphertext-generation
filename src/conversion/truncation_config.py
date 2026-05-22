from dataclasses import dataclass, field
from typing import Callable
import random


@dataclass
class TruncationConfig:
    """Configuration for the cipher truncation process."""

    max_length: int = 1000
    length_distribution: list[tuple[tuple[int, int], float]] = field(
        default_factory=lambda: [
            ((100, 1000), 1),
        ],
    )


def create_length_sampler(config: TruncationConfig) -> Callable[[], int]:
    """Create a length sampler based on the provided configuration.

    Args:
        config (TruncationConfig): The configuration object.

    Returns:
        Callable[[], int]: A function that returns a random length.

    """
    ranges = [item[0] for item in config.length_distribution]
    weights = [item[1] for item in config.length_distribution]

    def sampler() -> int:
        """Sample a target sequence length."""
        selected_range = random.choices(ranges, weights=weights, k=1)[0]
        return random.randint(selected_range[0], selected_range[1])

    return sampler
