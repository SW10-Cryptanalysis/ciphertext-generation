import argparse
import json
from pathlib import Path
from tqdm import tqdm

from dataset_stats.dataset_stats_aggregator import DatasetStatsAggregator
from utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        "-d",
        type=Path,
        required=True,
        help="The parent directory containing split subdirectories (e.g., Training, "
            "Validation) with .jsonl files.",
    )
    return parser.parse_args()


def record_cipher_stats(
    stats: DatasetStatsAggregator,
    file_path: Path,
    split_name: str,
) -> None:
    """Record stats for a single cipher file.

    Args:
        stats (DatasetStatsAggregator): The stats object to record to.
        file_path (Path): The path to the cipher file.
        split_name (str): The name of the split the file belongs to.

    """
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            cipher_data = json.loads(line)

            stats.record(
                split=split_name,
                length=len(cipher_data.get("plaintext", "")),
                homophones=cipher_data.get("num_symbols", 0),
                redundancy=cipher_data.get("redundancy", 0),
                genres=cipher_data.get("genres", ["unknown"]),
            )


def generate_global_stats_from_existing(parent_dir: Path) -> None:
    """Read existing cipher files across all splits and generate global statistics."""
    stats = DatasetStatsAggregator()

    cipher_files = list(parent_dir.rglob("*.jsonl"))

    if not cipher_files:
        logger.error(f"No .jsonl files found in or below {parent_dir}")
        return

    for file_path in tqdm(cipher_files, desc="Scanning all split files"):
        try:
            split_name = file_path.relative_to(parent_dir).parts[0]
        except ValueError:
            split_name = "Unknown"

        record_cipher_stats(stats, file_path, split_name)

    stats_file = parent_dir / "metadata_truncated.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats.__json__(), f, indent=4)

    logger.info(f"Successfully wrote comprehensive stats to {stats_file}")


if __name__ == "__main__":
    args = parse_args()
    generate_global_stats_from_existing(args.directory)
