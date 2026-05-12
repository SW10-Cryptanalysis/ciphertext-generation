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
    """Read existing cipher files sequentially for Training then Validation splits."""
    stats = DatasetStatsAggregator()
    parent_dir = parent_dir.resolve()

    target_splits = {"Training": "train", "Validation": "val"}

    for folder_name, split_key in target_splits.items():
        split_dir = parent_dir / folder_name

        if not split_dir.exists():
            logger.warning(f"Split directory missing, skipping: {split_dir}")
            continue

        cipher_files = list(split_dir.rglob("*.jsonl"))

        if not cipher_files:
            logger.warning(f"No .jsonl files found in {split_dir}")
            continue

        for file_path in tqdm(
            cipher_files,
            desc=f"Scanning {folder_name} -> '{split_key}'",
        ):
            record_cipher_stats(stats, file_path, split_key)

    stats_file = parent_dir / "metadata_truncated.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats.__json__(), f, indent=4)

    logger.info(f"Successfully wrote comprehensive stats to {stats_file}")


if __name__ == "__main__":
    args = parse_args()
    generate_global_stats_from_existing(args.directory)
