import json
from pathlib import Path
from typing import Any, Iterator
import argparse
from truncation.cipher_truncator import CipherTruncator
from truncation.truncation_config import TruncationConfig, create_length_sampler
from truncation.dataset_writer import DatasetWriter


def discover_dataset_files(dataset_dir: Path, extension: str = ".jsonl") -> list[Path]:
    """Discover all files in the dataset directory with the specified extension.

    Args:
        dataset_dir (Path): The directory to search for files.
        extension (str, optional): The file extension to search for.
            Defaults to ".jsonl".

    Returns:
        list[Path]: A list of all files with the specified extension in the dataset
            directory.

    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    file_paths = list(dataset_dir.rglob(extension))
    if not file_paths:
        raise FileNotFoundError(
            f"No files found with extension {extension} in {dataset_dir}",
        )

    return sorted(file_paths)


def generate_continuous_stream(file_paths: list[Path]) -> Iterator[dict[str, Any]]:
    """Generate a continuous stream of json data from a list of file paths.

    Args:
        file_paths (list[Path]): A list of file paths to read from.

    Yields:
        dict[str, Any]: A dictionary containing the json data from each file.

    """
    for file_path in file_paths:
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        "-d",
        type=Path,
        required=True,
        help="The directory containing the dataset files.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="The output directory for the truncated dataset.",
    )

    return parser.parse_args()


def run_pipeline(dataset_dir: Path, output_dir: Path) -> None:
    """Run the preprocessing pipeline.

    Args:
        dataset_dir (Path): The directory containing the dataset files.
        output_dir (Path): The directory to write the truncated files to.

    """
    file_paths = discover_dataset_files(dataset_dir)
    continuous_stream = generate_continuous_stream(file_paths)

    config = TruncationConfig()
    length_sampler = create_length_sampler(config)

    truncator = CipherTruncator(continuous_stream, config.max_length, length_sampler)
    truncated_stream = truncator.process_stream()

    with DatasetWriter(output_dir=output_dir) as writer:
        for truncated in truncated_stream:
            writer.write(truncated)




if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.directory, args.output)
