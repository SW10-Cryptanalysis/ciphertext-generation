from pathlib import Path
import argparse
from conversion.cipher_truncator import CipherTruncator
from conversion.truncation_config import TruncationConfig, create_length_sampler
from conversion.dataset_writer import DatasetWriter
from tqdm import tqdm
from utils.file_read import discover_dataset_files, generate_continuous_stream


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

    approximate_total = len(file_paths) * 10000

    with DatasetWriter(output_dir=output_dir) as writer:
        for truncated in tqdm(
            truncated_stream,
            total=approximate_total,
            desc="Truncating and routing ciphers",
            unit="seq",
            smoothing=0.1,
        ):
            writer.write(truncated)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.directory, args.output)
