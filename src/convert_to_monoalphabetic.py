import argparse
from pathlib import Path
from tqdm import tqdm

from conversion.monoalphabetic_converter import MonoalphabeticConverter
from conversion.dataset_writer import DatasetWriter
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
        help="The output directory for the monoalphabetic dataset.",
    )
    return parser.parse_args()


def run_pipeline(dataset_dir: Path, output_dir: Path) -> None:
    """Run the monoalphabetic conversion pipeline with an approximated progress bar."""
    file_paths = discover_dataset_files(dataset_dir)
    continuous_stream = generate_continuous_stream(file_paths)

    converter = MonoalphabeticConverter(continuous_stream)
    converted_stream = converter.process_stream()

    approximate_total = len(file_paths) * 10000

    with DatasetWriter(output_dir=output_dir) as writer:
        for cipher in tqdm(
            converted_stream,
            total=approximate_total,
            desc="Converting to Monoalphabetic",
            unit="seq",
            smoothing=0.1,
        ):
            writer.write(cipher)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.directory, args.output)
