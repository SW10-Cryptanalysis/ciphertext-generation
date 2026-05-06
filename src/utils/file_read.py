import json
from pathlib import Path
import zipfile
from typing import Any, Iterator

def discover_dataset_files(dataset_dir: Path, extension: str = ".zip") -> list[Path]:
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

    file_paths = list(dataset_dir.rglob(f"*{extension}"))
    if not file_paths:
        raise FileNotFoundError(
            f"No files found with extension {extension} in {dataset_dir}",
        )

    return sorted(file_paths)


def read_jsonl(
    jsonl_filenames: list[str],
    z: zipfile.ZipFile,
) -> Iterator[dict[str, Any]]:
    """Read a jsonl file from inside a zip archive and yield contents as dictionaries.

    Args:
        jsonl_filenames (list[str]): A list of file names to read from.
        z (zipfile.ZipFile): The zip archive to read from.

    Yields:
        dict[str, Any]: A dictionary containing the json data from the file.

    """
    for jsonl_filename in jsonl_filenames:
        with z.open(jsonl_filename, "r") as f:
            for line in f:
                decoded_line = line.decode("utf-8").strip()
                if decoded_line:
                    yield json.loads(decoded_line)


def generate_continuous_stream(file_paths: list[Path]) -> Iterator[dict[str, Any]]:
    """Generate a continuous stream of json data from a list of file paths.

    Args:
        file_paths (list[Path]): A list of file paths to read from.

    Yields:
        dict[str, Any]: A dictionary containing the json data from each file.

    """
    for zip_path in file_paths:
        with zipfile.ZipFile(zip_path, "r") as z:
            # Find all internal files that match your target extension
            jsonl_filenames = [name for name in z.namelist() if name.endswith(".jsonl")]

            yield from read_jsonl(jsonl_filenames, z=z)
