import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from utils.file_read import (
    discover_dataset_files,
    read_jsonl,
    generate_continuous_stream,
)


@dataclass
class DiscoverTestCase:
    """Defines parameters for testing discover_dataset_files."""

    setup_dir: bool
    create_files: list[str]
    extension: str
    expected_error: type[Exception] | None
    expected_matches: list[str]


discover_cases = [
    DiscoverTestCase(
        setup_dir=True,
        create_files=["data1.zip", "data2.zip", "readme.txt"],
        extension=".zip",
        expected_error=None,
        expected_matches=["data1.zip", "data2.zip"],
    ),
    DiscoverTestCase(
        setup_dir=True,
        create_files=["data.jsonl"],
        extension=".jsonl",
        expected_error=None,
        expected_matches=["data.jsonl"],
    ),
    DiscoverTestCase(
        setup_dir=True,
        create_files=["data.txt"],
        extension=".zip",
        expected_error=FileNotFoundError,
        expected_matches=[],
    ),
    DiscoverTestCase(
        setup_dir=False,
        create_files=[],
        extension=".zip",
        expected_error=FileNotFoundError,
        expected_matches=[],
    ),
]


@pytest.mark.parametrize("test_case", discover_cases)
def test_discover_dataset_files(tmp_path: Path, test_case: DiscoverTestCase):
    """Validates dataset discovery, including error states and extension filtering."""
    dataset_dir = tmp_path / "dataset"

    if test_case.setup_dir:
        dataset_dir.mkdir()
        for file_name in test_case.create_files:
            (dataset_dir / file_name).touch()

    if test_case.expected_error:
        with pytest.raises(test_case.expected_error):
            discover_dataset_files(dataset_dir, test_case.extension)
    else:
        result = discover_dataset_files(dataset_dir, test_case.extension)
        expected_paths = sorted(
            [dataset_dir / name for name in test_case.expected_matches]
        )
        assert result == expected_paths


@dataclass
class JsonlTestCase:
    """Defines parameters for testing read_jsonl."""

    zip_contents: dict[str, str]
    read_files: list[str]
    expected_output: list[dict[str, Any]]


jsonl_cases = [
    JsonlTestCase(
        zip_contents={"test.jsonl": '{"a": 1}\n{"b": 2}\n'},
        read_files=["test.jsonl"],
        expected_output=[{"a": 1}, {"b": 2}],
    ),
    JsonlTestCase(
        zip_contents={"test.jsonl": '{"a": 1}\n\n{"b": 2}'},
        read_files=["test.jsonl"],
        expected_output=[{"a": 1}, {"b": 2}],
    ),
    JsonlTestCase(
        zip_contents={"t1.jsonl": '{"a": 1}', "t2.jsonl": '{"b": 2}'},
        read_files=["t1.jsonl", "t2.jsonl"],
        expected_output=[{"a": 1}, {"b": 2}],
    ),
    JsonlTestCase(
        zip_contents={"ignore.txt": "not json"},
        read_files=[],
        expected_output=[],
    ),
]


@pytest.mark.parametrize("test_case", jsonl_cases)
def test_read_jsonl(tmp_path: Path, test_case: JsonlTestCase):
    """Validates reading, skipping blank lines, and parsing JSONL within a zip."""
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file_name, content in test_case.zip_contents.items():
            zf.writestr(file_name, content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        result = list(read_jsonl(test_case.read_files, zf))

    assert result == test_case.expected_output


@dataclass
class StreamTestCase:
    """Defines parameters for testing generate_continuous_stream."""

    zips_to_create: dict[str, dict[str, str]]
    expected_output: list[dict[str, Any]]


stream_cases = [
    StreamTestCase(
        zips_to_create={
            "archive1.zip": {"a.jsonl": '{"x": 1}'},
            "archive2.zip": {"b.jsonl": '{"y": 2}', "c.txt": "ignore me"},
        },
        expected_output=[{"x": 1}, {"y": 2}],
    ),
    StreamTestCase(
        zips_to_create={
            "empty_archive.zip": {"ignore.txt": "nothing to see here"},
        },
        expected_output=[],
    ),
]


@pytest.mark.parametrize("test_case", stream_cases)
def test_generate_continuous_stream(tmp_path: Path, test_case: StreamTestCase):
    """Validates yielding continuous dictionaries from multiple zip archives."""
    zip_paths = []

    for zip_name, contents in test_case.zips_to_create.items():
        zip_path = tmp_path / zip_name
        with zipfile.ZipFile(zip_path, "w") as zf:
            for file_name, content in contents.items():
                zf.writestr(file_name, content)
        zip_paths.append(zip_path)

    result = list(generate_continuous_stream(zip_paths))

    assert result == test_case.expected_output
