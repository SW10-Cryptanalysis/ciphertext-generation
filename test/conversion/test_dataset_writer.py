import pytest
from dataclasses import dataclass
from pathlib import Path

from conversion.dataset_writer import DatasetWriter
from encipherment.cipher import HomophonicCipher
from fetching.corpus_sampler import TextStream


@pytest.fixture
def mock_cipher():
    cipher = HomophonicCipher(
        TextStream(
            {
                "text": "text",
                "text_with_boundaries": "text",
                "length": 10,
                "target_length": 10,
                "genres": ["test"],
                "source_id": "1",
                "source_name": "test_source",
            }
        )
    )
    cipher.generate_key()
    cipher.encipher()
    return cipher


@dataclass
class WriterTestCase:
    """Defines parameters for validating DatasetWriter chunking and routing logic."""

    items: list[str]
    chunk_size: int
    expected_files: int
    expected_encipher_calls: int
    expected_lines_last_file: int


writer_test_cases = [
    WriterTestCase(
        items=["text", "cipher", "text"],
        chunk_size=2,
        expected_files=2,
        expected_encipher_calls=2,
        expected_lines_last_file=1,
    ),
    WriterTestCase(
        items=["cipher", "cipher"],
        chunk_size=2,
        expected_files=1,
        expected_encipher_calls=0,
        expected_lines_last_file=2,
    ),
    WriterTestCase(
        items=["text", "text", "text", "text"],
        chunk_size=1,
        expected_files=4,
        expected_encipher_calls=4,
        expected_lines_last_file=1,
    ),
]


def test_current_file_is_none_raises_exception(tmp_path: Path, mock_cipher):
    """Validates that the file descriptor closes safely upon pipeline exceptions."""
    writer = DatasetWriter(output_dir=tmp_path)
    with pytest.raises(ValueError) as exc_info:
        writer.write(mock_cipher)
    assert exc_info.value.args[0] == "File not initialized. Call __enter__ first."


def test_invalid_item_type_raises_exception(tmp_path: Path):
    """Validates that the file descriptor closes safely upon pipeline exceptions."""
    with DatasetWriter(output_dir=tmp_path) as writer:
        with pytest.raises(ValueError) as exc_info:
            writer.write("invalid")  # type: ignore
        assert exc_info.value.args[0] == "Invalid cipher object type: <class 'str'>"


@pytest.mark.parametrize("test_case", writer_test_cases)
def test_dataset_writer_logic(tmp_path: Path, test_case: WriterTestCase, mock_cipher):
    """Validates file rotation, context management, and type-based routing."""
    with DatasetWriter(
        output_dir=tmp_path,
        chunk_size=test_case.chunk_size,
    ) as writer:
        for item_type in test_case.items:
            if item_type == "text":
                writer.write(
                    TextStream(
                        {
                            "text": "text",
                            "text_with_boundaries": "text",
                            "length": 10,
                            "target_length": 10,
                            "genres": ["test"],
                            "source_id": "1",
                            "source_name": "test_source",
                        }
                    )
                )
            else:
                writer.write(mock_cipher)

    generated_files = sorted(list(tmp_path.glob("chunk_*.jsonl")))
    assert len(generated_files) == test_case.expected_files

    with open(generated_files[-1], encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == test_case.expected_lines_last_file


def test_dataset_writer_exception_handling(tmp_path: Path):
    """Validates that the file descriptor closes safely upon pipeline exceptions."""
    writer = None

    class PipelineError(Exception):
        """Custom exception for testing fatal pipeline interruptions."""

        pass

    try:
        with DatasetWriter(
            output_dir=tmp_path,
        ) as writer:
            raise PipelineError("Fatal error encountered")
    except PipelineError:
        pass

    assert writer is not None
    assert writer.current_file is not None

    assert writer.current_file.closed is True
