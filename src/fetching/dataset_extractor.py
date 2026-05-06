from datasets import load_dataset, interleave_datasets
import os
import dotenv
import logging
from datasets import IterableDataset
from typing import Iterator, TypedDict
import re


dotenv.load_dotenv()


class DatasetConfig(TypedDict):
    """A typed dictionary for dataset configurations.

    Attributes:
        path (str): The path to the dataset.
        type (str): The type of dataset.
        split_name (str): The name of the split to extract.
        column (str): The column to extract.
        prefix (str): The prefix to add to the ids.
        fallback_genres (list[str]): The list of genres to use if the column is missing.

    """

    path: str
    type: str
    split_name: str
    column: str
    prefix: str
    fallback_genres: list[str]


class DatasetExtractor:
    """A class for extracting dataset from Hugging Face."""

    def __init__(
        self,
        datasets: list[DatasetConfig],
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the DatasetExtractor with the dataset name and split.

        The config array should be a list of dictionaries with the following keys:
        ```python
        {
            "path": str,
            "type": str,
            "split_name": str,
            "column": str,
            "prefix": str,
            "fallback_genres": list[str],
        }
        ```

        Args:
            datasets (list[DatasetConfig]): The list of datasets to extract with their
                configurations.
            logger (logging.Logger | None, optional): Logger to use. Defaults to None.

        """
        self.configs = datasets
        self.token = os.environ.get("HF_TOKEN")

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("DatasetExtractor")
            self.logger.addHandler(logging.NullHandler())
            self.logger.setLevel(logging.CRITICAL)

    @staticmethod
    def _extract_title_from_text(text: str) -> str:
        """Extract a prospective title from the beginning of a raw document.

        Targets Markdown headers or the first distinct text block, normalizing
        newlines and stripping excess whitespace or appended URLs.

        Args:
            text (str): The raw text to extract the title from.

        Returns:
            str: The extracted title.

        """
        if not text or not isinstance(text, str):
            return "unknown"

        text = text.lstrip()

        paragraphs = re.split(r"\n\s*\n", text)
        first_block = paragraphs[0] if paragraphs else text

        title_match = re.match(r"^#+\s*(.*)", first_block, flags=re.DOTALL)
        raw_title = title_match.group(1) if title_match else first_block

        clean_title = re.sub(r"\s+", " ", raw_title).strip()

        url_index = clean_title.find("http")
        if url_index != -1:
            clean_title = clean_title[:url_index].strip()

        return clean_title or "unknown"

    @staticmethod
    def _normalize_record(x: dict, p: str, t: str, fg: list[str]) -> dict:
        """Normalize individual records and safely map metadata."""
        metadata = x.get("metadata", {})
        source_name = "unknown"
        if isinstance(metadata, dict):
            source_name = metadata.get("title", "unknown")

        if source_name == "unknown":
            source_name = DatasetExtractor._extract_title_from_text(x["text"])

        return {
            "id": f"{p}:{str(x.get('id', 'unknown'))}",
            "text": x["text"],
            "source_name": source_name,
            "source_type": t,
            "fallback_genres": fg,
        }

    def get_full_stream(self) -> "IterableDataset":
        """Get the full Hugging Face stream.

        Returns:
            IterableDataset: The full Hugging Face stream.

        """
        if self.logger:
            self.logger.info("Initializing full Hugging Face stream...")

        streams = []

        for config in self.configs:
            ds = load_dataset(
                config["path"],
                split=config.get("split_name", "train"),
                streaming=True,
                token=self.token,
            )

            if config["column"] != "text":
                ds = ds.rename_column(config["column"], "text")

            prefix = config.get("prefix", "unknown")
            source_type = config.get("type", "unknown")
            fallback_genres = config.get("fallback_genres", ["Other / Uncategorized"])

            ds = ds.map(
                DatasetExtractor._normalize_record,
                fn_kwargs={"p": prefix, "t": source_type, "fg": fallback_genres},
            )

            columns = [
                "id",
                "text",
                "source_name",
                "source_type",
                "fallback_genres",
            ]

            ds = ds.select_columns(columns)

            streams.append(ds)

        return interleave_datasets(
            streams,
            seed=42,
            stopping_strategy="all_exhausted_without_replacement",
        )

    def get_pg_id_stream(self) -> Iterator[str]:
        """Extract book IDs from the Project Gutenberg dataset.

        Returns:
            Iterator[str]: An iterator of book IDs.

        """
        dataset_config = next(
            (
                config
                for config in self.configs
                if config["type"] == "project_gutenberg"
            ),
            None,
        )

        if not dataset_config:
            raise ValueError("No Project Gutenberg dataset found in config.")

        stream = load_dataset(
            dataset_config["path"],
            split=dataset_config.get("split_name", "train"),
            streaming=True,
            token=self.token,
        ).select_columns(["id"])

        for item in stream:
            yield str(item["id"])
