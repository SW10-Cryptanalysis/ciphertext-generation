import random
from typing import Iterator, Iterable, TypedDict
from cipher_generation.config import DatasetConfig
from utils.text_sampling import (
    TextStream,
    Book,
    get_usable_text,
    get_source_genres,
    extract_specific_chunk,
)
from utils.constants import BOOK_IDS_VALIDATION


class Metadata(TypedDict):
    """A typed dictionary for book metadata."""

    source_id: str
    source_name: str
    genres: list[str]


class Target(TypedDict):
    """A typed dictionary for target lengths."""

    split: str
    len: int


class CorpusSampler:
    """Stateful extractor using length quotas and proportional partitioning."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        genre_map: dict[str, list[str]],
        max_chunks_per_book: int = 50,
        buffer_chars: int = 150,
    ) -> None:
        """Initialize the sampler with a dataset configuration and genre map.

        Args:
            dataset_config (DatasetConfig): Dataset configuration with cipher counts.
            genre_map (dict[str, list[str]]): Genre map to use for filtering.
            max_chunks_per_book (int, optional): Maximum number of chunks per book.
                Defaults to 50.
            buffer_chars (int, optional): Buffer size for each chunk. Defaults to 150.

        """
        self.config = dataset_config
        self.genre_map = genre_map
        self.max_chunks_per_book = max_chunks_per_book
        self.buffer = buffer_chars

        self.test_targets = {
            length: len(diffs) * self.config.ciphers_per_bin
            for length, diffs in self.config.test_matrix.items()
        }

        self.targets = {
            "train": self.config.training_num,
            "val": self.config.validation_num,
            "test": self.config.num_test_ciphers,
        }

        self.counts = {
            "train": 0,
            "val": 0,
            "test": {length: 0 for length in self.test_targets},
        }
        self.total_test_count = 0

        self.test_pool = self._build_test_pool()

        self.dist_ranges = [
            self.config.foundation_range,
            self.config.transition_range,
            self.config.frontier_range,
        ]
        self.dist_weights = [
            self.config.foundation_pct,
            self.config.transition_pct,
            self.config.frontier_pct,
        ]

    def _build_test_pool(self) -> list[int]:
        """Create a randomized pool of required lengths for the test split."""
        pool = []
        for length, total_needed in self.test_targets.items():
            pool.extend([length] * total_needed)
        random.shuffle(pool)
        return pool

    def _is_complete(self) -> bool:
        """Check if all dataset quotas (train, val, and test lengths) are satisfied."""
        if self.counts["train"] < self.targets["train"]:
            return False
        if self.counts["val"] < self.targets["val"]:
            return False
        return self.total_test_count >= self.targets["test"]

    def generate_stream(self, stream: Iterable) -> Iterator[tuple[str, TextStream]]:
        """Dispatcher loop that iterates through books until quotas are met.

        Args:
            stream (Iterable): The iterable source of text chunks.

        Yields:
            Iterator[tuple[str, TextStream]]: The generated stream of text chunks.

        """
        for book in stream:
            if self._is_complete():
                break

            if str(book.get("id", "")) in BOOK_IDS_VALIDATION:
                continue

            yield from self._process_book(book)

    def _adjust_quota(self, split: str, target_length: int, amount: int) -> None:
        """Adjust internal tracking counts by the given amount.

        Args:
            split (str): The split to adjust.
            target_length (int): The target length (used for test tracking).
            amount (int): The amount to add or subtract.

        """
        if split == "test":
            self.counts["test"][target_length] += amount
            self.total_test_count += amount
        else:
            self.counts[split] += amount

    def requeue_target(self, split: str, target_length: int) -> None:
        """Handle a failed cipher by returning its quota to the sampler.

        Args:
            split (str): The split the cipher was generated in.
            target_length (int): The length of the cipher.

        """
        self._adjust_quota(split, target_length, amount=-1)
        if split == "test":
            self.test_pool.append(target_length)

    def _reserve_target(self, split: str, target_len: int) -> None:
        """Reserve a target length for the current book.

        Args:
            split (str): The split the cipher was generated in.
            target_len (int): The length of the cipher.

        """
        self._adjust_quota(split, target_len, amount=1)

    def _get_weighted_split(self) -> str | None:
        """Choose split based on remaining quota ratios."""
        rem = {}
        if (t_rem := self.targets["train"] - self.counts["train"]) > 0:
            rem["train"] = t_rem
        if (v_rem := self.targets["val"] - self.counts["val"]) > 0:
            rem["val"] = v_rem
        if (ts_rem := self.targets["test"] - self.total_test_count) > 0:
            rem["test"] = ts_rem

        return (
            random.choices(list(rem.keys()), weights=list(rem.values()), k=1)[0]
            if rem
            else None
        )

    def _get_burst_targets(self) -> list[Target]:
        """Plan a burst of target lengths for the current book."""
        burst_targets: list[Target] = []

        for _ in range(self.max_chunks_per_book):
            split = self._get_weighted_split()
            if not split:
                break

            target_len = self._pop_target_len(split)
            if target_len is not None:
                burst_targets.append({"split": split, "len": target_len})

                self._reserve_target(split, target_len)

        return burst_targets

    def _pop_target_len(self, split: str) -> int | None:
        """Handle the source of the length based on the split type."""
        if split == "test":
            return self.test_pool.pop() if self.test_pool else None

        bounds = random.choices(self.dist_ranges, weights=self.dist_weights, k=1)[0]
        return random.randint(*bounds)

    def _process_book(self, book: Book) -> Iterator[tuple[str, TextStream]]:
        """Extract ciphers from a book using proportional partitioning."""
        usable_text = get_usable_text(book["text"], self.config.frontier_range)
        targets = self._get_burst_targets()

        if not targets:
            return

        valid_targets = self._fit_targets_to_book(targets, len(usable_text))
        if not valid_targets:
            return

        yield from self._extract_chunks_from_partitions(
            usable_text,
            valid_targets,
            book,
        )

    def _fit_targets_to_book(
        self,
        targets: list[Target],
        text_len: int,
    ) -> list[Target]:
        """Trim the target list until it fits within the text length."""
        total_req = sum(t["len"] for t in targets) + (len(targets) * self.buffer)

        while targets and total_req > text_len:
            removed = targets.pop()
            total_req -= removed["len"] + self.buffer
            self.requeue_target(removed["split"], removed["len"])

        if targets and text_len > total_req * 1.5:
            random.shuffle(targets)

        return targets

    def _extract_chunks_from_partitions(
        self,
        text: str,
        targets: list[Target],
        book: Book,
    ) -> Iterator[tuple[str, TextStream]]:
        """Iterate through partitions and yield validated text streams."""
        total_req = sum(t["len"] for t in targets) + (len(targets) * self.buffer)
        scale_factor = max(1.0, len(text) / total_req)
        cursor = 0

        meta: Metadata = {
            "source_id": str(book.get("id", "unknown")),
            "source_name": str(book.get("metadata", {}).get("title", "unknown")),
            "genres": get_source_genres(book, self.genre_map),
        }

        for target in targets:
            p_size = int((target["len"] + self.buffer) * scale_factor)
            result = extract_specific_chunk(text, target["len"], cursor, p_size)

            if result:
                yield self._record_and_format(target, result, meta)
            else:
                self.requeue_target(target["split"], target["len"])

            cursor += p_size

    def _record_and_format(
        self,
        target: Target,
        result: tuple,
        meta: Metadata,
    ) -> tuple[str, TextStream]:
        """Update internal counts and format the final stream object."""
        chunk, bounded = result
        split, target_len = target["split"], target["len"]

        stream_data: TextStream = {
            "text": chunk,
            "text_with_boundaries": bounded,
            "length": len(chunk),
            "target_length": target_len,
            **meta,
        }
        return split, stream_data
