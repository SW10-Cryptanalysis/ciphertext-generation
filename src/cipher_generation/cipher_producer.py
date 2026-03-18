import multiprocessing as mp
import os
import queue
import json
import zipfile
from typing import Any, TypedDict, Literal
from dataclasses import dataclass
from multiprocessing.queues import Queue as MPQueue

from utils.logging import get_logger_tqdm
from encipherment.cipher import (
    SubstitutionCipher,
    HomophonicCipher,
    MonoalphabeticCipher,
)
from utils.text_sampling import TextStream
from dataset_stats import DatasetStatsAggregator
from pathlib import Path
from cipher_generation.task import CipherTask, UploadTask


log = get_logger_tqdm("CipherProducer", 20)


class FileInfo(TypedDict):
    """A typed dictionary for tracking open file handles and paths."""

    handle: Any
    filepath: Path


@dataclass
class ProducerConfig:
    """A dataclass to store the configuration for the CipherProducer."""

    batch_size: int
    input_queue: MPQueue[CipherTask | Literal["STOP"]]
    output_queue: MPQueue[UploadTask | Literal["STOP"]]
    stats_queue: MPQueue[DatasetStatsAggregator | Literal["STOP"]]
    feedback_queue: MPQueue[dict]
    temp_dir: Path


@dataclass
class BatchInfo:
    """A dataclass to store the current batch information."""

    batch_indices: dict[str, int]
    files: dict[str, FileInfo]
    current_counts: dict[str, int]


class CipherProducer(mp.Process):
    """A multiprocessing process for generating and streaming ciphers to disk."""

    def __init__(
        self,
        config: ProducerConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the CipherProducer.

        Args:
            config (ProducerConfig): The configuration object.
                Contains the queues and batch size to use.
            *args: Additional arguments to pass to the parent class.
            **kwargs: Additional keyword arguments to pass to the parent class.

        """
        super().__init__(*args, **kwargs)
        self.input_queue = config.input_queue
        self.output_queue = config.output_queue
        self.stats_queue = config.stats_queue
        self.feedback_queue = config.feedback_queue

        self.batch_size = config.batch_size
        self.temp_dir = config.temp_dir

    def run(self) -> None:
        """Execute the cipher generation process loop."""
        stats = DatasetStatsAggregator()
        process_name = self.name

        os.makedirs(self.temp_dir, exist_ok=True)

        batch_indices: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        current_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        files = dict[str, FileInfo]()

        batch_info = BatchInfo(batch_indices, files, current_counts)

        log.info(f"{process_name} started.")

        while True:
            try:
                item = self.input_queue.get(timeout=5)

                if item == "STOP":
                    self._cleanup_all(
                        batch_info,
                        stats,
                    )
                    log.info(f"{process_name} received STOP signal.")
                    break

                cipher = self.generate_cipher(item.text_data, item.target_redundancy)

                if cipher is None:
                    self.feedback_queue.put(
                        {
                            "status": "fail",
                            "split": item.split,
                            "target_length": item.text_data.get("target_length", 0),
                            "target_redundancy": item.target_redundancy,
                        },
                    )
                    continue

                if not self._check_redundancy(cipher, item):
                    self.feedback_queue.put(
                        {
                            "status": "fail",
                            "split": item.split,
                            "target_length": item.text_data.get("target_length", 0),
                            "target_redundancy": item.target_redundancy,
                        },
                    )
                    continue

                stats.record(
                    split=item.split,
                    length=len(cipher.plaintext),
                    homophones=cipher.num_symbols,
                    redundancy=cipher.redundancy or 0,
                    genres=cipher.genres,
                )

                self._write_and_batch_cipher(item.split, cipher, batch_info)

                self.feedback_queue.put({"status": "success"})

            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Producer {process_name} failed: {e}")

        log.info(f"{process_name} finished generation.")

    def _check_redundancy(self, cipher: SubstitutionCipher, item: CipherTask) -> bool:
        """Check if the cipher has the correct redundancy.

        Args:
            cipher (SubstitutionCipher): The cipher to check.
            item (CipherTask): The cipher task to check.

        Returns:
            bool: True if the cipher has the correct redundancy, False otherwise.

        """
        if item.split != "test":
            return True

        return (
            cipher.redundancy is not None
            and cipher.redundancy == item.target_redundancy
        )

    def _write_and_batch_cipher(
        self,
        split: str,
        cipher: SubstitutionCipher,
        batch_info: BatchInfo,
    ) -> None:
        """Write the cipher to disk and zip the batch if the limit is reached."""
        if split not in batch_info.files:
            self._open_new_batch_file(split, batch_info)

        cipher_str = json.dumps(cipher.__json__())
        batch_info.files[split]["handle"].write(cipher_str + "\n")
        batch_info.current_counts[split] += 1

        if batch_info.current_counts[split] >= self.batch_size and split == "train":
            self._close_and_zip_batch(
                split,
                batch_info=batch_info,
            )
            batch_info.current_counts[split] = 0
            batch_info.batch_indices[split] += 1
            del batch_info.files[split]

    def _open_new_batch_file(
        self,
        split: str,
        batch_info: BatchInfo,
    ) -> None:
        """Open a new raw JSONL file for streaming."""
        filename = (
            f"raw_batch_{split}_{os.getpid()}_{batch_info.batch_indices[split]}.jsonl"
        )
        filepath = self.temp_dir / filename

        batch_info.files[split] = {
            "handle": open(filepath, "w", encoding="utf-8"),  # noqa: SIM115
            "filepath": filepath,
        }

    def _close_and_zip_batch(
        self,
        split: str,
        batch_info: BatchInfo,
    ) -> None:
        """Close the current JSONL file, zip it, queue it, and delete the raw file."""
        batch_info.files[split]["handle"].close()

        filepath = batch_info.files[split]["filepath"]

        archive_name = Path(
            f"batch_{split}_{os.getpid()}_{batch_info.batch_indices[split]}.zip",
        )
        zip_filepath = self.temp_dir / archive_name

        with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(filepath, arcname=filepath.name)

        os.remove(filepath)

        task = UploadTask(
            filepath=zip_filepath,
            filename=archive_name,
            cipher_count=batch_info.current_counts[split],
            split=split,
        )
        self.output_queue.put(task)

    def _cleanup_all(
        self,
        batch_info: BatchInfo,
        stats: DatasetStatsAggregator,
    ) -> None:
        """Flush any open files, zip them, and send final stats."""
        for split in list(batch_info.files.keys()):
            if batch_info.current_counts[split] > 0:
                if split in ["val", "test"]:
                    batch_info.files[split]["handle"].close()
                    filename = Path(batch_info.files[split]["filepath"].name)
                    task = UploadTask(
                        filepath=batch_info.files[split]["filepath"],
                        filename=filename,
                        cipher_count=batch_info.current_counts[split],
                        split=split,
                    )
                    self.output_queue.put(task)
                else:
                    self._close_and_zip_batch(
                        split,
                        batch_info,
                    )

        self.stats_queue.put(stats)

    def generate_cipher(
        self,
        text: TextStream,
        target_redundancy: int | None,
    ) -> SubstitutionCipher | None:
        """Generate a cipher from the provided text string, applying redundancy limits.

        Args:
            text (TextStream): The text stream to generate the cipher from.
            target_redundancy (float | int | None): The specific redundancy to hit,
                0 for monoalphabetic, or None to generate a random continuous
                redundancy.

        Returns:
            SubstitutionCipher | None: The generated cipher, or None if an error
                occurred.

        """
        try:
            if target_redundancy == 0:
                cipher = MonoalphabeticCipher(text)

            elif target_redundancy is None:
                cipher = HomophonicCipher(text)

            else:
                cipher = HomophonicCipher(text, redundancy=target_redundancy)

            cipher.generate_key()
            cipher.encipher()
            return cipher

        except ValueError as e:
            log.error(f"Error generating cipher: {e}")
            return None
        except Exception as e:
            log.error(f"Unexpected cipher generation error: {e}")
            return None
