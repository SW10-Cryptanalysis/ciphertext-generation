import orjson
import json
import zipfile
import os
import argparse
import logging
from datasets import Dataset, Features, Value
from typing import Any, Generator
from pathlib import Path
from dataclasses import dataclass


os.environ["HF_DATASETS_CACHE"] = (
    "/ceph/project/SW10-CausalLM/ciphertext-generation/hf_cache"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("preprocess.py")


@dataclass
class Config:
    """Config for arrow dataset creation."""

    task: str = "causal"
    use_spaces: bool = False
    unique_homophones: int = 0
    data_dir: Path = Path(__file__).parent.parent.parent / "Ciphers"
    homophone_file: str = "metadata.json"

    @property
    def sep_token_id(self) -> int:
        """Get the separator token ID."""
        return self.unique_homophones + 1

    @property
    def space_token_id(self) -> int:
        """Get the space token ID."""
        return self.sep_token_id + 1

    @property
    def bos_token_id(self) -> int:
        """Get the beginning-of-sequence token ID."""
        return self.space_token_id + 1

    @property
    def eos_token_id(self) -> int:
        """Get the end-of-sequence token ID."""
        return self.bos_token_id + 1

    @property
    def char_offset(self) -> int:
        """Get the character offset for the tokenization scheme."""
        return self.eos_token_id + 1

    @property
    def tokenized_dir(self) -> Path:
        """Dynamic path based on spacing AND task to prevent overwriting."""
        suffix = "spaced" if self.use_spaces else "normal"
        if self.task == "mapping":
            suffix += "_mapping"
        return self.data_dir / f"tokenized_{suffix}"

    def load_homophones(self) -> None:
        """Load the homophone metadata file."""
        homophone_path = self.data_dir / self.homophone_file
        if not homophone_path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {homophone_path}")
        try:
            with open(homophone_path) as f:
                meta = json.load(f)
                self.unique_homophones = int(meta["max_symbol_id"])
        except Exception as e:
            raise ValueError(f"Failed loading {homophone_path}") from e


features = Features(
    {
        "ciphertext": Value("string"),
        "plaintext": Value("string"),
        "ciphertext_with_boundaries": Value("string"),
        "plaintext_with_boundaries": Value("string"),
        "redundancy": Value("int32"),
    },
)


class RawToArrowConverter:
    """Encapsulates the tokenization logic for both tasks."""

    def __init__(self, config: Config) -> None:
        """Initialize the RawToArrowConverter.

        Args:
            config (Config): The configuration object.

        """
        self.cfg = config
        self.t_key = "plaintext_with_boundaries" if config.use_spaces else "plaintext"
        self.c_key = "ciphertext_with_boundaries" if config.use_spaces else "ciphertext"

    def _tokenize_causal(self, cipher_ids: list[int], raw_plain: str) -> dict[str, Any]:
        """Tokenize items for the causal task.

        Args:
            cipher_ids (list[int]): The list of cipher IDs.
            raw_plain (str): The plaintext to tokenize.

        Returns:
            dict[str, Any]: The tokenized data.

        """
        plain_ids = []
        for char in raw_plain:
            if char == "_":
                plain_ids.append(self.cfg.space_token_id)
            elif "a" <= char <= "z":
                plain_ids.append(ord(char) - ord("a") + self.cfg.char_offset)

        input_ids = (
            [self.cfg.bos_token_id]
            + cipher_ids
            + [self.cfg.sep_token_id]
            + plain_ids
            + [self.cfg.eos_token_id]
        )
        labels = list(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "raw_plaintext": raw_plain,
            "plain_length": len(plain_ids),
        }

    def _tokenize_mapping(
        self,
        cipher_ids: list[int],
        raw_plain: str,
    ) -> dict[str, Any]:
        """Tokenize items for the mapping classification task.

        Args:
            cipher_ids (list[int]): The list of cipher IDs.
            raw_plain (str): The plaintext to tokenize.

        Returns:
            dict[str, Any]: The tokenized data.

        """
        input_ids = [self.cfg.bos_token_id] + cipher_ids + [self.cfg.eos_token_id]

        clean_plain = raw_plain.replace(" ", "")

        mapping_dict = {
            cid: ord(pchar) - ord("a")
            for cid, pchar in zip(cipher_ids, clean_plain, strict=True)
            if "a" <= pchar <= "z"
        }

        unique_symbols = sorted(set(sym for sym in input_ids if sym > 0))

        labels = [mapping_dict.get(sym, -100) for sym in unique_symbols]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "raw_plaintext": raw_plain,
            "plain_length": 0,
        }

    def tokenize_fn(self, example: dict[str, Any]) -> dict[str, Any]:
        """Routes tokenization based on the task."""
        raw_cipher = example[self.c_key].split()
        cipher_ids = [
            self.cfg.space_token_id if x == "_" else int(x) for x in raw_cipher
        ]
        raw_plain = example[self.t_key]

        if self.cfg.task == "causal":
            return self._tokenize_causal(cipher_ids, raw_plain)

        return self._tokenize_mapping(cipher_ids, raw_plain)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    parser.add_argument(
        "--task",
        choices=["causal", "mapping"],
        default="causal",
        help="Format data for sequence generation (causal) or token classification"
        " (mapping).",
    )
    return parser.parse_args()


def process_split(split_name: str, cfg: Config, converter: RawToArrowConverter) -> int:
    """Process a single dataset split, saves it to disk, and returns the max length.

    Args:
        split_name (str): The name of the split (e.g., "Training", "Test").
        cfg (Config): The preprocessing configuration.
        converter (RawToArrowConverter): The initialized tokenization converter.

    Returns:
        int: The maximum sequence length observed in this split.

    """
    split_path = cfg.data_dir / split_name
    if not split_path.exists():
        logger.warning("Split path %s does not exist. Skipping.", split_path)
        return 0

    logger.info("Processing %s...", split_name)

    raw_ds = Dataset.from_generator(
        _json_generator,
        gen_kwargs={"path": split_path},
        features=features,
    )

    if not isinstance(raw_ds, Dataset):
        raise TypeError(f"Expected a Dataset, but got {type(raw_ds)}")

    tokenized_ds = raw_ds.map(
        converter.tokenize_fn,
        num_proc=8,
        remove_columns=[
            "ciphertext",
            "ciphertext_with_boundaries",
            "plaintext",
            "plaintext_with_boundaries",
        ],
    )

    if cfg.task == "causal":
        current_max_len = max(tokenized_ds["plain_length"])
    else:
        current_max_len = max(len(x) for x in tokenized_ds["input_ids"])

    tokenized_ds = tokenized_ds.remove_columns(["plain_length"])

    save_path = cfg.tokenized_dir / split_name
    tokenized_ds.save_to_disk(str(save_path))
    logger.info("Saved to %s", save_path)

    return current_max_len


def main() -> None:
    """Entry point for preprocessing."""
    args = parse_args()

    cfg = Config(use_spaces=args.spaces, task=args.task)
    cfg.load_homophones()

    if cfg.unique_homophones == 0:
        raise ValueError("unique_homophones has not been set.")

    logger.info(f"Task              : {cfg.task.upper()}")
    logger.info(f"unique_homophones : {cfg.unique_homophones}")
    logger.info(f"sep_token_id      : {cfg.sep_token_id}")
    logger.info(f"space_token_id    : {cfg.space_token_id}")
    logger.info(f"bos_token_id      : {cfg.bos_token_id}")
    logger.info(f"eos_token_id      : {cfg.eos_token_id}")
    logger.info(f"char_offset       : {cfg.char_offset}")
    logger.info(f"tokenized_dir     : {cfg.tokenized_dir}")

    converter = RawToArrowConverter(cfg)
    global_max_len = 0

    for split in ["Training", "Test", "Validation"]:
        current_max_len = process_split(split, cfg, converter)
        global_max_len = max(global_max_len, current_max_len)

    logger.info("Max sequence length observed: %d", global_max_len)


def _yield_from_zip(zip_path: Path) -> Generator[dict[str, Any], None, None]:
    """Extract and yield JSON records from all .jsonl files in a single zip archive."""
    with zipfile.ZipFile(zip_path, "r") as z:
        jsonl_files = [f for f in z.namelist() if f.endswith(".jsonl")]

        for filename in jsonl_files:
            with z.open(filename) as f:
                for line in f:
                    if line.strip():
                        yield orjson.loads(line)


def _json_generator(path: Path) -> Generator[dict[str, Any], None, None]:
    """Yield JSON records from all .zip archives in the specified directory."""
    zip_files = list(path.glob("*.zip"))

    if not zip_files:
        raise FileNotFoundError(f"No .zip files found in: {path}")

    for zip_path in zip_files:
        yield from _yield_from_zip(zip_path)


if __name__ == "__main__":
    main()
