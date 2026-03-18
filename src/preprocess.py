import json
import os
import argparse
import logging
from datasets import Dataset, Features, Value
from typing import Any, Generator
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("preprocess.py")

TEXT_LEN = 10_000
SEQ_LEN = TEXT_LEN * 2 + 3


@dataclass
class Config:
    """Config for arrow dataset creation."""

    # 10k*2 characters + BOS, EOS, SEP
    max_context: int = SEQ_LEN
    unique_homophones: int = 500
    data_dir: Path = Path(__file__).parent.parent.parent / "Ciphers"
    output_dir: Path = Path(__file__).parent.parent.parent / "outputs"
    homophone_file: str = "metadata"
    use_spaces: bool = False

    @property
    def final_output_dir(self) -> Path:
        """Dynamic output dir to either outputs/spaces/ or outputs/normal/."""
        suffix = "spaces" if self.use_spaces else "normal"
        return self.output_dir / suffix

    # TOKEN PROPERTIES
    @property
    def sep_token_id(self) -> int:
        """Seperator token."""
        return self.unique_homophones + 1

    @property
    def space_token_id(self) -> int:
        """Space token."""
        return self.sep_token_id + 1

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token."""
        return self.space_token_id + 1

    @property
    def eos_token_id(self) -> int:
        """End of sequence token."""
        return self.bos_token_id + 1

    @property
    def char_offset(self) -> int:
        """Character offset to avoid clashes with defined tokens."""
        return self.eos_token_id + 1

    @property
    def tokenized_dir(self) -> Path:
        """Dynamic path based on whether we use spaces or not."""
        suffix = "spaced" if self.use_spaces else "normal"
        return self.data_dir / f"tokenized_{suffix}"

    def load_homophones(self) -> None:
        """Load the homophone metadata file and set the unique homophone count."""
        homophone_path = os.path.join(self.data_dir, self.homophone_file)
        if os.path.exists(homophone_path):
            try:
                with open(homophone_path) as f:
                    meta = json.load(f)
                    homophones = int(meta["max_symbol_id"])
                    self.unique_homophones = homophones
            except OSError as e:
                logger.warning(f"Could not read file: {self.homophone_file}")
                logger.warning(f"Using default value: {self.unique_homophones}")
                logger.warning(f"Error: {e}")
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid or missing data in {self.homophone_file}")
                logger.warning(f"Using default value: {self.unique_homophones}")
                logger.warning(f"Error: {e}")


features = Features(
    {
        "ciphertext": Value("string"),
        "plaintext": Value("string"),
        "ciphertext_with_boundaries": Value("string"),
        "plaintext_with_boundaries": Value("string"),
        "difficulty": Value("int32"),
    },
)


class RawToArrowConverter:
    """Encapsulates the tokenization logic for testing and execution."""

    def __init__(self, config: Config) -> None:
        """Initialize the converter with model configuration.

        Args:
                config (Config): The configuration object containing token offsets.

        """
        self.cfg = config
        self.t_key = "plaintext_with_boundaries" if config.use_spaces else "plaintext"
        self.c_key = "ciphertext_with_boundaries" if config.use_spaces else "ciphertext"

    def tokenize_fn(self, example: dict[str, Any]) -> dict[str, Any]:
        """Tokenize a single example from the dataset.

        Args:
                example (Dict[str, Any]): A raw dictionary of text and cipher strings.

        Returns:
                Dict[str, Any]: A dictionary containing 'input_ids' and 'labels'.

        """
        # Cipher mapping (splitting and handling _)
        raw_cipher = example[self.c_key].split()
        cipher_ids = [
            self.cfg.space_token_id if x == "_" else int(x) for x in raw_cipher
        ]

        # Plaintext mapping (char by char)
        plain_ids = []
        for char in example[self.t_key]:
            if char == "_":
                plain_ids.append(self.cfg.space_token_id)
            elif "a" <= char <= "z":
                plain_ids.append(ord(char) - ord("a") + self.cfg.char_offset)

        # BOS, EOS, SEP
        special_tokens = [
            self.cfg.bos_token_id,
            self.cfg.sep_token_id,
            self.cfg.eos_token_id,
        ]
        # max allowed length, but with room for special tokens
        max_content_budget = self.cfg.max_context - len(special_tokens)
        total_content_len = len(cipher_ids) + len(plain_ids)

        if total_content_len > max_content_budget:
            # cut equally if spaces makes the sequence longer than allowed
            half_budget = max_content_budget // 2
            cipher_ids = cipher_ids[:half_budget]
            plain_ids = plain_ids[: (max_content_budget - len(cipher_ids))]

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
            "raw_plaintext": example[self.t_key],
        }


def preprocess_data() -> None:
    """Execute the entry point for preprocessing raw JSON data into Arrow format.

    Parses CLI arguments, loads the configuration, and iterates through
    data splits to save tokenized datasets to disk.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.use_spaces = args.spaces
    cfg.load_homophones()

    # Initialize the converter
    converter = RawToArrowConverter(cfg)

    # Load Raw JSONs
    for split in ["Training", "Test", "Validation"]:
        logger.info("Converting %s (Spaces: %s)...", split, cfg.use_spaces)
        split_path = cfg.data_dir / split
        if not split_path.exists():
            logger.warning(f"Split path {split_path} does not exist. Skipping.")
            continue

        logger.info(f"Processing {split} (Spaces: {cfg.use_spaces})...")

        raw_ds = Dataset.from_generator(
            _json_generator,
            gen_kwargs={"path": split_path},
            features=features,
        )

        if not isinstance(raw_ds, Dataset):
            raise TypeError(
                f"Expected a Dataset from the generator, but got {type(raw_ds)}.",
            )

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

        save_path = cfg.tokenized_dir / split
        tokenized_ds.save_to_disk(str(save_path))
        logger.info("Saved to %s", save_path)


def _json_generator(path: Path) -> Generator[dict[str, Any], None, None]:
    for file_path in path.iterdir():
        if file_path.suffix == ".json":
            with open(file_path, encoding="utf-8") as f:
                yield json.load(f)


if __name__ == "__main__":
    preprocess_data()
