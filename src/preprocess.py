import orjson
import json
import zipfile
import os


# Force Hugging Face cache to a visible directory to bypass Slurm/HPC issues
os.environ["HF_DATASETS_CACHE"] = (
    "/ceph/project/SW10-CausalLM/ciphertext-generation/hf_cache"
)

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


@dataclass
class Config:
    """Config for arrow dataset creation."""

    unique_homophones: int = 2503
    data_dir: Path = Path(__file__).parent.parent.parent / "Ciphers"
    output_dir: Path = Path(__file__).parent.parent.parent / "outputs"
    homophone_file: str = "metadata.json"
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
        if not os.path.exists(homophone_path):
            raise FileNotFoundError(
                f"Metadata file not found at: {homophone_path}. "
                "Cannot determine unique_homophones — aborting.",
            )
        try:
            with open(homophone_path) as f:
                meta = json.load(f)
                self.unique_homophones = int(meta["max_symbol_id"])
        except OSError as e:
            raise OSError(f"Could not read file: {homophone_path}") from e
        except (ValueError, KeyError) as e:
            raise ValueError(
                f"Invalid or missing 'max_symbol_id' in {homophone_path}",
            ) from e


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
            "plain_length": len(plain_ids),
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

    logger.info(f"unique_homophones : {cfg.unique_homophones}")
    logger.info(f"sep_token_id      : {cfg.sep_token_id}")
    logger.info(f"space_token_id    : {cfg.space_token_id}")
    logger.info(f"bos_token_id      : {cfg.bos_token_id}")
    logger.info(f"eos_token_id      : {cfg.eos_token_id}")
    logger.info(f"char_offset       : {cfg.char_offset}")
    logger.info(f"tokenized_dir     : {cfg.tokenized_dir}")

    # Initialize the converter
    converter = RawToArrowConverter(cfg)

    global_max_len = 0

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

        current_max_len = max(tokenized_ds["plain_length"])
        if current_max_len > global_max_len:
            global_max_len = current_max_len

        tokenized_ds = tokenized_ds.remove_columns(["plain_length"])

        save_path = cfg.tokenized_dir / split
        tokenized_ds.save_to_disk(str(save_path))
        logger.info("Saved to %s", save_path)

    logger.info(f"Max length {global_max_len}!!!")


def _json_generator(path: Path) -> Generator[dict[str, Any], None, None]:
    """Yield JSON records from .zip archives with .jsonl files."""
    zip_read = False

    for file_path in path.iterdir():
        if file_path.suffix == ".zip":
            zip_read = True
            with zipfile.ZipFile(file_path, "r") as z:
                for filename in z.namelist():
                    if filename.endswith(".jsonl"):
                        with z.open(filename) as f:
                            for line in f:
                                if line.strip():
                                    yield orjson.loads(line)

    if not zip_read:
        raise FileNotFoundError(f"No .zip files were found in dir: {path}!")


if __name__ == "__main__":
    preprocess_data()
