import os
from utils.logging import get_logger
from cipher_generation.cipher_manager import CipherManager
from cipher_generation.config import DatasetConfig, CipherConfig
from typing import Iterator

from fetching.dataset_extractor import DatasetExtractor
from fetching.corpus_sampler import CorpusSampler
from utils.genres import load_existing_genre_map
from utils.text_sampling import randomize_stream, TextStream
from utils.constants import (
    NUM_TRAINING_CIPHERS,
    NUM_VALIDATION_CIPHERS,
    GENRE_MAP_PATH,
    DATASETS,
)


def get_text_stream(
    config: DatasetConfig,
    extractor: DatasetExtractor | None = None,
) -> tuple[CorpusSampler, Iterator[tuple[str, TextStream]]]:
    """Initialize dependencies and return the randomized text stream.

    Args:
        config (DatasetConfig): The dataset configuration.
        extractor (DatasetExtractor | None, optional): The dataset extractor instance.

    Returns:
        Iterator[tuple[str, TextStream]]: The generated stream of text chunks.

    """
    if extractor is None:
        extractor = DatasetExtractor(DATASETS)

    genre_map = load_existing_genre_map(GENRE_MAP_PATH, None)
    full_stream = randomize_stream(extractor.get_full_stream())

    sampler = CorpusSampler(config, genre_map)

    return sampler, sampler.generate_stream(full_stream)


def get_folder_id(env_var: str) -> str:
    """Get the folder ID from an environment variable."""
    folder_id = os.getenv(env_var)
    if not folder_id:
        raise OSError(
            f"Environment variable {env_var} not set. Please set it before running.",
        )
    return folder_id


def load_folder_ids() -> tuple[str, str, str, str]:
    """Load the folder IDs for train, val, and test splits."""
    folder_id_train = get_folder_id("FOLDER_ID_TRAIN")
    folder_id_val = get_folder_id("FOLDER_ID_VAL")
    folder_id_test = get_folder_id("FOLDER_ID_TEST")
    folder_id_metadata = get_folder_id("FOLDER_ID_METADATA")

    return folder_id_train, folder_id_val, folder_id_test, folder_id_metadata


if __name__ == "__main__":
    if not os.path.exists(GENRE_MAP_PATH):
        raise FileNotFoundError(f"Genre map not found at {GENRE_MAP_PATH}")

    log = get_logger("TrainingGeneration")
    folder_id_train, folder_id_val, folder_id_test, folder_id_metadata = (
        load_folder_ids()
    )

    dataset_config = DatasetConfig(
        training_num=NUM_TRAINING_CIPHERS,
        validation_num=NUM_VALIDATION_CIPHERS,
    )

    sampler, text_stream = get_text_stream(dataset_config)

    config = CipherConfig(
        train_folder=folder_id_train,
        val_folder=folder_id_val,
        test_folder=folder_id_test,
        metadata_folder=folder_id_metadata,
        batch_size=10_000,
        dataset_config=dataset_config,
    )

    manager = CipherManager(
        config=config,
        text_stream_source=text_stream,
        sampler=sampler,
    )

    try:
        manager.execute()
    except Exception as e:
        log.critical(
            f"A critical error occurred in the main process: {e}",
            exc_info=True,
        )
