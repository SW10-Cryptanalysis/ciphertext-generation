import os
import re
import pytest
import tqdm

from cipher_generation.cipher_manager import CipherManager
from cipher_generation.config import CipherConfig, DatasetConfig
from encipherment.cipher import HomophonicCipher
from cipher_generation.cipher_producer import CipherProducer, ProducerConfig
from cipher_generation.drive_uploader import DriveUploader, DriveUploaderConfig
from fetching.corpus_sampler import CorpusSampler


@pytest.fixture
def mock_pbar(mocker):
    """Provide a standard mocked tqdm context manager."""
    mock_pbar = mocker.MagicMock(spec=tqdm.tqdm)
    mock_pbar.__enter__.return_value = mock_pbar
    return mock_pbar


@pytest.fixture
def mock_cipher(mocker):
    """Provide a standard mocked HomophonicCipher."""
    mock_cipher = mocker.MagicMock(spec=HomophonicCipher)
    mock_cipher.__json__.return_value = {"mock_key": "mock_value"}
    mock_cipher.plaintext = "abc"
    mock_cipher.num_symbols = 3
    mock_cipher.redundancy = 1
    mock_cipher.genres = ["Fiction"]
    return mock_cipher


@pytest.fixture
def tiny_config():
    """Provide a standardized mock CipherConfig for testing."""
    return CipherConfig(
        train_folder="train_folder",
        val_folder="val_folder",
        test_folder="test_folder",
        metadata_folder="metadata_folder",
        batch_size=100,
        dataset_config=DatasetConfig(
            training_num=1,
            validation_num=0,
            test_matrix={},
            ciphers_per_bin=100,
        ),
        num_workers=1,
    )


@pytest.fixture
def default_genre_map():
    return {"1": ["Fiction"], "2": ["Science Fiction"]}


@pytest.fixture
def mocks(mock_pbar, mock_cipher, tiny_config):
    """Provide a standardized set of mocked objects for testing."""
    return (mock_pbar, mock_cipher, tiny_config)


@pytest.fixture
def mock_for_end_to_end_pipeline_sync(mocker, mock_pbar, mock_cipher):
    mock_lock = mocker.MagicMock()
    mock_lock.__enter__.return_value = mock_lock

    mock_tqdm_uploader = mocker.patch(
        "cipher_generation.drive_uploader.tqdm", return_value=mock_pbar
    )
    mock_tqdm_uploader.get_lock.return_value = mock_lock
    mocker.patch("tqdm.tqdm.set_lock")

    mocker.patch("cipher_generation.drive_uploader.authenticate_drive_terminal")
    mock_upload = mocker.patch(
        "cipher_generation.drive_uploader.upload_to_drive", return_value="fake_id"
    )

    mocker.patch(
        "cipher_generation.cipher_producer.HomophonicCipher", return_value=mock_cipher
    )

    return mock_upload


def test_end_to_end_pipeline_sync(
    mocker, tmp_path, tiny_config, mock_for_end_to_end_pipeline_sync
):
    """Verify the pipeline logic by executing methods sequentially in one process."""
    original_join = os.path.join
    mock_upload = mock_for_end_to_end_pipeline_sync

    def mock_join(*args):
        if args[0] == "temp_ciphers":
            path = tmp_path / "temp_ciphers"
            path.mkdir(exist_ok=True)
            return str(path / args[1])
        return original_join(*args)

    mocker.patch("os.path.join", side_effect=mock_join)

    def create_mock_text_stream(raw_text: str) -> dict:
        clean_text = re.sub(r"[^a-z]", "", raw_text.lower())
        return {
            "text": clean_text,
            "text_with_boundaries": raw_text.lower().replace(" ", "_"),
            "source_id": "123",
            "source_name": "Mock Book",
            "length": len(clean_text),
            "genres": ["Fiction"],
        }

    tiny_stream = [("train", create_mock_text_stream("First text"))]
    mock_sampler = mocker.Mock(spec=CorpusSampler)

    manager = CipherManager(
        config=tiny_config,
        text_stream_source=tiny_stream,
        sampler=mock_sampler,
    )

    mocker.patch.object(manager, "_process_feedback", return_value=(1, 0))

    manager._feeder_stream(mocker.Mock())
    manager.job_queue.put("STOP")

    config = ProducerConfig(
        input_queue=manager.job_queue,
        output_queue=manager.result_queue,
        stats_queue=manager.stats_queue,
        feedback_queue=manager.feedback_queue,
        batch_size=100,
        temp_dir=tmp_path / "temp_ciphers",
    )

    worker = CipherProducer(
        config=config,
        name="TestWorker",
    )
    worker.run()

    manager._upload_metadata()
    manager.result_queue.put("STOP")

    uploader_config = DriveUploaderConfig(
        split_folders=manager.split_folders,
        total_ciphers=manager.total_count,
        tqdm_lock=mocker.Mock(),
    )
    uploader = DriveUploader(upload_queue=manager.result_queue, config=uploader_config)
    uploader.run()

    assert mock_upload.call_count == 2


@pytest.mark.integration
def test_requeue_integration_real_stream(mocker, tmp_path, default_genre_map):
    """True multiprocess integration test enforcing the requeue math on test splits."""
    integration_config = CipherConfig(
        train_folder="train_folder",
        val_folder="val_folder",
        test_folder="test_folder",
        metadata_folder="metadata_folder",
        batch_size=1,
        num_workers=1,
        dataset_config=DatasetConfig(
            training_num=0,
            validation_num=0,
            test_matrix={350: [25]},
            ciphers_per_bin=1,
        ),
    )

    def intelligent_mock_stream():
        # First Book: 26 unique chars. 350 chunk max redundancy = 350//26 = 13.
        # Target 25 will cap at 13, fail strict validation, and trigger a Requeue.
        yield {
            "id": "1",
            "text": "abcdefghijklmnopqrstuvwxyz" * 20,
            "metadata": {"title": "Normal Book"},
        }
        # Second Book: 4 unique chars. 350 chunk max redundancy = 350//4 = 87.
        # Will easily hit 25 and succeed.
        yield {"id": "2", "text": "abcd" * 125, "metadata": {"title": "Skewed Book"}}

    sampler = CorpusSampler(integration_config.dataset_config, default_genre_map)
    text_stream = sampler.generate_stream(intelligent_mock_stream())

    manager = CipherManager(
        config=integration_config,
        text_stream_source=text_stream,
        sampler=sampler,
    )
    manager.temp_dir = tmp_path / "temp_ciphers"

    spy_requeue = mocker.spy(sampler, "requeue_target")
    mocker.patch("cipher_generation.drive_uploader.DriveUploader.start")
    mocker.patch("cipher_generation.drive_uploader.DriveUploader.join")

    manager.execute()

    # Verify the failure was caught and a replacement text was drawn
    assert spy_requeue.call_count == 1
    spy_requeue.assert_called_with("test", 350)
    assert manager.master_stats.splits["test"].total_count == 1  # type: ignore


@pytest.mark.integration
def test_train_continuous_redundancy_integration(mocker, tmp_path, default_genre_map):
    """Verify that train tasks accept clamped random redundancy without requeuing."""
    integration_config = CipherConfig(
        train_folder="train_folder",
        val_folder="val_folder",
        test_folder="test_folder",
        metadata_folder="metadata_folder",
        batch_size=1,
        num_workers=1,
        dataset_config=DatasetConfig(
            training_num=1,
            validation_num=0,
            test_matrix={},
            ciphers_per_bin=0,
        ),
    )

    def train_mock_stream():
        # Yield a normal string that forces the cipher to clamp its random choice.
        # Added a space and multiplied by 500 so extract_specific_chunk has valid word boundaries and length!
        yield {
            "id": "1",
            "text": "abcdefghijklmnopqrstuvwxyz " * 500,
            "metadata": {"title": "Normal Book"},
        }

    sampler = CorpusSampler(integration_config.dataset_config, default_genre_map)
    text_stream = sampler.generate_stream(train_mock_stream())

    manager = CipherManager(
        config=integration_config,
        text_stream_source=text_stream,
        sampler=sampler,
    )
    manager.temp_dir = tmp_path / "temp_ciphers"

    spy_requeue = mocker.spy(sampler, "requeue_target")
    mocker.patch("cipher_generation.drive_uploader.DriveUploader.start")
    mocker.patch("cipher_generation.drive_uploader.DriveUploader.join")

    manager.execute()

    # Verify the train cipher succeeded purely on the first try despite clamping
    assert spy_requeue.call_count == 0
    assert manager.master_stats.splits["train"].total_count == 1  # type: ignore
