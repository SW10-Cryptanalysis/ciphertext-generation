import pytest
import json
import os
import queue
from dataclasses import dataclass
from pathlib import Path

from cipher_generation.config import CipherConfig, DatasetConfig
from cipher_generation.cipher_manager import CipherManager
from cipher_generation.task import UploadTask, CipherTask
from dataset_stats import DatasetStatsAggregator


@pytest.fixture
def manager_config():
    """Provide a standard configuration dictionary for manager initialization."""
    dataset_config = DatasetConfig(
        training_num=100,
        validation_num=10,
        test_matrix={
            4000: [5, 10, 15, 20, 25, 30, 50, 100, 200, 300, 0],
            5000: [5, 10, 15, 20, 25, 30, 50, 100, 200, 300, 0],
            6500: [5, 10, 15, 20, 25, 30, 50, 100, 200, 300, 0],
            8000: [5, 10, 15, 20, 25, 30, 50, 100, 200, 300, 0],
        },
        ciphers_per_bin=100,
    )
    return CipherConfig(
        train_folder="train_folder",
        val_folder="val_folder",
        test_folder="test_folder",
        metadata_folder="metadata_folder",
        batch_size=10,
        dataset_config=dataset_config,
    )


@pytest.fixture
def manager_config_no_test():
    """Provide a minimal configuration dictionary for fast stream testing."""
    dataset_config = DatasetConfig(
        training_num=3,
        validation_num=0,
        test_matrix={},
        ciphers_per_bin=100,
    )
    return CipherConfig(
        train_folder="train_folder",
        val_folder="val_folder",
        test_folder="test_folder",
        metadata_folder="metadata_folder",
        num_workers=4,
        batch_size=10,
        dataset_config=dataset_config,
    )


@pytest.fixture
def mock_sampler(mocker):
    """Provide a mock CorpusSampler."""
    return mocker.Mock(spec=["requeue_target"])


@pytest.fixture
def mock_config_sampler(mock_sampler, manager_config):
    """Provide a mock CipherManager with a mock sampler."""
    return mock_sampler, manager_config


@pytest.fixture
def mock_mp_queue(mocker):
    """Patch multiprocessing.Queue and RLock to inject mocks."""
    mock_queue_cls = mocker.patch("multiprocessing.Queue")

    mock_job_queue = mocker.Mock()
    mock_result_queue = mocker.Mock()
    mock_stats_queue = mocker.Mock()
    mock_feedback_queue = mocker.Mock()

    mock_queue_cls.side_effect = [
        mock_job_queue,
        mock_result_queue,
        mock_stats_queue,
        mock_feedback_queue,
    ]

    mocker.patch("multiprocessing.RLock", return_value=mocker.Mock())

    return (
        mock_queue_cls,
        mock_job_queue,
        mock_result_queue,
        mock_stats_queue,
        mock_feedback_queue,
    )


class StreamSimulator:
    """A helper to simulate different iterable behaviors for execution testing."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __iter__(self):
        if self.mode == "normal":
            yield ("train", {"text": "A"})
            yield ("train", {"text": "B"})
            yield ("val", {"text": "C"})
        elif self.mode == "exception":
            raise Exception("Stream Failure")
        elif self.mode == "interrupt":
            raise KeyboardInterrupt()


@dataclass
class ExecuteCase:
    """A dataclass for execution outcome test parameters."""

    id: str
    stream_mode: str
    expected_log_level: str
    expected_log_snippet: str


execute_cases = [
    ExecuteCase(
        id="success",
        stream_mode="normal",
        expected_log_level="info",
        expected_log_snippet="Stream finished. Fed 3 items.",
    ),
    ExecuteCase(
        id="exception",
        stream_mode="exception",
        expected_log_level="error",
        expected_log_snippet="Stream error: Stream Failure",
    ),
    ExecuteCase(
        id="interrupt",
        stream_mode="interrupt",
        expected_log_level="warning",
        expected_log_snippet="Job interrupted! Stopping...",
    ),
]


class TestCipherManagerInitialization:
    def test_initialization(self, mocker, mock_mp_queue, mock_config_sampler):
        """Verify the manager sets up the queues and stats properly."""
        _, mock_job_q, mock_result_q, mock_stats_q, mock_feedback_q = mock_mp_queue
        mock_sampler, manager_config = mock_config_sampler
        mocker.patch("os.cpu_count", return_value=6)

        manager = CipherManager(
            config=manager_config,
            text_stream_source=[],
            sampler=mock_sampler,
        )

        assert manager.split_folders == {
            "train": "train_folder",
            "val": "val_folder",
            "test": "test_folder",
            "metadata": "metadata_folder",
        }
        test_bins = sum(
            len(diffs) for diffs in manager_config.dataset_config.test_matrix.values()
        )
        test_count = test_bins * manager_config.dataset_config.ciphers_per_bin
        expected_total_count = (
            manager_config.dataset_config.training_num
            + manager_config.dataset_config.validation_num
            + test_count
        )
        assert manager.total_count == expected_total_count
        assert manager.num_workers == 4

        assert manager.job_queue == mock_job_q
        assert manager.result_queue == mock_result_q
        assert manager.stats_queue == mock_stats_q
        assert manager.feedback_queue == mock_feedback_q
        assert manager.sampler == mock_sampler
        assert isinstance(manager.master_stats, DatasetStatsAggregator)

    def test_initialization_low_cpu(
        self, mocker, mock_mp_queue, manager_config, mock_sampler
    ):
        """Verify the worker count defaults safely on low-core machines."""
        mocker.patch("os.cpu_count", return_value=2)
        manager = CipherManager(manager_config, [], mock_sampler)
        assert manager.num_workers == 1


class TestCipherManagerExecution:
    @pytest.mark.parametrize("case", execute_cases, ids=lambda c: c.id)
    def test_execute_outcomes(
        self, mocker, mock_mp_queue, mock_config_sampler, case: ExecuteCase
    ):
        """Verify the orchestration blocks handle all stream outcomes appropriately."""
        _, mock_job_q, mock_result_q, _, _ = mock_mp_queue
        mock_sampler, manager_config = mock_config_sampler

        mock_log = mocker.patch("cipher_generation.cipher_manager.log")
        mocker.patch("os.cpu_count", return_value=4)

        mock_uploader_cls = mocker.patch(
            "cipher_generation.cipher_manager.DriveUploader"
        )
        mock_producer_cls = mocker.patch(
            "cipher_generation.cipher_manager.CipherProducer"
        )

        manager = CipherManager(
            manager_config, StreamSimulator(case.stream_mode), mock_sampler
        )

        if case.stream_mode == "normal":
            mocker.patch.object(manager, "_feeder_stream", return_value=3)
        elif case.stream_mode == "exception":
            mocker.patch.object(
                manager, "_feeder_stream", side_effect=Exception("Stream Failure")
            )
        elif case.stream_mode == "interrupt":
            mocker.patch.object(
                manager, "_feeder_stream", side_effect=KeyboardInterrupt()
            )

        mocker.patch.object(
            manager.stats_queue, "get", return_value=DatasetStatsAggregator()
        )

        manager.execute()

        log_method = getattr(mock_log, case.expected_log_level)
        log_found = any(
            case.expected_log_snippet in str(call) for call in log_method.call_args_list
        )
        assert log_found, (
            f"Expected to find '{case.expected_log_snippet}' in "
            f"{case.expected_log_level} logs."
        )

        assert mock_producer_cls.call_count == 2

        mock_job_q.put.assert_any_call("STOP")
        mock_result_q.put.assert_any_call("STOP")

        mock_uploader_cls.assert_called_once()
        _, kwargs = mock_uploader_cls.call_args
        assert kwargs["upload_queue"] == mock_result_q

    def test_process_feedback_test_failure(self, manager_config, mock_sampler):
        """Verify that a test split failure correctly decrements the test_tracker."""
        manager = CipherManager(manager_config, [], mock_sampler)
        manager.feedback_queue = queue.Queue()  # type: ignore

        # Artificially increment the tracker to test decrement
        manager.test_tracker[4000][5] = 1

        manager.feedback_queue.put(
            {
                "status": "fail",
                "split": "test",
                "target_length": 4000,
                "target_redundancy": 5,
            }
        )

        successes, failures = manager._process_feedback()

        assert successes == 0
        assert failures == 1
        assert manager.test_tracker[4000][5] == 0
        mock_sampler.requeue_target.assert_called_once_with("test", 4000)

    def test_feeder_stream_processes_nacks_and_requeues(
        self, mocker, manager_config_no_test, mock_sampler
    ):
        """Verify that a worker failure triggers a target requeue and replaces the text."""
        manager_config_no_test.dataset_config.training_num = 2
        stream = iter(
            [
                ("train", {"text": "A", "target_length": 100}),
                ("train", {"text": "B", "target_length": 200}),
                ("train", {"text": "C", "target_length": 300}),
            ]
        )
        manager = CipherManager(manager_config_no_test, stream, mock_sampler)

        manager.feedback_queue = queue.Queue()  # type: ignore
        manager.job_queue = mocker.Mock()

        call_counts = {"count": 0}

        def mock_put(item):
            """Simulate the first cipher failing, and the rest succeeding."""
            call_counts["count"] += 1
            if call_counts["count"] == 1:
                manager.feedback_queue.put(
                    {
                        "status": "fail",
                        "split": item.split,
                        "target_length": item.text_data.get("target_length", 0),
                        "target_redundancy": item.target_redundancy,
                    }
                )
            else:
                manager.feedback_queue.put({"status": "success"})

        manager.job_queue.put.side_effect = mock_put  # type: ignore

        mocker.patch("tqdm.tqdm", return_value=mocker.MagicMock())
        successes = manager._feeder_stream(mocker.Mock())

        assert successes == 2
        assert manager.job_queue.put.call_count == 3  # type: ignore
        mock_sampler.requeue_target.assert_called_once_with("train", 100)

    def test_feeder_stream_respects_total_count(
        self, mocker, manager_config_no_test, mock_sampler
    ):
        """Verify the stream loop cleanly exits once the exact total count is reached."""
        stream = [("train", {"text": f"text_{i}"}) for i in range(10)]
        manager = CipherManager(manager_config_no_test, stream, mock_sampler)

        manager.feedback_queue = queue.Queue()  # type: ignore
        manager.job_queue = mocker.Mock()
        manager.job_queue.put.side_effect = lambda x: manager.feedback_queue.put(  # type: ignore
            {"status": "success"}
        )

        mocker.patch("tqdm.tqdm", return_value=mocker.MagicMock())

        items_fed = manager._feeder_stream(mocker.Mock())

        assert items_fed == 3
        assert manager.job_queue.put.call_count == 3  # type: ignore

    def test_feeder_stream_failsafe_triggered(
        self, mocker, manager_config_no_test, mock_sampler
    ):
        """Verify the stream loop breaks and logs an error if exhausted before target."""
        # Provide a stream with 1 item, but the target is 3.
        stream = [("train", {"text": "A"})]
        manager = CipherManager(manager_config_no_test, stream, mock_sampler)

        manager.feedback_queue = queue.Queue()  # type: ignore
        manager.job_queue = mocker.Mock()
        manager.job_queue.put.side_effect = lambda x: manager.feedback_queue.put(  # type: ignore
            {"status": "success"}
        )

        mock_log = mocker.patch("cipher_generation.cipher_manager.log")
        mocker.patch("tqdm.tqdm", return_value=mocker.MagicMock())

        items_fed = manager._feeder_stream(mocker.Mock())

        assert items_fed == 1
        mock_log.error.assert_called_once_with(
            "Stream exhausted with no in-flight tasks, but target not reached."
        )

    def test_merge_stats_receives_stop_signal(
        self, mocker, mock_mp_queue, manager_config, mock_sampler
    ):
        """Ensure _merge_stats breaks early if a STOP signal is received."""
        _, _, _, mock_stats_q, _ = mock_mp_queue

        manager = CipherManager(manager_config, [], mock_sampler)
        manager.num_workers = 3

        mock_valid_stats = mocker.Mock(spec=DatasetStatsAggregator)
        mock_stats_q.get.side_effect = [mock_valid_stats, "STOP", mock_valid_stats]
        mock_merge = mocker.patch.object(manager.master_stats, "merge")

        manager._merge_stats()

        assert mock_stats_q.get.call_count == 2
        mock_merge.assert_called_once_with(mock_valid_stats)


class TestCipherManagerPeakUpload:
    def test_manager_writes_and_queues_metadata(
        self, mocker, mock_mp_queue, tmp_path, mock_config_sampler
    ):
        """Verify the metadata JSON is accurately generated and queued for upload."""
        _, _, mock_result_q, _, _ = mock_mp_queue
        mock_sampler, manager_config = mock_config_sampler

        mocker.patch("cipher_generation.cipher_manager.DriveUploader")
        mocker.patch("cipher_generation.cipher_manager.CipherProducer")
        mocker.patch("cipher_generation.cipher_manager.shutil.rmtree")

        manager = CipherManager(
            config=manager_config,
            text_stream_source=[],
            sampler=mock_sampler,
        )

        mocker.patch.object(manager, "_feeder_stream", return_value=1)

        expected_peak = 2501
        mocker.patch.object(
            DatasetStatsAggregator,
            "global_max_homophones",
            new_callable=mocker.PropertyMock,
            return_value=expected_peak,
        )

        mocker.patch.object(
            manager.stats_queue, "get", return_value=DatasetStatsAggregator()
        )

        manager.temp_dir = tmp_path / "temp_ciphers"
        manager.execute()

        put_calls = mock_result_q.put.call_args_list
        metadata_payload = put_calls[-2][0][0]

        assert isinstance(metadata_payload, UploadTask)
        assert metadata_payload.split == "metadata"
        assert metadata_payload.filename == Path("metadata.json")
        assert metadata_payload.cipher_count == 0

        assert os.path.exists(metadata_payload.filepath)

        with open(metadata_payload.filepath, encoding="utf-8") as f:
            payload = json.load(f)

        assert payload["max_symbol_id"] == expected_peak


class TestCipherManagerRouting:
    def test_assign_test_redundancy_valid(self, manager_config, mock_sampler):
        """Ensure the manager assigns valid difficulties and increments the tracker."""
        manager = CipherManager(manager_config, [], mock_sampler)

        diff = manager._assign_test_redundancy(4000)

        assert diff is not None
        assert diff in manager_config.dataset_config.test_matrix[4000]
        assert manager.test_tracker[4000][diff] == 1

    def test_assign_test_redundancy_exhausted(self, manager_config, mock_sampler):
        """Ensure the manager returns None when all bins for a length are full."""
        manager = CipherManager(manager_config, [], mock_sampler)
        manager.test_samples_per_bin = 2

        for diff in manager.test_tracker[4000]:
            manager.test_tracker[4000][diff] = 2

        assert manager._assign_test_redundancy(4000) is None

    def test_assign_test_redundancy_invalid_length(
        self, mocker, manager_config, mock_sampler
    ):
        """Ensure invalid lengths are caught and logged."""
        mock_log = mocker.patch("cipher_generation.cipher_manager.log")
        manager = CipherManager(manager_config, [], mock_sampler)

        result = manager._assign_test_redundancy(99999)

        assert result is None
        mock_log.warning.assert_called_once_with(
            "Length 99999 not found in test matrix."
        )

    def test_feeder_stream_payloads(self, mocker, manager_config, mock_sampler):
        """Ensure the stream generates correct CipherTask dataclasses for each split."""
        stream = [
            ("train", {"text": "A", "target_length": 4000}),
            ("val", {"text": "B", "target_length": 5000}),
            ("test", {"text": "C", "target_length": 6500}),
        ]
        manager = CipherManager(manager_config, stream, mock_sampler)

        manager.feedback_queue = queue.Queue()  # type: ignore
        manager.job_queue = mocker.Mock()
        manager.job_queue.put.side_effect = lambda x: manager.feedback_queue.put(  # type: ignore
            {"status": "success"}
        )

        mocker.patch("tqdm.tqdm", return_value=mocker.MagicMock())
        manager._feeder_stream(mocker.Mock())

        assert manager.job_queue.put.call_count == 3  # type: ignore
        calls = manager.job_queue.put.call_args_list  # type: ignore

        train_task = calls[0][0][0]
        assert isinstance(train_task, CipherTask)
        assert train_task.split == "train"
        assert train_task.target_redundancy is None

        val_task = calls[1][0][0]
        assert val_task.split == "val"
        assert val_task.target_redundancy is None

        test_task = calls[2][0][0]
        assert test_task.split == "test"
        assert isinstance(test_task.target_redundancy, int)
        assert (
            test_task.target_redundancy
            in manager_config.dataset_config.test_matrix[6500]
        )

    def test_feeder_stream_skips_exhausted_test_bins(
        self, mocker, manager_config, mock_sampler
    ):
        """Ensure the stream immediately requeues the task if the test bin is full."""
        stream = [("test", {"text": "D", "target_length": 8000})]
        manager = CipherManager(manager_config, stream, mock_sampler)

        for diff in manager.test_tracker[8000]:
            manager.test_tracker[8000][diff] = manager.test_samples_per_bin

        manager.feedback_queue = queue.Queue()  # type: ignore
        manager.job_queue = mocker.Mock()

        mocker.patch("tqdm.tqdm", return_value=mocker.MagicMock())
        manager._feeder_stream(mocker.Mock())

        assert manager.job_queue.put.call_count == 0  # type: ignore
        mock_sampler.requeue_target.assert_called_once_with("test", 8000)
