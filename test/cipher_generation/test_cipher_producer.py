import pytest
import queue
import os
from dataclasses import dataclass, field
from typing import Any

from cipher_generation.cipher_producer import CipherProducer, ProducerConfig
from cipher_generation.task import CipherTask, UploadTask


@pytest.fixture
def valid_text_stream():
    """Provide a standardized mock text stream dictionary."""
    return {
        "text": "mocktext",
        "source_id": "1",
        "target_length": 8,
        "genres": ["Sci-Fi"],
        "text_with_boundaries": "mock_text",
        "source_name": "Test",
        "length": 8,
    }


@pytest.fixture
def valid_train_task(valid_text_stream):
    """Provide a standardized mock CipherTask for training."""
    return CipherTask(
        split="train",
        text_data=valid_text_stream,
        target_redundancy=None,
    )


@pytest.fixture
def valid_val_task(valid_text_stream):
    """Provide a standardized mock CipherTask for validation."""
    return CipherTask(
        split="val",
        text_data=valid_text_stream,
        target_redundancy=None,
    )


@pytest.fixture
def valid_test_task(valid_text_stream):
    """Provide a standardized mock CipherTask for testing."""
    return CipherTask(
        split="test",
        text_data=valid_text_stream,
        target_redundancy=20,
    )


@pytest.fixture
def valid_tasks(valid_train_task, valid_val_task, valid_test_task):
    """Provide a standardized mock CipherTask for testing."""
    return {
        "train": valid_train_task,
        "val": valid_val_task,
        "test": valid_test_task,
    }


@pytest.fixture
def mock_cipher(mocker):
    """Mock a SubstitutionCipher with a valid __json__ return value."""
    mock = mocker.Mock()
    mock.plaintext = "a" * 10
    mock.redundancy = 5
    mock.num_symbols = 3
    mock.genres = ["Sci-Fi"]
    mock.__json__ = mocker.Mock(return_value={"mock_key": "mock_value"})
    return mock


@pytest.fixture
def producer(mocker, tmp_path):
    """Provide a standard CipherProducer configured with a small batch size."""
    mock_input_queue = mocker.Mock()
    mock_output_queue = mocker.Mock()
    mock_stats_queue = mocker.Mock()
    mock_feedback_queue = mocker.Mock()

    config = ProducerConfig(
        input_queue=mock_input_queue,
        output_queue=mock_output_queue,
        stats_queue=mock_stats_queue,
        feedback_queue=mock_feedback_queue,
        batch_size=2,
        temp_dir=tmp_path / "temp_ciphers",
    )

    p = CipherProducer(
        config=config,
        name="TestProducer",
    )
    p.temp_dir = tmp_path / "temp_ciphers"
    return p


@dataclass
class CipherErrorCase:
    """Defines the parameters for testing various cipher generation failures."""

    id: str
    exception: Exception
    expected_log: str


cipher_error_cases = [
    CipherErrorCase(
        id="value_error",
        exception=ValueError("Invalid cipher setup"),
        expected_log="Error generating cipher: Invalid cipher setup",
    ),
    CipherErrorCase(
        id="unexpected_exception",
        exception=Exception("Critical system failure"),
        expected_log="Unexpected cipher generation error: Critical system failure",
    ),
]


class TestCipherProducerRunLoop:
    """Tests covering the batching loop, disk writing, and cleanup logic."""

    def test_successful_train_batch_zip(
        self,
        mocker,
        producer,
        valid_train_task,
        mock_cipher,
    ):
        """Verify the producer streams to disk and zips exactly at the batch size."""
        producer.input_queue.get.side_effect = [
            valid_train_task,
            valid_train_task,
            "STOP",
        ]

        mocker.patch.object(producer, "generate_cipher", return_value=mock_cipher)
        mocker.patch.object(producer, "_check_redundancy", return_value=True)

        producer.run()

        assert producer.output_queue.put.call_count == 1
        assert producer.feedback_queue.put.call_count == 2
        producer.feedback_queue.put.assert_called_with({"status": "success"})

        upload_task = producer.output_queue.put.call_args[0][0]

        assert isinstance(upload_task, UploadTask)
        assert upload_task.split == "train"
        assert str(upload_task.filename).startswith("batch_train_")
        assert str(upload_task.filename).endswith(".zip")
        assert upload_task.cipher_count == 2
        assert os.path.exists(upload_task.filepath)
        assert not os.path.exists(upload_task.filepath.with_suffix(".jsonl"))

    def test_cleanup_orphan_train_batch(
        self,
        mocker,
        producer,
        valid_train_task,
        mock_cipher,
    ):
        """Verify that STOP forces a zip of an incomplete train batch."""
        producer.input_queue.get.side_effect = [valid_train_task, "STOP"]
        mocker.patch.object(producer, "generate_cipher", return_value=mock_cipher)
        mocker.patch.object(producer, "_check_redundancy", return_value=True)

        producer.run()

        assert producer.output_queue.put.call_count == 1
        upload_task = producer.output_queue.put.call_args[0][0]

        assert upload_task.split == "train"
        assert str(upload_task.filename).endswith(".zip")
        assert upload_task.cipher_count == 1
        assert os.path.exists(upload_task.filepath)

    def test_cleanup_val_test_hoard_signal(
        self,
        mocker,
        producer,
        valid_tasks,
        mock_cipher,
    ):
        """Verify that val and test splits output JSONL tasks for the uploader to merge."""
        producer.input_queue.get.side_effect = [
            valid_tasks["val"],
            valid_tasks["test"],
            "STOP",
        ]
        mocker.patch.object(producer, "generate_cipher", return_value=mock_cipher)
        mocker.patch.object(producer, "_check_redundancy", return_value=True)

        producer.run()

        assert producer.output_queue.put.call_count == 2
        assert producer.feedback_queue.put.call_count == 2

        val_task = producer.output_queue.put.call_args_list[0][0][0]
        test_task = producer.output_queue.put.call_args_list[1][0][0]

        assert isinstance(val_task, UploadTask)
        assert val_task.split == "val"
        assert str(val_task.filename).endswith(".jsonl")
        assert val_task.cipher_count == 1
        assert os.path.exists(val_task.filepath)

        assert isinstance(test_task, UploadTask)
        assert test_task.split == "test"
        assert str(test_task.filename).endswith(".jsonl")

    def test_empty_queue_timeout(self, producer):
        """Verify the producer survives queue timeouts and continues listening."""
        producer.input_queue.get.side_effect = [queue.Empty, queue.Empty, "STOP"]

        producer.run()

        assert producer.input_queue.get.call_count == 3

    def test_unexpected_queue_exception(self, mocker, producer):
        """Verify the producer logs unexpected queue failures and continues listening."""
        mock_log = mocker.patch("cipher_generation.cipher_producer.log")
        producer.input_queue.get.side_effect = [Exception("Queue disconnect"), "STOP"]

        producer.run()

        mock_log.error.assert_called_once()
        assert "Queue disconnect" in mock_log.error.call_args[0][0]
        assert producer.input_queue.get.call_count == 2

    def test_run_sends_nack_on_invalid_cipher(self, mocker, producer, valid_train_task):
        """Verify the loop sends a failure signal if cipher generation returns None."""
        producer.input_queue.get.side_effect = [valid_train_task, "STOP"]
        mocker.patch.object(producer, "generate_cipher", return_value=None)

        producer.run()

        assert producer.input_queue.get.call_count == 2
        producer.output_queue.put.assert_not_called()
        producer.feedback_queue.put.assert_called_once_with(
            {
                "status": "fail",
                "split": "train",
                "target_length": 8,
                "target_redundancy": None,
            }
        )

    def test_run_sends_nack_on_redundancy_failure(
        self, mocker, producer, valid_test_task, mock_cipher
    ):
        """Verify the loop sends a failure signal if the cipher misses its target."""
        producer.input_queue.get.side_effect = [valid_test_task, "STOP"]
        mocker.patch.object(producer, "generate_cipher", return_value=mock_cipher)
        mocker.patch.object(producer, "_check_redundancy", return_value=False)

        producer.run()

        assert producer.input_queue.get.call_count == 2
        producer.output_queue.put.assert_not_called()
        producer.feedback_queue.put.assert_called_once_with(
            {
                "status": "fail",
                "split": "test",
                "target_length": 8,
                "target_redundancy": 20,
            }
        )


class TestCipherProducerRedundancyCheck:
    """Tests covering the redundancy validation logic."""

    def test_check_redundancy_ignores_non_test_splits(
        self, producer, valid_train_task, mock_cipher
    ):
        assert producer._check_redundancy(mock_cipher, valid_train_task) is True

    def test_check_redundancy_validates_exact_match(
        self, producer, valid_test_task, mock_cipher
    ):
        mock_cipher.redundancy = 20
        assert producer._check_redundancy(mock_cipher, valid_test_task) is True

    def test_check_redundancy_rejects_mismatch(
        self, producer, valid_test_task, mock_cipher
    ):
        mock_cipher.redundancy = 19
        assert producer._check_redundancy(mock_cipher, valid_test_task) is False

    def test_check_redundancy_rejects_none(
        self, producer, valid_test_task, mock_cipher
    ):
        mock_cipher.redundancy = None
        assert producer._check_redundancy(mock_cipher, valid_test_task) is False


@dataclass
class CipherSuccessCase:
    """Defines the parameters for testing successful cipher generation routing."""

    id: str
    target_redundancy: float | int | None
    mock_target: str
    expected_kwargs: dict[str, Any] = field(default_factory=dict)
    mock_random: float | None = None


cipher_success_cases = [
    CipherSuccessCase(
        id="continuous_homophonic",
        target_redundancy=None,
        mock_target="cipher_generation.cipher_producer.HomophonicCipher",
        expected_kwargs={},
    ),
    CipherSuccessCase(
        id="discrete_homophonic",
        target_redundancy=25,
        mock_target="cipher_generation.cipher_producer.HomophonicCipher",
        expected_kwargs={"redundancy": 25},
    ),
    CipherSuccessCase(
        id="monoalphabetic",
        target_redundancy=0,
        mock_target="cipher_generation.cipher_producer.MonoalphabeticCipher",
        expected_kwargs={},
    ),
]


class TestGenerateCipherLogic:
    """Tests covering the discrete and continuous routing of the cipher engines."""

    @pytest.mark.parametrize("case", cipher_success_cases, ids=lambda c: c.id)
    def test_generate_cipher_success_routing(
        self, mocker, producer, valid_text_stream, case: CipherSuccessCase
    ):
        """Verify that specific difficulties cleanly route to the correct cipher class."""
        mock_cipher_instance = mocker.Mock()
        mocked_cipher_class = mocker.patch(
            case.mock_target,
            return_value=mock_cipher_instance,
        )

        cipher = producer.generate_cipher(
            valid_text_stream, target_redundancy=case.target_redundancy
        )

        mocked_cipher_class.assert_called_once_with(
            valid_text_stream, **case.expected_kwargs
        )

        mock_cipher_instance.generate_key.assert_called_once()
        mock_cipher_instance.encipher.assert_called_once()

        assert cipher == mock_cipher_instance

    @pytest.mark.parametrize("case", cipher_error_cases, ids=lambda c: c.id)
    def test_generate_cipher_errors(
        self, mocker, producer, valid_text_stream, case: CipherErrorCase
    ):
        """Verify that cipher generation failures are caught and logged gracefully."""
        mock_log = mocker.patch("cipher_generation.cipher_producer.log")

        mocker.patch(
            "cipher_generation.cipher_producer.HomophonicCipher",
            side_effect=case.exception,
        )

        result = producer.generate_cipher(valid_text_stream, target_redundancy=20)

        assert result is None
        mock_log.error.assert_called_once()
        assert case.expected_log in mock_log.error.call_args[0][0]
