import pytest
from dataclasses import dataclass
from cipher_generation.config import DatasetConfig
from fetching.corpus_sampler import CorpusSampler, Metadata, Target
from utils.text_sampling import Book


@pytest.fixture
def dummy_config():
    """Provides a controlled DatasetConfig for deterministic testing."""
    return DatasetConfig(
        training_num=100,
        validation_num=10,
        test_matrix={350: [5, 10], 400: [0]},
        ciphers_per_bin=2,
        foundation_range=(100, 200),
        transition_range=(201, 300),
        frontier_range=(301, 400),
    )


@pytest.fixture
def default_genre_map():
    return {"1": ["Fiction"], "2": ["Science Fiction"]}


@pytest.fixture
def mock_constants(mocker):
    mocker.patch("fetching.corpus_sampler.BOOK_IDS_VALIDATION", ["999"])


class TestCorpusSamplerInit:
    def test_init_flattens_test_matrix_correctly(self, dummy_config, default_genre_map):
        sampler = CorpusSampler(dummy_config, default_genre_map)

        assert sampler.test_targets[350] == 4
        assert sampler.test_targets[400] == 2
        assert sampler.targets["train"] == 100
        assert sampler.targets["val"] == 10
        assert sampler.targets["test"] == 6

    def test_init_builds_test_pool_correctly(self, dummy_config, default_genre_map):
        sampler = CorpusSampler(dummy_config, default_genre_map)

        assert len(sampler.test_pool) == 6
        assert sampler.test_pool.count(350) == 4
        assert sampler.test_pool.count(400) == 2


class TestCorpusSamplerIsComplete:
    def test_is_complete_returns_false_initially(self, dummy_config, default_genre_map):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        assert not sampler._is_complete()

    @dataclass
    class IsCompleteTestCase:
        """Encapsulates parameters for testing the completeness check."""

        desc: str
        train_c: int
        val_c: int
        test_c: int
        expected: bool

    is_complete_test_cases = [
        IsCompleteTestCase(
            desc="Complete: All quotas met",
            train_c=100,
            val_c=10,
            test_c=6,
            expected=True,
        ),
        IsCompleteTestCase(
            desc="Complete: One quota met",
            train_c=105,
            val_c=12,
            test_c=8,
            expected=True,
        ),
        IsCompleteTestCase(
            desc="Incomplete: One quota not met",
            train_c=99,
            val_c=10,
            test_c=6,
            expected=False,
        ),
        IsCompleteTestCase(
            desc="Incomplete: Two quotas not met",
            train_c=100,
            val_c=9,
            test_c=6,
            expected=False,
        ),
        IsCompleteTestCase(
            desc="Incomplete: All quotas not met",
            train_c=100,
            val_c=10,
            test_c=5,
            expected=False,
        ),
    ]

    @pytest.mark.parametrize("case", is_complete_test_cases, ids=lambda c: c.desc)
    def test_is_complete_matrix(
        self, dummy_config, default_genre_map, case: IsCompleteTestCase
    ):
        sampler = CorpusSampler(dummy_config, default_genre_map, buffer_chars=100)
        sampler.counts["train"] = case.train_c
        sampler.counts["val"] = case.val_c
        sampler.total_test_count = case.test_c

        assert sampler._is_complete() == case.expected


class TestCorpusSamplerGenerateStream:
    def test_generate_stream_stops_when_complete(
        self, mocker, mock_constants, dummy_config, default_genre_map
    ):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        mocker.patch.object(sampler, "_is_complete", return_value=True)
        mock_process = mocker.patch.object(sampler, "_process_book")

        dummy_stream = [{"id": "1", "text": "dummy"}]
        results = list(sampler.generate_stream(dummy_stream))

        assert len(results) == 0
        mock_process.assert_not_called()

    def test_generate_stream_skips_validation_ids(
        self, mocker, mock_constants, dummy_config, default_genre_map
    ):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        mock_process = mocker.patch.object(sampler, "_process_book")

        dummy_stream = [{"id": "999", "text": "dummy"}]
        list(sampler.generate_stream(dummy_stream))

        mock_process.assert_not_called()

    def test_generate_stream_yields_from_process_book(
        self, mocker, mock_constants, dummy_config, default_genre_map
    ):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        expected_yield = ("train", {"text": "chunk", "target_length": 100})
        mocker.patch.object(
            sampler, "_process_book", return_value=iter([expected_yield])
        )

        dummy_stream = [{"id": "1", "text": "dummy"}]
        results = list(sampler.generate_stream(dummy_stream))

        assert results == [expected_yield]


class TestCorpusSamplerWeightedSplit:
    def test_returns_none_when_all_full(self, dummy_config, default_genre_map):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        sampler.counts["train"] = 100
        sampler.counts["val"] = 10
        sampler.total_test_count = 6

        assert sampler._get_weighted_split() is None

    def test_returns_only_available_split(self, dummy_config, default_genre_map):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        sampler.counts["train"] = 100
        sampler.total_test_count = 6
        sampler.counts["val"] = 5

        assert sampler._get_weighted_split() == "val"

    def test_returns_train_when_others_full(self, dummy_config, default_genre_map):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        sampler.counts["train"] = 50
        sampler.counts["val"] = 10
        sampler.total_test_count = 6

        assert sampler._get_weighted_split() == "train"

    def test_returns_test_when_others_full(self, dummy_config, default_genre_map):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        sampler.counts["train"] = 100
        sampler.counts["val"] = 10
        sampler.total_test_count = 2

        assert sampler._get_weighted_split() == "test"


class TestCorpusSamplerBurstTargets:
    def test_get_burst_targets_breaks_on_none_split(
        self, mocker, dummy_config, default_genre_map
    ):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        mocker.patch.object(sampler, "_get_weighted_split", side_effect=["train", None])
        mocker.patch.object(sampler, "_pop_target_len", return_value=100)

        targets = sampler._get_burst_targets()

        assert len(targets) == 1
        assert targets[0]["split"] == "train"

    def test_get_burst_targets_skips_none_target_len(
        self, mocker, dummy_config, default_genre_map
    ):
        sampler = CorpusSampler(dummy_config, default_genre_map, max_chunks_per_book=1)
        mocker.patch.object(sampler, "_get_weighted_split", return_value="test")
        mocker.patch.object(sampler, "_pop_target_len", return_value=None)

        targets = sampler._get_burst_targets()

        assert len(targets) == 0

    def test_get_burst_targets_reserves_quotas_upfront(
        self, mocker, dummy_config, default_genre_map
    ):
        """Verify that _get_burst_targets reserves quotas the moment they are planned."""
        sampler = CorpusSampler(dummy_config, default_genre_map, max_chunks_per_book=1)
        mocker.patch.object(sampler, "_get_weighted_split", return_value="test")
        mocker.patch.object(sampler, "_pop_target_len", return_value=350)
        mock_reserve = mocker.patch.object(sampler, "_reserve_target")

        targets = sampler._get_burst_targets()

        assert len(targets) == 1
        mock_reserve.assert_called_once_with("test", 350)

    def test_pop_target_len_returns_test_pool_item(
        self, dummy_config, default_genre_map
    ):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        sampler.test_pool = [350]

        result = sampler._pop_target_len("test")

        assert result == 350
        assert len(sampler.test_pool) == 0

    def test_pop_target_len_returns_none_if_test_pool_empty(
        self, dummy_config, default_genre_map
    ):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        sampler.test_pool = []

        result = sampler._pop_target_len("test")

        assert result is None

    def test_pop_target_len_generates_stochastic_length(
        self, mocker, dummy_config, default_genre_map
    ):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        mocker.patch("random.choices", return_value=[(100, 200)])
        mocker.patch("random.randint", return_value=150)

        result = sampler._pop_target_len("train")

        assert result == 150


class TestCorpusSamplerFitTargets:
    def test_fit_targets_to_book_trims_and_requeues_removed_targets(
        self, mocker, dummy_config, default_genre_map
    ):
        """Verify targets that exceed the book length are removed and refunded."""
        sampler = CorpusSampler(dummy_config, default_genre_map, buffer_chars=100)
        mock_requeue = mocker.patch.object(sampler, "requeue_target")

        targets: list[Target] = [
            {"split": "train", "len": 1000},
            {"split": "test", "len": 500},
        ]

        valid_targets = sampler._fit_targets_to_book(targets, 1500)

        assert len(valid_targets) == 1
        assert valid_targets[0]["split"] == "train"
        mock_requeue.assert_called_once_with("test", 500)

    def test_fit_targets_shuffles_when_book_is_large(
        self, mocker, dummy_config, default_genre_map
    ):
        sampler = CorpusSampler(dummy_config, default_genre_map, buffer_chars=10)
        mock_shuffle = mocker.patch("random.shuffle")
        targets: list[Target] = [{"split": "train", "len": 100}]

        sampler._fit_targets_to_book(targets, 1000)

        mock_shuffle.assert_called_once_with(targets)


class TestCorpusSamplerProcessBookBranches:
    def test_process_book_returns_early_no_targets(
        self, mocker, dummy_config, default_genre_map
    ):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        mocker.patch.object(sampler, "_get_burst_targets", return_value=[])

        book = Book(
            id="1",
            text="dummy",
            source_name="Title",
            source_type="",
            fallback_genres=[],
        )
        results = list(sampler._process_book(book))

        assert len(results) == 0

    def test_process_book_returns_early_no_valid_targets(
        self, mocker, dummy_config, default_genre_map
    ):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        mocker.patch.object(
            sampler, "_get_burst_targets", return_value=[{"split": "train", "len": 100}]
        )
        mocker.patch.object(sampler, "_fit_targets_to_book", return_value=[])

        book = Book(
            id="1",
            text="dummy",
            source_name="Title",
            source_type="",
            fallback_genres=[],
        )
        results = list(sampler._process_book(book))

        assert len(results) == 0


class TestCorpusSamplerExtractAndFormat:
    def test_extract_chunks_skips_none_results_and_requeues_all_splits(
        self, mocker, dummy_config, default_genre_map
    ):
        """Verify that failed extractions trigger requeue_target for any split type."""
        sampler = CorpusSampler(dummy_config, default_genre_map)
        mock_requeue = mocker.patch.object(sampler, "requeue_target")

        mocker.patch(
            "fetching.corpus_sampler.extract_specific_chunk", return_value=None
        )
        mock_record = mocker.patch.object(sampler, "_record_and_format")

        book = Book(
            id="1",
            text="dummy",
            source_name="Title",
            source_type="",
            fallback_genres=[],
        )

        targets: list[Target] = [
            {"split": "train", "len": 100},
            {"split": "test", "len": 350},
        ]

        results = list(sampler._extract_chunks_from_partitions("dummy", targets, book))

        assert len(results) == 0
        mock_record.assert_not_called()

        assert mock_requeue.call_count == 2
        mock_requeue.assert_any_call("train", 100)
        mock_requeue.assert_any_call("test", 350)

    def test_record_and_format_structures_data_without_incrementing(
        self, dummy_config, default_genre_map
    ):
        """Verify _record_and_format purely structures the data and does not alter tracking state."""
        sampler = CorpusSampler(dummy_config, default_genre_map)
        target: Target = {"split": "test", "len": 350}
        result = ("clean_text", "clean_text_bounds")
        meta: Metadata = {
            "source_id": "1",
            "source_name": "Title",
            "genres": ["Fiction"],
        }

        split, stream_data = sampler._record_and_format(target, result, meta)

        assert split == "test"
        assert stream_data["text"] == "clean_text"
        assert stream_data["text_with_boundaries"] == "clean_text_bounds"
        assert stream_data["target_length"] == 350
        assert stream_data["genres"] == ["Fiction"]

        # Ensure state remained fully pristine
        assert sampler.total_test_count == 0
        assert sampler.counts["test"][350] == 0


class TestCorpusSamplerStateTracking:
    def test_adjust_quota_train_split(self, dummy_config, default_genre_map):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        sampler._adjust_quota("train", 1000, 1)
        assert sampler.counts["train"] == 1

    def test_adjust_quota_test_split(self, dummy_config, default_genre_map):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        sampler._adjust_quota("test", 350, 2)
        assert sampler.counts["test"][350] == 2
        assert sampler.total_test_count == 2

    def test_reserve_target(self, mocker, dummy_config, default_genre_map):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        mock_adjust = mocker.patch.object(sampler, "_adjust_quota")

        sampler._reserve_target("val", 200)
        mock_adjust.assert_called_once_with("val", 200, amount=1)

    def test_requeue_target(self, mocker, dummy_config, default_genre_map):
        sampler = CorpusSampler(dummy_config, default_genre_map)
        mock_adjust = mocker.patch.object(sampler, "_adjust_quota")
        sampler.test_pool = []

        sampler.requeue_target("test", 400)

        mock_adjust.assert_called_once_with("test", 400, amount=-1)
        assert sampler.test_pool == [400]
