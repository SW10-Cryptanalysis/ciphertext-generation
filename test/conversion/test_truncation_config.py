import pytest
from dataclasses import dataclass
from conversion.truncation_config import TruncationConfig, create_length_sampler


@dataclass
class SamplerTestCase:
    """Defines parameters for validating the stochastic length sampler."""

    config: TruncationConfig
    mocked_choice_index: int
    mocked_randint_return: int
    expected_length: int


sampler_test_cases = [
    SamplerTestCase(
        config=TruncationConfig(length_distribution=[((350, 1000), 1.0)]),
        mocked_choice_index=0,
        mocked_randint_return=500,
        expected_length=500,
    ),
    SamplerTestCase(
        config=TruncationConfig(),
        mocked_choice_index=1,
        mocked_randint_return=2500,
        expected_length=2500,
    ),
]


@pytest.mark.parametrize("test_case", sampler_test_cases)
def test_create_length_sampler(mocker, test_case: SamplerTestCase):
    """Validate that the sampler correctly applies weights and selects integers."""
    mock_choices = mocker.patch("random.choices")
    mock_choices.return_value = [
        test_case.config.length_distribution[test_case.mocked_choice_index][0]
    ]

    mock_randint = mocker.patch("random.randint")
    mock_randint.return_value = test_case.mocked_randint_return

    sampler = create_length_sampler(test_case.config)
    result = sampler()

    assert result == test_case.expected_length
    mock_choices.assert_called_once()
    mock_randint.assert_called_once_with(
        test_case.config.length_distribution[test_case.mocked_choice_index][0][0],
        test_case.config.length_distribution[test_case.mocked_choice_index][0][1],
    )
