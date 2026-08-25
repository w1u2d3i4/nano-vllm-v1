import torch

from nanovllm.layers.sampler import _exponential_race


def test_log_space_exponential_race_matches_probability_space() -> None:
    logits = torch.tensor(
        [
            [0.25, -1.5, 3.0, 0.75],
            [-2.0, 1.25, 0.5, 2.5],
            [4.0, 3.75, -0.25, -1.0],
        ],
        dtype=torch.float32,
    )
    temperatures = torch.tensor([0.7, 1.0, 1.5], dtype=torch.float32)
    exponential = torch.tensor(
        [
            [0.4, 1.2, 0.9, 2.0],
            [1.4, 0.7, 2.2, 0.3],
            [0.8, 1.5, 0.2, 2.5],
        ],
        dtype=torch.float32,
    )

    scaled_logits = logits / temperatures.unsqueeze(1)
    probability_space = torch.softmax(scaled_logits, dim=-1)
    expected = probability_space.div(exponential.clamp_min(1e-10)).argmax(dim=-1)
    actual = _exponential_race(
        logits.clone(), temperatures, exponential.clone()
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
