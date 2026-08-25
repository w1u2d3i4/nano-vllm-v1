import torch
from torch import nn


def _exponential_race(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    exponential: torch.Tensor,
) -> torch.Tensor:
    scores = logits.float().div_(temperatures.unsqueeze(dim=1))
    exponential.clamp_min_(1e-10).log_()
    return scores.sub_(exponential).argmax(dim=-1)


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        exponential = torch.empty_like(logits).exponential_(1)
        return _exponential_race(logits, temperatures, exponential)
