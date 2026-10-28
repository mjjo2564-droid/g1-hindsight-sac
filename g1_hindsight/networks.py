from typing import Tuple

import torch
import torch.nn as nn


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def build_mlp(input_size: int, hidden_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


class GaussianActor(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(observation)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, observation: torch.Tensor):
        mean, log_std = self(observation)
        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)

        log_probability = distribution.log_prob(raw_action)
        log_probability -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_probability = log_probability.sum(dim=-1, keepdim=True)

        deterministic_action = torch.tanh(mean)
        return action, log_probability, deterministic_action


class TwinCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int):
        super().__init__()
        input_size = observation_size + action_size
        self.q1 = build_mlp(input_size, hidden_size, 1)
        self.q2 = build_mlp(input_size, hidden_size, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        critic_input = torch.cat((observation, action), dim=-1)
        return self.q1(critic_input), self.q2(critic_input)
