from pathlib import Path
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .config import ExperimentConfig
from .goal import make_input
from .networks import GaussianActor, TwinCritic


class SACAgent:
    def __init__(self, config: ExperimentConfig):
        requested_device = config.sac_device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA를 찾지 못해 SAC를 CPU에서 실행합니다.")
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.config = config

        self.actor = GaussianActor(4, 2, config.hidden_size).to(self.device)
        self.critic = TwinCritic(4, 2, config.hidden_size).to(self.device)
        self.target_critic = TwinCritic(4, 2, config.hidden_size).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.learning_rate)

    def observation(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return make_input(state, goal, self.config.x_scale, self.config.y_scale)

    @torch.no_grad()
    def act(self, state: np.ndarray, goal: np.ndarray, deterministic: bool = False) -> np.ndarray:
        observation = torch.as_tensor(
            self.observation(state, goal),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        action, _, deterministic_action = self.actor.sample(observation)
        selected = deterministic_action if deterministic else action
        return selected.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, replay, batch_size: int) -> Dict[str, float]:
        states, actions, rewards, next_states, goals, dones = replay.sample(batch_size)

        observations_np = np.concatenate(
            (
                states / np.asarray([self.config.x_scale, self.config.y_scale], dtype=np.float32),
                goals / np.asarray([self.config.x_scale, self.config.y_scale], dtype=np.float32),
            ),
            axis=1,
        )
        next_observations_np = np.concatenate(
            (
                next_states / np.asarray([self.config.x_scale, self.config.y_scale], dtype=np.float32),
                goals / np.asarray([self.config.x_scale, self.config.y_scale], dtype=np.float32),
            ),
            axis=1,
        )

        observations = torch.as_tensor(observations_np, dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(next_observations_np, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions, next_log_probability, _ = self.actor.sample(next_observations)
            target_q1, target_q2 = self.target_critic(next_observations, next_actions)
            target_q = torch.minimum(target_q1, target_q2)
            target_q -= self.config.alpha * next_log_probability
            backup = rewards_tensor + self.config.gamma * (1.0 - dones_tensor) * target_q

        q1, q2 = self.critic(observations, actions_tensor)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        policy_actions, log_probability, _ = self.actor.sample(observations)
        policy_q1, policy_q2 = self.critic(observations, policy_actions)
        policy_q = torch.minimum(policy_q1, policy_q2)
        actor_loss = (self.config.alpha * log_probability - policy_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_parameter.mul_(1.0 - self.config.tau)
                target_parameter.add_(self.config.tau * parameter)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def save(self, path: str, total_steps: int, episode: int) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "seed": self.config.seed,
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_steps": total_steps,
                "episode": episode,
            },
            output_path,
        )

    def load(self, path: str) -> Dict[str, int]:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        return {
            "total_steps": int(checkpoint.get("total_steps", 0)),
            "episode": int(checkpoint.get("episode", 0)),
        }
