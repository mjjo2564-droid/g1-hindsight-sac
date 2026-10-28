from dataclasses import dataclass
from typing import List

import numpy as np

from .goal import compute_reward


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    goal: np.ndarray
    done: bool
    failure: bool = False


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((capacity, 2), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, 2), dtype=np.float32)
        self.goals = np.zeros((capacity, 2), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, transition: Transition) -> None:
        index = self.position
        self.states[index] = transition.state
        self.actions[index] = transition.action
        self.rewards[index, 0] = transition.reward
        self.next_states[index] = transition.next_state
        self.goals[index] = transition.goal
        self.dones[index, 0] = float(transition.done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_future_her(
        self,
        trajectory: List[Transition],
        her_k: int,
        success_radius: float,
        reward_mode: str = "sparse",
        max_distance: float = 1.0,
        progress_scale: float = 10.0,
    ) -> int:
        """future strategy로 재라벨링한 전이만 추가합니다.

        원래 목표 전이는 수집 시점에 이미 buffer에 들어가 있습니다. 넘어지며
        만들어진 최종 위치는 새로운 성공 목표로 사용하지 않습니다.
        """
        added = 0
        for index, transition in enumerate(trajectory):
            if transition.failure:
                continue

            valid_future_indices = [
                future_index
                for future_index in range(index, len(trajectory))
                if not trajectory[future_index].failure
            ]
            if not valid_future_indices:
                continue

            for _ in range(her_k):
                future_index = int(self.rng.choice(valid_future_indices))
                new_goal = trajectory[future_index].next_state.copy()
                new_reward, new_success = compute_reward(
                    transition.state,
                    transition.next_state,
                    new_goal,
                    success_radius,
                    reward_mode,
                    max_distance,
                    progress_scale,
                )
                self.add(
                    Transition(
                        state=transition.state,
                        action=transition.action,
                        reward=new_reward,
                        next_state=transition.next_state,
                        goal=new_goal,
                        done=new_success,
                        failure=False,
                    )
                )
                added += 1
        return added

    def sample(self, batch_size: int):
        if self.size < batch_size:
            raise ValueError("Replay buffer에 batch_size만큼의 전이가 없습니다.")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.goals[indices],
            self.dones[indices],
        )
