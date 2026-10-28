from typing import Tuple

import numpy as np


def make_input(
    state: np.ndarray,
    goal: np.ndarray,
    x_scale: float,
    y_scale: float,
) -> np.ndarray:
    """원본 GridWorld와 같은 (x, y, goal_x, goal_y) 입력을 만듭니다."""
    return np.asarray(
        [
            state[0] / x_scale,
            state[1] / y_scale,
            goal[0] / x_scale,
            goal[1] / y_scale,
        ],
        dtype=np.float32,
    )


def sparse_reward(
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    success_radius: float,
) -> Tuple[float, bool]:
    success = bool(np.linalg.norm(achieved_goal - desired_goal) <= success_radius)
    return (0.0 if success else -1.0), success


def dense_reward(
    state: np.ndarray,
    next_state: np.ndarray,
    desired_goal: np.ndarray,
    success_radius: float,
    max_distance: float,
    progress_scale: float = 10.0,
) -> Tuple[float, bool]:
    """gridbadHER와 같은 진행도 기반 조밀보상을 계산합니다."""
    _, success = sparse_reward(next_state, desired_goal, success_radius)
    if success:
        return 0.0, True

    current_distance = float(np.abs(state - desired_goal).sum())
    next_distance = float(np.abs(next_state - desired_goal).sum())
    distance_progress = (current_distance - next_distance) / max_distance
    reward = -1.0 + progress_scale * distance_progress
    return float(reward), False


def compute_reward(
    state: np.ndarray,
    next_state: np.ndarray,
    desired_goal: np.ndarray,
    success_radius: float,
    reward_mode: str,
    max_distance: float,
    progress_scale: float = 10.0,
) -> Tuple[float, bool]:
    if reward_mode == "sparse":
        return sparse_reward(next_state, desired_goal, success_radius)
    if reward_mode == "dense":
        return dense_reward(
            state,
            next_state,
            desired_goal,
            success_radius,
            max_distance,
            progress_scale,
        )
    raise ValueError(f"지원하지 않는 reward_mode입니다: {reward_mode}")
