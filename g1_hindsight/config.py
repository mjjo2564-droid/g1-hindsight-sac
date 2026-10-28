from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List
import os

import yaml


@dataclass
class ExperimentConfig:
    unitree_root: str = "third_party/unitree_rl_gym"
    policy_device: str = "cpu"
    sac_device: str = "cuda"

    seed: int = 0
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    episodes: int = 3000
    max_steps: int = 128

    grid_x_size: int = 64
    grid_y_size: int = 64
    cell_size: float = 0.05
    success_radius: float = 0.15
    progress_reward_scale: float = 10.0

    max_velocity_x: float = 1.0
    max_velocity_y: float = 1.0
    yaw_gain: float = 1.5
    max_yaw_rate: float = 1.0

    high_level_dt: float = 0.2
    reset_warmup_seconds: float = 1.0
    fall_height: float = 0.45
    min_upright_gravity_z: float = -0.5

    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.98
    tau: float = 0.005
    learning_rate: float = 3e-4
    alpha: float = 0.2
    hidden_size: int = 256

    start_steps: int = 1000
    update_after: int = 1000
    update_every: int = 50
    her_k: int = 4

    moving_average_window: int = 20
    plot_every: int = 1
    checkpoint_every: int = 100

    project_root: str = ""

    @property
    def x_scale(self) -> float:
        return max((self.grid_x_size - 1) * self.cell_size, self.cell_size)

    @property
    def y_scale(self) -> float:
        return max((self.grid_y_size - 1) * self.cell_size, self.cell_size)

    @property
    def max_manhattan_distance(self) -> float:
        return self.x_scale + self.y_scale

    def validate(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed는 0 이상 2**32 미만의 정수여야 합니다.")
        if not isinstance(self.seeds, list) or not self.seeds:
            raise ValueError("seeds는 비어 있지 않은 정수 목록이어야 합니다.")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise ValueError("seeds의 값은 0 이상 2**32 미만의 정수여야 합니다.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds에 중복된 시드가 있습니다.")
        if min(self.moving_average_window, self.plot_every, self.checkpoint_every) <= 0:
            raise ValueError("이동평균 창과 그래프/체크포인트 저장 간격은 양수여야 합니다.")
        if self.grid_x_size * self.grid_y_size < 2:
            raise ValueError("목표 격자는 최소 두 칸 이상이어야 합니다.")
        if self.episodes <= 0 or self.max_steps <= 0:
            raise ValueError("episodes와 max_steps는 양수여야 합니다.")
        if self.cell_size <= 0 or self.success_radius <= 0:
            raise ValueError("cell_size와 success_radius는 양수여야 합니다.")
        if self.progress_reward_scale <= 0:
            raise ValueError("progress_reward_scale은 양수여야 합니다.")
        if self.high_level_dt <= 0:
            raise ValueError("high_level_dt는 양수여야 합니다.")
        if self.batch_size <= 0 or self.buffer_size < self.batch_size:
            raise ValueError("buffer_size는 batch_size 이상이어야 합니다.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma는 0과 1 사이여야 합니다.")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau는 0보다 크고 1 이하여야 합니다.")
        if self.her_k < 0:
            raise ValueError("her_k는 0 이상이어야 합니다.")


def load_config(path: str) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw: Dict[str, Any] = yaml.safe_load(file) or {}

    allowed = {item.name for item in fields(ExperimentConfig)} - {"project_root"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"알 수 없는 config 항목: {', '.join(unknown)}")

    config = ExperimentConfig(**raw)
    config.project_root = str(config_path.parent)

    env_root = os.environ.get("UNITREE_RL_GYM_ROOT")
    root_value = env_root if env_root else config.unitree_root
    root_path = Path(root_value).expanduser()
    if not root_path.is_absolute():
        root_path = config_path.parent / root_path
    config.unitree_root = str(root_path.resolve())

    config.validate()
    return config
