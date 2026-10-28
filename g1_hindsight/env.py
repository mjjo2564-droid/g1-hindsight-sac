from pathlib import Path
from typing import Dict, Optional, Tuple
import math

import numpy as np
import yaml

from .config import ExperimentConfig
from .goal import compute_reward


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """MuJoCo의 wxyz quaternion에서 body-frame gravity 방향을 계산합니다."""
    qw, qx, qy, qz = quaternion
    return np.asarray(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def quaternion_yaw(quaternion: np.ndarray) -> float:
    qw, qx, qy, qz = quaternion
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


class UnitreeG1GoalEnv:
    """고정된 G1 보행정책 위에 연속 2차원 속도 명령을 적용합니다."""

    def __init__(
        self,
        config: ExperimentConfig,
        render: bool = False,
        reward_mode: str = "sparse",
    ):
        try:
            import mujoco
            import torch
        except ImportError as error:
            raise RuntimeError(
                "mujoco와 torch가 필요합니다. 먼저 environment.yml로 환경을 만드십시오."
            ) from error

        self.mujoco = mujoco
        self.torch = torch
        self.config = config
        if reward_mode not in {"sparse", "dense"}:
            raise ValueError(f"지원하지 않는 reward_mode입니다: {reward_mode}")
        self.reward_mode = reward_mode
        self.render_enabled = render
        self.viewer = None
        self.rng = np.random.default_rng(config.seed)

        unitree_root = Path(config.unitree_root)
        yaml_path = unitree_root / "deploy/deploy_mujoco/configs/g1.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(
                f"Unitree RL Gym을 찾을 수 없습니다: {yaml_path}\n"
                "프로젝트 루트에서 bash setup_unitree.sh를 실행하십시오."
            )

        with yaml_path.open("r", encoding="utf-8") as file:
            low_level: Dict = yaml.safe_load(file)

        def resolve_unitree_path(value: str) -> str:
            return value.replace("{LEGGED_GYM_ROOT_DIR}", str(unitree_root))

        policy_path = resolve_unitree_path(low_level["policy_path"])
        xml_path = resolve_unitree_path(low_level["xml_path"])
        if not Path(policy_path).is_file():
            raise FileNotFoundError(f"G1 보행정책을 찾을 수 없습니다: {policy_path}")
        if not Path(xml_path).is_file():
            raise FileNotFoundError(f"G1 MuJoCo 모델을 찾을 수 없습니다: {xml_path}")

        self.simulation_dt = float(low_level["simulation_dt"])
        self.control_decimation = int(low_level["control_decimation"])
        self.kps = np.asarray(low_level["kps"], dtype=np.float32)
        self.kds = np.asarray(low_level["kds"], dtype=np.float32)
        self.default_angles = np.asarray(low_level["default_angles"], dtype=np.float32)
        self.ang_vel_scale = float(low_level["ang_vel_scale"])
        self.dof_pos_scale = float(low_level["dof_pos_scale"])
        self.dof_vel_scale = float(low_level["dof_vel_scale"])
        self.action_scale = float(low_level["action_scale"])
        self.cmd_scale = np.asarray(low_level["cmd_scale"], dtype=np.float32)
        self.num_actions = int(low_level["num_actions"])
        self.num_obs = int(low_level["num_obs"])

        high_level_ratio = config.high_level_dt / self.simulation_dt
        self.physics_steps_per_action = int(round(high_level_ratio))
        if not math.isclose(
            self.physics_steps_per_action * self.simulation_dt,
            config.high_level_dt,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("high_level_dt는 simulation_dt의 정수배여야 합니다.")

        policy_device = config.policy_device
        if policy_device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA를 찾지 못해 저수준 보행정책을 CPU에서 실행합니다.")
            policy_device = "cpu"
        self.policy_device = torch.device(policy_device)
        self.policy = torch.jit.load(policy_path, map_location=self.policy_device)
        self.policy.eval()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.simulation_dt

        if self.model.nu != self.num_actions:
            raise ValueError(
                f"MuJoCo actuator 수({self.model.nu})와 정책 action 수({self.num_actions})가 다릅니다."
            )

        self.low_level_action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.low_level_obs = np.zeros(self.num_obs, dtype=np.float32)
        self.counter = 0
        self.origin_xy = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.initial_yaw = 0.0

        if render:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _sample_goal(self) -> np.ndarray:
        while True:
            x_index = int(self.rng.integers(0, self.config.grid_x_size))
            y_index = int(self.rng.integers(0, self.config.grid_y_size))
            if x_index != 0 or y_index != 0:
                return np.asarray(
                    [x_index * self.config.cell_size, y_index * self.config.cell_size],
                    dtype=np.float32,
                )

    def _relative_position(self) -> np.ndarray:
        return self.data.qpos[:2].astype(np.float32).copy() - self.origin_xy

    def _update_goal_marker(self, success: bool = False) -> None:
        if self.viewer is None:
            return

        world_goal = self.origin_xy + self.goal
        color = np.asarray(
            [0.0, 1.0, 0.0, 0.7] if success else [1.0, 0.0, 0.0, 0.7],
            dtype=np.float32,
        )

        with self.viewer.lock():
            scene = self.viewer.user_scn
            scene.ngeom = 1
            self.mujoco.mjv_initGeom(
                scene.geoms[0],
                self.mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.asarray(
                    [
                        self.config.success_radius,
                        self.config.success_radius,
                        0.015,
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    [world_goal[0], world_goal[1], 0.015],
                    dtype=np.float64,
                ),
                np.eye(3, dtype=np.float64).reshape(-1),
                color,
            )

    def _enforce_grid_boundary(self, world_velocity: np.ndarray) -> np.ndarray:
        velocity = world_velocity.copy()
        position = self._relative_position()
        if position[0] <= 0.0 and velocity[0] < 0.0:
            velocity[0] = 0.0
        elif position[0] >= self.config.x_scale and velocity[0] > 0.0:
            velocity[0] = 0.0
        if position[1] <= 0.0 and velocity[1] < 0.0:
            velocity[1] = 0.0
        elif position[1] >= self.config.y_scale and velocity[1] > 0.0:
            velocity[1] = 0.0
        return velocity

    def _command_in_body_frame(self, world_velocity: np.ndarray) -> np.ndarray:
        yaw = quaternion_yaw(self.data.qpos[3:7])
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        body_vx = cosine * world_velocity[0] + sine * world_velocity[1]
        body_vy = -sine * world_velocity[0] + cosine * world_velocity[1]
        yaw_error = wrap_to_pi(yaw - self.initial_yaw)
        yaw_rate = np.clip(
            -self.config.yaw_gain * yaw_error,
            -self.config.max_yaw_rate,
            self.config.max_yaw_rate,
        )
        return np.asarray([body_vx, body_vy, yaw_rate], dtype=np.float32)

    def _build_low_level_observation(self, command: np.ndarray) -> np.ndarray:
        qj = self.data.qpos[7:].astype(np.float32)
        dqj = self.data.qvel[6:].astype(np.float32)
        quaternion = self.data.qpos[3:7].astype(np.float32)
        angular_velocity = self.data.qvel[3:6].astype(np.float32)

        qj = (qj - self.default_angles) * self.dof_pos_scale
        dqj = dqj * self.dof_vel_scale
        gravity_orientation = get_gravity_orientation(quaternion)
        angular_velocity = angular_velocity * self.ang_vel_scale

        period = 0.8
        elapsed = self.counter * self.simulation_dt
        phase = elapsed % period / period
        sin_phase = np.sin(2.0 * np.pi * phase)
        cos_phase = np.cos(2.0 * np.pi * phase)

        observation = self.low_level_obs
        observation[:3] = angular_velocity
        observation[3:6] = gravity_orientation
        observation[6:9] = command * self.cmd_scale
        observation[9 : 9 + self.num_actions] = qj
        observation[9 + self.num_actions : 9 + 2 * self.num_actions] = dqj
        observation[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = self.low_level_action
        observation[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = (
            sin_phase,
            cos_phase,
        )
        return observation

    def _physics_step(self, world_velocity: np.ndarray) -> None:
        torque = pd_control(
            self.target_dof_pos,
            self.data.qpos[7:],
            self.kps,
            np.zeros_like(self.kds),
            self.data.qvel[6:],
            self.kds,
        )
        self.data.ctrl[:] = torque
        self.mujoco.mj_step(self.model, self.data)
        self.counter += 1

        if self.counter % self.control_decimation == 0:
            bounded_velocity = self._enforce_grid_boundary(world_velocity)
            command = self._command_in_body_frame(bounded_velocity)
            observation = self._build_low_level_observation(command)
            observation_tensor = self.torch.from_numpy(observation).to(
                device=self.policy_device,
                dtype=self.torch.float32,
            ).unsqueeze(0)
            with self.torch.no_grad():
                policy_action = self.policy(observation_tensor)
            self.low_level_action = (
                policy_action.squeeze(0).detach().cpu().numpy().astype(np.float32)
            )
            self.target_dof_pos = (
                self.low_level_action * self.action_scale + self.default_angles
            )

            if self.viewer is not None:
                if not self.viewer.is_running():
                    raise KeyboardInterrupt("MuJoCo viewer가 닫혔습니다.")
                self.viewer.sync()

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7 : 7 + self.num_actions] = self.default_angles
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

        # 공식 G1 정책은 LSTM으로 export되어 있으므로 에피소드 사이의 은닉 상태를
        # 공유하지 않습니다.
        if hasattr(self.policy, "reset_memory"):
            self.policy.reset_memory()

        self.low_level_action.fill(0.0)
        self.target_dof_pos = self.default_angles.copy()
        self.low_level_obs.fill(0.0)
        self.counter = 0
        self.initial_yaw = quaternion_yaw(self.data.qpos[3:7])

        zero_velocity = np.zeros(2, dtype=np.float32)
        warmup_steps = int(round(self.config.reset_warmup_seconds / self.simulation_dt))
        for _ in range(warmup_steps):
            self._physics_step(zero_velocity)

        self.origin_xy = self.data.qpos[:2].astype(np.float32).copy()
        self.initial_yaw = quaternion_yaw(self.data.qpos[3:7])
        self.goal = self._sample_goal()
        self._update_goal_marker(success=False)
        if self.viewer is not None:
            self.viewer.sync()
        return np.zeros(2, dtype=np.float32), self.goal.copy()

    def step(self, normalized_action: np.ndarray):
        state = self._relative_position()
        action = np.clip(np.asarray(normalized_action, dtype=np.float32), -1.0, 1.0)
        world_velocity = action * np.asarray(
            [self.config.max_velocity_x, self.config.max_velocity_y],
            dtype=np.float32,
        )

        for _ in range(self.physics_steps_per_action):
            self._physics_step(world_velocity)

        next_state = self._relative_position()
        reward, success = compute_reward(
            state,
            next_state,
            self.goal,
            self.config.success_radius,
            self.reward_mode,
            self.config.max_manhattan_distance,
            self.config.progress_reward_scale,
        )

        gravity_z = float(get_gravity_orientation(self.data.qpos[3:7])[2])
        fallen = bool(
            self.data.qpos[2] < self.config.fall_height
            or gravity_z > self.config.min_upright_gravity_z
        )
        if fallen:
            success = False
            reward = -1.0

        done = success or fallen
        if success:
            self._update_goal_marker(success=True)
            if self.viewer is not None:
                self.viewer.sync()
        info = {
            "success": success,
            "fallen": fallen,
            "distance": float(np.linalg.norm(next_state - self.goal)),
            "base_height": float(self.data.qpos[2]),
            "gravity_z": gravity_z,
        }
        return next_state, reward, done, info

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
