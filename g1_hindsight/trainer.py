from pathlib import Path
from dataclasses import asdict
import json
import random

import numpy as np
import torch

from .agent import SACAgent
from .config import ExperimentConfig
from .env import UnitreeG1GoalEnv
from .plotting import save_history_csv, save_success_graph
from .replay import ReplayBuffer, Transition


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    config: ExperimentConfig,
    use_her: bool,
    render: bool = False,
    reward_mode: str = "sparse",
) -> None:
    config.validate()
    set_seed(config.seed)
    if reward_mode not in {"sparse", "dense"}:
        raise ValueError(f"지원하지 않는 reward_mode입니다: {reward_mode}")

    her_name = "her" if use_her else "noher"
    variant = her_name if reward_mode == "sparse" else f"dense_{her_name}"
    reward_name = "Dense" if reward_mode == "dense" else "Sparse"
    display_name = f"SAC {reward_name} with HER" if use_her else f"SAC {reward_name} without HER"

    project_root = Path(config.project_root)
    run_name = f"seed_{config.seed}"
    result_dir = project_root / "results" / variant / run_name
    checkpoint_path = project_root / "checkpoints" / variant / run_name / "latest.pt"
    history_path = result_dir / "history.csv"
    graph_path = result_dir / "success_rate.png"
    metadata_path = result_dir / "run_config.json"
    if metadata_path.exists() or history_path.exists() or checkpoint_path.exists():
        raise FileExistsError(f"기존 실험을 덮어쓰지 않습니다: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": config.seed,
        "variant": variant,
        "reward_mode": reward_mode,
        "use_her": use_her,
        "status": "running",
        "config": asdict(config),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"학습 시작: {display_name} | Seed: {config.seed}", flush=True)

    replay = ReplayBuffer(config.buffer_size, seed=config.seed)
    agent = SACAgent(config)
    action_rng = np.random.default_rng(config.seed + 1)

    total_steps = 0
    history = []

    with UnitreeG1GoalEnv(config, render=render, reward_mode=reward_mode) as env:
        for episode in range(1, config.episodes + 1):
            state, goal = env.reset()
            trajectory = []
            success = False
            fallen = False
            episode_return = 0.0
            info = {"distance": float(np.linalg.norm(goal)), "fallen": False}

            for step in range(1, config.max_steps + 1):
                if total_steps < config.start_steps:
                    action = action_rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
                else:
                    action = agent.act(state, goal, deterministic=False)

                next_state, reward, done, info = env.step(action)
                transition = Transition(
                    state=state.copy(),
                    action=action.copy(),
                    reward=reward,
                    next_state=next_state.copy(),
                    goal=goal.copy(),
                    done=done,
                    failure=bool(info["fallen"]),
                )
                replay.add(transition)
                trajectory.append(transition)

                state = next_state
                episode_return += reward
                total_steps += 1

                if (
                    total_steps >= config.update_after
                    and total_steps % config.update_every == 0
                    and len(replay) >= config.batch_size
                ):
                    for _ in range(config.update_every):
                        agent.update(replay, config.batch_size)

                if done:
                    success = bool(info["success"])
                    fallen = bool(info["fallen"])
                    break

            if use_her:
                replay.add_future_her(
                    trajectory,
                    her_k=config.her_k,
                    success_radius=config.success_radius,
                    reward_mode=reward_mode,
                    max_distance=config.max_manhattan_distance,
                    progress_scale=config.progress_reward_scale,
                )

            history.append(
                {
                    "seed": config.seed,
                    "episode": episode,
                    "goal_x": float(goal[0]),
                    "goal_y": float(goal[1]),
                    "success": int(success),
                    "fallen": int(fallen),
                    "steps": step,
                    "return": float(episode_return),
                    "final_distance": float(info["distance"]),
                    "buffer_size": len(replay),
                }
            )

            recent = history[-config.moving_average_window :]
            recent_success_rate = sum(row["success"] for row in recent) / len(recent)
            print(
                f"Seed: {config.seed} | Episode: {episode:4d} | "
                f"Goal: ({goal[0]:.2f}, {goal[1]:.2f}) | "
                f"Success: {success} | Fallen: {fallen} | "
                f"Steps: {step:3d} | Avg{config.moving_average_window}: {recent_success_rate:.3f}"
            )

            save_history_csv(history, history_path)
            if episode % config.plot_every == 0 or episode == config.episodes:
                save_success_graph(
                    history,
                    graph_path,
                    title=f"{display_name} (seed={config.seed})",
                    window=config.moving_average_window,
                    total_episodes=config.episodes,
                )
            if episode % config.checkpoint_every == 0 or episode == config.episodes:
                agent.save(str(checkpoint_path), total_steps, episode)

    metadata["status"] = "complete"
    metadata["completed_episodes"] = len(history)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"학습 완료: {display_name} | Seed: {config.seed}")
    print(f"그래프: {graph_path}")
    print(f"체크포인트: {checkpoint_path}")


def cli(use_her: bool, reward_mode: str = "sparse") -> None:
    from .experiments import cli as experiments_cli

    variant = "her" if use_her else "noher"
    if reward_mode == "dense":
        variant = f"dense_{variant}"
    experiments_cli((variant,))
