from pathlib import Path
import argparse
import time

from g1_hindsight.agent import SACAgent
from g1_hindsight.config import load_config
from g1_hindsight.env import UnitreeG1GoalEnv
from g1_hindsight.trainer import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["her", "noher", "dense_her", "dense_noher"],
        default="her",
    )
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    config.validate()
    set_seed(config.seed)
    reward_mode = "dense" if args.variant.startswith("dense_") else "sparse"
    checkpoint_path = (
        Path(config.project_root) / "checkpoints" / args.variant / f"seed_{config.seed}" / "latest.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")

    agent = SACAgent(config)
    checkpoint_info = agent.load(str(checkpoint_path))
    print(f"Checkpoint: {checkpoint_path}")
    print(f"학습 Episode: {checkpoint_info['episode']}")
    print(f"학습 Total steps: {checkpoint_info['total_steps']}")

    try:
        with UnitreeG1GoalEnv(
            config,
            render=not args.headless,
            reward_mode=reward_mode,
        ) as env:
            for episode in range(1, args.episodes + 1):
                state, goal = env.reset()
                success = False
                fallen = False
                info = {"distance": float("nan")}

                for step in range(1, config.max_steps + 1):
                    action = agent.act(state, goal, deterministic=True)
                    state, _, done, info = env.step(action)
                    if done:
                        success = bool(info["success"])
                        fallen = bool(info["fallen"])
                        break

                print(
                    f"Episode: {episode:3d} | "
                    f"Goal: ({goal[0]:.2f}, {goal[1]:.2f}) | "
                    f"Success: {success} | Fallen: {fallen} | "
                    f"Steps: {step:3d} | Distance: {info['distance']:.3f}"
                )
                if not args.headless:
                    time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n실행을 종료합니다.")


if __name__ == "__main__":
    main()
