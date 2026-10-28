from pathlib import Path
import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from g1_hindsight.config import load_config
from g1_hindsight.experiments import DEFAULT_CONFIG, add_seed_arguments, configure_seeds
from g1_hindsight.plotting import moving_average


def load_seed_successes(root: Path, variant: str, seed: int):
    directory = root / "results" / variant / f"seed_{seed}"
    metadata_path = directory / "run_config.json"
    history_path = directory / "history.csv"
    if not metadata_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"시드 결과가 없습니다: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("seed") != seed or metadata.get("variant") != variant:
        raise ValueError(f"실험 메타데이터가 경로와 일치하지 않습니다: {directory}")
    if metadata.get("status") != "complete":
        raise ValueError(f"완료되지 않은 실험입니다: {directory}")
    with history_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != metadata["config"]["episodes"] or not rows:
        raise ValueError(f"에피소드 수가 실행 설정과 다릅니다: {history_path}")
    for episode, row in enumerate(rows, start=1):
        if int(row["seed"]) != seed or int(row["episode"]) != episode or row["success"] not in ("0", "1"):
            raise ValueError(f"시드/에피소드/성공 값이 잘못되었습니다: {history_path}")
    return np.asarray([int(row["success"]) for row in rows], dtype=np.float64), metadata["config"]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_std(values, axis=None):
    if len(values) < 2:
        return np.zeros_like(np.mean(values, axis=axis))
    return np.std(values, axis=axis, ddof=1)


def load_results(config, variants):
    config.validate()
    root = Path(config.project_root)
    results = {}
    reference_settings = None
    for variant in variants:
        results[variant] = []
        for seed in config.seeds:
            successes, saved_config = load_seed_successes(root, variant, seed)
            # All algorithm/environment parameters must match across compared runs.
            settings = {key: value for key, value in saved_config.items() if key not in {
                "seed", "seeds", "project_root", "unitree_root", "plot_every", "checkpoint_every"
            }}
            if reference_settings is None:
                reference_settings = settings
            elif settings != reference_settings:
                raise ValueError("비교할 실험의 하이퍼파라미터 또는 에피소드 수가 서로 다릅니다.")
            results[variant].append(successes)
        results[variant] = np.stack(results[variant])
    return results


def save_mean_graphs(config, variants, results=None):
    """Save one across-seed mean curve per algorithm/reward condition."""
    if results is None:
        results = load_results(config, variants)
    image_dir = Path(config.project_root) / "graphs"
    image_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for variant in variants:
        values = results[variant]
        window = min(config.moving_average_window, values.shape[1])
        curves = np.stack([moving_average(row, window)[0] for row in values])
        episodes = np.arange(window, values.shape[1] + 1)
        mean, std = curves.mean(axis=0), sample_std(curves, axis=0)
        reward_mode = "dense" if variant.startswith("dense_") else "sparse"
        use_her = variant in ("her", "dense_her")
        algorithm = "her" if use_her else "noher"
        color = "tab:red" if use_her else "tab:blue"
        title = f"SAC {'with' if use_her else 'without'} HER - {reward_mode.capitalize()} reward"
        figure, axes = plt.subplots(figsize=(10, 5))
        axes.plot(episodes, mean, color=color, linewidth=2, label=f"Mean across {len(config.seeds)} seeds")
        if len(config.seeds) > 1:
            axes.fill_between(
                episodes, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                color=color, alpha=0.2, label="+/- 1 sample SD across seeds",
            )
        axes.set_xlabel("Episode")
        axes.set_ylabel(f"{window}-episode training success rate")
        axes.set_title(f"{title}\nSeeds: {', '.join(map(str, config.seeds))}")
        axes.set_ylim(-0.05, 1.05)
        axes.grid(True)
        axes.legend()
        figure.tight_layout()
        output = image_dir / f"{reward_mode}_{algorithm}_mean.png"
        figure.savefig(output, dpi=150)
        plt.close(figure)
        outputs.append(output)
        print(f"시드 평균 그래프: {output}", flush=True)
    return outputs


def compare(config, reward_mode: str) -> Path:
    if reward_mode not in {"sparse", "dense"}:
        raise ValueError(f"지원하지 않는 reward mode입니다: {reward_mode}")
    root = Path(config.project_root)
    prefix = "" if reward_mode == "sparse" else "dense_"
    variants = (f"{prefix}noher", f"{prefix}her")
    results = load_results(config, variants)

    episode_count = results[variants[0]].shape[1]
    window = min(config.moving_average_window, episode_count)
    seed_tag = "seeds_" + "_".join(str(seed) for seed in config.seeds)
    output_dir = root / "results" / "aggregate" / seed_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_rows, summary_rows = [], []
    reward_name = "Sparse" if reward_mode == "sparse" else "Dense"
    figure, axes = plt.subplots(figsize=(10, 5))
    for variant, label, color in (
        (variants[0], "without HER", "tab:blue"),
        (variants[1], "with HER", "tab:red"),
    ):
        values = results[variant]
        curves = np.stack([moving_average(row, window)[0] for row in values])
        episodes = np.arange(window, episode_count + 1)
        mean, std = curves.mean(axis=0), sample_std(curves, axis=0)
        axes.plot(episodes, mean, label=f"SAC {reward_name} {label} (n={len(config.seeds)})", color=color, linewidth=2)
        if len(config.seeds) > 1:
            axes.fill_between(episodes, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.2)
        overall = values.mean(axis=1)
        final = values[:, -window:].mean(axis=1)
        for index, seed in enumerate(config.seeds):
            per_seed_rows.append({
                "variant": variant, "seed": seed, "episodes": episode_count,
                "overall_success_rate": float(overall[index]),
                "final_window": window, "final_window_success_rate": float(final[index]),
            })
        summary_rows.append({
            "variant": variant, "num_seeds": len(config.seeds), "seeds": " ".join(map(str, config.seeds)),
            "episodes_per_seed": episode_count, "final_window": window,
            "overall_success_mean": float(overall.mean()), "overall_success_std": float(sample_std(overall)),
            "final_window_success_mean": float(final.mean()), "final_window_success_std": float(sample_std(final)),
        })
    axes.set_xlabel("Episode")
    axes.set_ylabel(f"{window}-episode training success rate")
    axes.set_title(f"Unitree G1: {reward_name}, mean +/- 1 sample SD across seeds")
    axes.set_ylim(-0.05, 1.05)
    axes.grid(True)
    axes.legend()
    figure.tight_layout()
    output = output_dir / f"comparison_{reward_mode}.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    save_mean_graphs(config, variants, results=results)
    write_csv(output_dir / f"per_seed_{reward_mode}.csv", per_seed_rows)
    write_csv(output_dir / f"summary_{reward_mode}.csv", summary_rows)
    print(f"시드 {len(config.seeds)}개 비교 그래프 및 통계: {output_dir}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="완료된 시드별 학습 결과의 평균과 표준편차를 비교합니다.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--reward-mode", choices=["sparse", "dense"], default="sparse")
    add_seed_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    configure_seeds(config, args)
    compare(config, args.reward_mode)


if __name__ == "__main__":
    main()
