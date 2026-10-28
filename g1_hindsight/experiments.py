"""Run independent SAC experiments with a shared list of comparison seeds."""

import argparse
from dataclasses import replace
from pathlib import Path

from .config import load_config


VARIANTS = ("noher", "her", "dense_noher", "dense_her")
DEFAULT_CONFIG = str(Path(__file__).resolve().parents[1] / "config.yaml")


def add_seed_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, help="이 시드 하나만 실행/비교합니다.")
    group.add_argument("--seeds", nargs="+", type=int, help="실행/비교할 시드 목록입니다.")


def configure_seeds(config, args):
    if args.seed is not None:
        config.seeds = [args.seed]
    elif args.seeds is not None:
        config.seeds = args.seeds
    config.validate()


def run_experiments(config, variants, render=False, dry_run=False):
    config.validate()
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError("중복 없는 지원 실험 목록을 지정하십시오.")
    runs = [(seed, variant) for seed in config.seeds for variant in variants]
    print(f"시드: {config.seeds} | 실험: {list(variants)} | 총 {len(runs)}회", flush=True)

    # Check the whole schedule before starting a costly sweep.
    for seed, variant in runs:
        result = Path(config.project_root) / "results" / variant / f"seed_{seed}"
        checkpoint = Path(config.project_root) / "checkpoints" / variant / f"seed_{seed}" / "latest.pt"
        if not dry_run and (any((result / name).exists() for name in ("history.csv", "run_config.json")) or checkpoint.exists()):
            raise FileExistsError(f"기존 실험을 덮어쓰지 않습니다: {result}")

    for index, (seed, variant) in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] {variant} seed={seed} episodes={config.episodes} -> results/{variant}/seed_{seed}/", flush=True)
        if dry_run:
            continue
        from .trainer import train

        train(
            replace(config, seed=seed),
            use_her=variant in ("her", "dense_her"),
            render=render,
            reward_mode="dense" if variant.startswith("dense_") else "sparse",
        )

    if not dry_run:
        from compare_results import compare, save_mean_graphs

        paired_variants = set()
        for reward_mode, pair in (("sparse", {"noher", "her"}), ("dense", {"dense_noher", "dense_her"})):
            if pair.issubset(variants):
                compare(config, reward_mode)
                paired_variants.update(pair)
        remaining = [variant for variant in variants if variant not in paired_variants]
        if remaining:
            save_mean_graphs(config, remaining)
    return runs


def cli(variants=VARIANTS, allow_variant_selection=False):
    parser = argparse.ArgumentParser(description="설정된 각 시드로 독립적인 학습을 실행합니다.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="학습 없이 실행 목록만 확인합니다.")
    add_seed_arguments(parser)
    if allow_variant_selection:
        parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(variants))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.episodes is not None:
        config.episodes = args.episodes
    configure_seeds(config, args)
    run_experiments(
        config,
        args.variants if allow_variant_selection else variants,
        render=args.render,
        dry_run=args.dry_run,
    )
