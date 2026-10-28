from pathlib import Path
from typing import Dict, List
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: np.ndarray, window: int):
    if len(values) < window:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.int32)
    average = np.convolve(values, np.ones(window) / window, mode="valid")
    episodes = np.arange(window, len(values) + 1)
    return average, episodes


def save_history_csv(history: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "episode",
        "goal_x",
        "goal_y",
        "success",
        "fallen",
        "steps",
        "return",
        "final_distance",
        "buffer_size",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_success_graph(
    history: List[Dict],
    path: Path,
    title: str,
    window: int,
    total_episodes: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    successes = np.asarray([row["success"] for row in history], dtype=np.float32)
    episodes = np.arange(1, len(successes) + 1)
    average, average_episodes = moving_average(successes, window)

    plt.figure(figsize=(10, 5))
    plt.scatter(episodes, successes, s=6, alpha=0.25, label="Episode result")
    if len(average):
        plt.plot(
            average_episodes,
            average,
            color="red",
            linewidth=2,
            label=f"{window}-episode success rate",
        )
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.title(title)
    plt.xlim(1, max(total_episodes, 2))
    plt.ylim(-0.05, 1.05)
    plt.yticks([0, 1], ["Failure", "Success"])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
