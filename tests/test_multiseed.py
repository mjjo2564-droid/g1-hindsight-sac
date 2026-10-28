import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict

import numpy as np
from matplotlib.figure import Figure

from compare_results import compare, save_mean_graphs
from g1_hindsight.config import ExperimentConfig
from g1_hindsight.experiments import VARIANTS, run_experiments


class MultiSeedTest(unittest.TestCase):
    def test_invalid_seed_lists_are_rejected(self):
        for seeds in ([], [0, 0], [-1], [2**32], [True], [0.5], "0,1"):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                ExperimentConfig(seeds=seeds).validate()

    def test_sweep_uses_ten_independent_configs_for_each_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(project_root=directory)
            with patch("g1_hindsight.trainer.train") as train, patch("compare_results.compare") as aggregate, redirect_stdout(io.StringIO()):
                runs = run_experiments(config, VARIANTS)
            self.assertEqual(len(set(runs)), 40)
            self.assertEqual(train.call_count, 40)
            self.assertEqual(aggregate.call_count, 2)
            configs = [call.args[0] for call in train.call_args_list]
            self.assertEqual(len({id(c) for c in configs}), 40)
            for call, (seed, variant) in zip(train.call_args_list, runs):
                self.assertEqual(call.args[0].seed, seed)
                self.assertEqual(call.kwargs["use_her"], variant in ("her", "dense_her"))
                self.assertEqual(call.kwargs["reward_mode"], "dense" if variant.startswith("dense_") else "sparse")
            self.assertEqual(config.seed, 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_late_seed_stops_sweep_before_any_training(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(project_root=directory)
            existing = Path(directory) / "results/her/seed_9/run_config.json"
            existing.parent.mkdir(parents=True)
            existing.write_text("{}")
            with patch("g1_hindsight.trainer.train") as train, redirect_stdout(io.StringIO()):
                with self.assertRaises(FileExistsError):
                    run_experiments(config, VARIANTS)
            train.assert_not_called()

    def make_histories(self, root, variants=("noher", "her")):
        config = ExperimentConfig(project_root=str(root), seeds=[0, 1], episodes=4, moving_average_window=2)
        for variant in variants:
            for seed, values in ((0, [0, 0, 0, 1]), (1, [1, 1, 1, 1])):
                directory = root / "results" / variant / f"seed_{seed}"
                directory.mkdir(parents=True)
                saved = asdict(config)
                saved["seed"] = seed
                metadata = {"seed": seed, "variant": variant, "status": "complete", "config": saved}
                (directory / "run_config.json").write_text(json.dumps(metadata))
                with (directory / "history.csv").open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=["seed", "episode", "success"])
                    writer.writeheader()
                    writer.writerows({"seed": seed, "episode": index, "success": value} for index, value in enumerate(values, 1))
        return config

    def test_four_condition_graphs_plot_one_correct_mean_curve_each(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = self.make_histories(root, VARIANTS)
            captured = {}
            original_savefig = Figure.savefig

            def capture(figure, path, *args, **kwargs):
                lines = figure.axes[0].lines
                self.assertEqual(len(lines), 1)
                captured[Path(path).name] = (lines[0].get_xdata(), lines[0].get_ydata())
                return original_savefig(figure, path, *args, **kwargs)

            with patch.object(Figure, "savefig", new=capture):
                outputs = save_mean_graphs(config, VARIANTS)
            expected = {"sparse_noher_mean.png", "sparse_her_mean.png", "dense_noher_mean.png", "dense_her_mean.png"}
            self.assertEqual({path.name for path in outputs}, expected)
            self.assertEqual({path.name for path in (root / "graphs").iterdir()}, expected)
            for x_values, y_values in captured.values():
                np.testing.assert_array_equal(x_values, [2, 3, 4])
                np.testing.assert_allclose(y_values, [0.5, 0.5, 0.75])

    def test_individual_experiment_also_saves_its_mean_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(project_root=directory)
            with patch("g1_hindsight.trainer.train") as train, patch("compare_results.compare") as compare_pair, patch("compare_results.save_mean_graphs") as means, redirect_stdout(io.StringIO()):
                run_experiments(config, ("her",))
            self.assertEqual(train.call_count, 10)
            compare_pair.assert_not_called()
            means.assert_called_once_with(config, ["her"])

    def test_aggregation_uses_between_seed_sample_std(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = self.make_histories(root)
            output = compare(config, "sparse")
            self.assertTrue(output.is_file())
            with (output.parent / "summary_sparse.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(int(row["num_seeds"]), 2)
                self.assertAlmostEqual(float(row["final_window_success_mean"]), 0.75)
                self.assertAlmostEqual(float(row["final_window_success_std"]), np.sqrt(0.125))
                self.assertAlmostEqual(float(row["overall_success_mean"]), 0.625)

    def test_missing_or_incomplete_seed_is_not_silently_averaged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_histories(root)
            config.seeds.append(2)
            with self.assertRaises(FileNotFoundError):
                compare(config, "sparse")
            with self.assertRaises(FileNotFoundError):
                save_mean_graphs(config, ("her",))
            self.assertFalse((root / "graphs").exists())
            config.seeds = [0, 1]
            metadata_path = root / "results/her/seed_1/run_config.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["status"] = "running"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                compare(config, "sparse")


if __name__ == "__main__":
    unittest.main()
