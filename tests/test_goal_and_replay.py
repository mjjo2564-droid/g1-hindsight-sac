import unittest

import numpy as np

from g1_hindsight.goal import dense_reward, make_input, sparse_reward
from g1_hindsight.replay import ReplayBuffer, Transition


class GoalAndReplayTest(unittest.TestCase):
    def test_make_input_matches_gridworld_layout(self):
        result = make_input(
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([3.0, 4.0], dtype=np.float32),
            x_scale=4.0,
            y_scale=8.0,
        )
        np.testing.assert_allclose(result, [0.25, 0.25, 0.75, 0.5])

    def test_sparse_reward(self):
        reward, success = sparse_reward(
            np.asarray([1.0, 1.0]),
            np.asarray([1.1, 1.0]),
            success_radius=0.15,
        )
        self.assertTrue(success)
        self.assertEqual(reward, 0.0)

        reward, success = sparse_reward(
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0]),
            success_radius=0.15,
        )
        self.assertFalse(success)
        self.assertEqual(reward, -1.0)

    def test_dense_reward_uses_manhattan_progress(self):
        reward, success = dense_reward(
            state=np.asarray([0.0, 0.0], dtype=np.float32),
            next_state=np.asarray([0.05, 0.0], dtype=np.float32),
            desired_goal=np.asarray([1.0, 0.0], dtype=np.float32),
            success_radius=0.01,
            max_distance=6.3,
            progress_scale=10.0,
        )
        self.assertFalse(success)
        self.assertAlmostEqual(reward, -1.0 + 10.0 * 0.05 / 6.3, places=6)

        away_reward, _ = dense_reward(
            state=np.asarray([0.05, 0.0], dtype=np.float32),
            next_state=np.asarray([0.0, 0.0], dtype=np.float32),
            desired_goal=np.asarray([1.0, 0.0], dtype=np.float32),
            success_radius=0.01,
            max_distance=6.3,
            progress_scale=10.0,
        )
        self.assertLess(away_reward, -1.0)

    def test_dense_reward_is_zero_at_goal(self):
        reward, success = dense_reward(
            state=np.asarray([0.0, 0.0], dtype=np.float32),
            next_state=np.asarray([1.0, 1.0], dtype=np.float32),
            desired_goal=np.asarray([1.0, 1.0], dtype=np.float32),
            success_radius=0.15,
            max_distance=6.3,
        )
        self.assertTrue(success)
        self.assertEqual(reward, 0.0)

    def test_future_her_adds_k_transitions_per_non_failure_step(self):
        replay = ReplayBuffer(capacity=100, seed=0)
        trajectory = []
        for index in range(3):
            transition = Transition(
                state=np.asarray([index, 0], dtype=np.float32),
                action=np.zeros(2, dtype=np.float32),
                reward=-1.0,
                next_state=np.asarray([index + 1, 0], dtype=np.float32),
                goal=np.asarray([10, 0], dtype=np.float32),
                done=False,
                failure=False,
            )
            replay.add(transition)
            trajectory.append(transition)

        added = replay.add_future_her(trajectory, her_k=4, success_radius=0.01)
        self.assertEqual(added, 12)
        self.assertEqual(len(replay), 15)

    def test_fallen_state_is_not_used_as_her_goal(self):
        replay = ReplayBuffer(capacity=100, seed=0)
        safe = Transition(
            state=np.asarray([0, 0], dtype=np.float32),
            action=np.zeros(2, dtype=np.float32),
            reward=-1.0,
            next_state=np.asarray([1, 0], dtype=np.float32),
            goal=np.asarray([10, 0], dtype=np.float32),
            done=False,
            failure=False,
        )
        fallen = Transition(
            state=np.asarray([1, 0], dtype=np.float32),
            action=np.zeros(2, dtype=np.float32),
            reward=-1.0,
            next_state=np.asarray([2, 0], dtype=np.float32),
            goal=np.asarray([10, 0], dtype=np.float32),
            done=True,
            failure=True,
        )
        added = replay.add_future_her([safe, fallen], her_k=2, success_radius=0.01)
        self.assertEqual(added, 2)
        self.assertEqual(len(replay), 2)
        np.testing.assert_allclose(replay.goals[:2], [[1, 0], [1, 0]])


if __name__ == "__main__":
    unittest.main()
