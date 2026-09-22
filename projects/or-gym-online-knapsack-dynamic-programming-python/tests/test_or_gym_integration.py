import math
import unittest

import numpy as np

from or_gym_online_knapsack_dp import (
    make_benchmark_instance,
    make_configured_env,
    run_episode,
    safe_reset_online_env,
    solve_exact_online_knapsack_dp,
)


class ORGymnasiumIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import gymnasium  # noqa: F401
            import or_gymnasium  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("OR-Gymnasium integration dependency not installed")

    def test_registered_environment_and_action_mask(self):
        instance = make_benchmark_instance(seed=17)
        env = make_configured_env(instance)
        obs, _ = safe_reset_online_env(env, seed=123)
        self.assertIn("state", obs)
        self.assertIn("action_mask", obs)
        self.assertEqual(obs["action_mask"].shape, (2,))
        self.assertEqual(env.action_space.n, 2)
        env.close()

    def test_safe_reset_restarts_online_step_counter(self):
        instance = make_benchmark_instance(seed=18)
        env = make_configured_env(instance)
        safe_reset_online_env(env, seed=100)
        env.step(0)
        self.assertEqual(env.unwrapped.step_counter, 1)
        safe_reset_online_env(env, seed=101)
        self.assertEqual(env.unwrapped.step_counter, 0)
        env.close()

    def test_exact_dp_episode_is_feasible(self):
        instance = make_benchmark_instance(seed=19)
        solution = solve_exact_online_knapsack_dp(instance)
        env = make_configured_env(instance)
        result = run_episode(
            env,
            instance,
            solution.action,
            seed=500,
        )
        self.assertLessEqual(result.final_weight, instance.capacity)
        self.assertLessEqual(result.steps, instance.horizon)
        self.assertGreaterEqual(result.reward, 0.0)
        env.close()

    def test_monte_carlo_mean_tracks_exact_expected_value(self):
        # Small enough for CI, large enough for a broad statistical guard.
        instance = make_benchmark_instance(
            seed=20,
            n_items=200,
            capacity=200,
            horizon=50,
        )
        solution = solve_exact_online_knapsack_dp(instance)
        env = make_configured_env(instance)
        rewards = np.array([
            run_episode(env, instance, solution.action, seed=10000+i).reward
            for i in range(800)
        ])
        env.close()

        mean = float(np.mean(rewards))
        se = float(np.std(rewards, ddof=1) / math.sqrt(len(rewards)))
        # 6 standard errors is deliberately a robustness guard, not a
        # statistical performance claim.
        self.assertLessEqual(
            abs(mean - solution.optimal_expected_reward),
            6.0 * se + 1e-9,
        )


if __name__ == "__main__":
    unittest.main()
