import math
import unittest

import numpy as np

from or_gym_online_knapsack_dp import (
    KnapsackInstance,
    evaluate_static_threshold_exact,
    exact_threshold_candidates,
    make_benchmark_instance,
    optimize_static_threshold,
    solve_exact_online_knapsack_dp,
)


class DynamicProgrammingTests(unittest.TestCase):
    def test_hand_solvable_single_item_type(self):
        instance = KnapsackInstance(
            weights=np.array([2], dtype=np.int32),
            values=np.array([5], dtype=np.int32),
            probabilities=np.array([1.0]),
            capacity=4,
            horizon=3,
        )
        solution = solve_exact_online_knapsack_dp(instance)
        self.assertTrue(math.isclose(
            solution.optimal_expected_reward,
            10.0,
            abs_tol=1e-12,
        ))
        # With three draws left, accepting or waiting both yield 10.
        # The implementation rejects exact ties by convention.
        self.assertEqual(solution.action(3, 4, 0), 0)
        self.assertEqual(solution.action(1, 2, 0), 1)

    def test_dynamic_programming_rejects_low_value_item_for_future_option(self):
        instance = KnapsackInstance(
            weights=np.array([2, 2], dtype=np.int32),
            values=np.array([1, 10], dtype=np.int32),
            probabilities=np.array([0.0, 1.0]),
            capacity=2,
            horizon=2,
        )
        solution = solve_exact_online_knapsack_dp(instance)
        self.assertEqual(solution.action(2, 2, 0), 0)

    def test_exact_dp_dominates_optimized_static_threshold(self):
        instance = make_benchmark_instance(
            seed=123,
            n_items=30,
            capacity=40,
            horizon=10,
        )
        exact = solve_exact_online_knapsack_dp(instance)
        threshold = optimize_static_threshold(instance)
        self.assertGreaterEqual(
            exact.optimal_expected_reward + 1e-10,
            threshold.expected_reward,
        )

    def test_threshold_optimizer_matches_exhaustive_candidate_enumeration(self):
        instance = make_benchmark_instance(
            seed=555,
            n_items=20,
            capacity=30,
            horizon=8,
        )
        best = optimize_static_threshold(instance)
        brute = max(
            evaluate_static_threshold_exact(instance, float(x)).expected_reward
            for x in exact_threshold_candidates(instance)
        )
        self.assertTrue(math.isclose(
            best.expected_reward,
            brute,
            abs_tol=1e-10,
        ))

    def test_benchmark_instance_is_reproducible(self):
        a = make_benchmark_instance(seed=999)
        b = make_benchmark_instance(seed=999)
        np.testing.assert_array_equal(a.weights, b.weights)
        np.testing.assert_array_equal(a.values, b.values)
        np.testing.assert_array_equal(a.probabilities, b.probabilities)


if __name__ == "__main__":
    unittest.main()
