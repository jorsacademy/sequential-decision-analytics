from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.stats import t as student_t


ActionFn = Callable[[int, int, int], int]


@dataclass(frozen=True)
class KnapsackInstance:
    weights: np.ndarray
    values: np.ndarray
    probabilities: np.ndarray
    capacity: int = 200
    horizon: int = 50

    def __post_init__(self) -> None:
        w = np.asarray(self.weights)
        v = np.asarray(self.values)
        p = np.asarray(self.probabilities, dtype=float)
        if w.ndim != 1 or v.ndim != 1 or p.ndim != 1:
            raise ValueError("instance arrays must be one-dimensional")
        if not (len(w) == len(v) == len(p)) or len(w) == 0:
            raise ValueError("instance arrays must be nonempty and equally sized")
        if np.any(w <= 0) or np.any(v < 0) or np.any(p < 0):
            raise ValueError("invalid weights, values, or probabilities")
        if not math.isclose(float(p.sum()), 1.0, abs_tol=1e-12):
            raise ValueError("probabilities must sum to one")
        if self.capacity <= 0 or self.horizon <= 0:
            raise ValueError("capacity and horizon must be positive")


@dataclass(frozen=True)
class DPSolution:
    future_value: np.ndarray
    instance: KnapsackInstance

    @property
    def optimal_expected_reward(self) -> float:
        return float(self.future_value[self.instance.horizon, self.instance.capacity])

    def action(self, remaining_steps: int, remaining_capacity: int, item_index: int) -> int:
        if not 1 <= remaining_steps <= self.instance.horizon:
            raise ValueError("remaining_steps outside horizon")
        if not 0 <= remaining_capacity <= self.instance.capacity:
            raise ValueError("remaining_capacity outside capacity")
        if not 0 <= item_index < len(self.instance.weights):
            raise ValueError("invalid item index")
        weight = int(self.instance.weights[item_index])
        if weight > remaining_capacity:
            return 0
        reject = self.future_value[remaining_steps - 1, remaining_capacity]
        accept = (
            float(self.instance.values[item_index])
            + self.future_value[remaining_steps - 1, remaining_capacity - weight]
        )
        # Reject exact value ties; either action is Bellman-optimal in a tie.
        return int(accept > reject + 1e-12)


@dataclass(frozen=True)
class ThresholdPolicyResult:
    threshold: float
    expected_reward: float
    future_value: np.ndarray


@dataclass(frozen=True)
class EpisodeResult:
    reward: float
    accepted_items: int
    final_weight: int
    steps: int


@dataclass(frozen=True)
class EvaluationSummary:
    policy_name: str
    episodes: int
    mean_reward: float
    std_reward: float
    ci95_low: float
    ci95_high: float
    mean_final_weight: float
    mean_accepted_items: float


@dataclass(frozen=True)
class ExperimentResult:
    exact_expected_reward: float
    threshold: float
    threshold_expected_reward: float
    exact_mc: EvaluationSummary
    threshold_mc: EvaluationSummary
    always_accept_mc: EvaluationSummary
    random_mc: EvaluationSummary
    exact_minus_threshold_mean: float
    exact_minus_threshold_ci95_low: float
    exact_minus_threshold_ci95_high: float


def make_benchmark_instance(
    *, seed: int = 20260827, n_items: int = 200, capacity: int = 200, horizon: int = 50
) -> KnapsackInstance:
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 100, size=n_items, dtype=np.int32)
    values = rng.integers(1, 100, size=n_items, dtype=np.int32)
    popularity = rng.integers(1, 10, size=n_items, dtype=np.int32)
    probabilities = popularity.astype(np.float64)
    probabilities /= probabilities.sum()
    return KnapsackInstance(weights, values, probabilities, capacity, horizon)


def solve_exact_online_knapsack_dp(instance: KnapsackInstance) -> DPSolution:
    """Exact finite-horizon Bellman recursion for i.i.d. online item arrivals."""
    H, C = instance.horizon, instance.capacity
    weights = np.asarray(instance.weights, dtype=np.int32)
    values = np.asarray(instance.values, dtype=np.float64)
    probs = np.asarray(instance.probabilities, dtype=np.float64)

    V = np.zeros((H + 1, C + 1), dtype=np.float64)
    capacities = np.arange(C + 1, dtype=np.int32)[None, :]
    weights_col = weights[:, None]
    feasible = weights_col <= capacities
    continuation_index = np.maximum(capacities - weights_col, 0)

    for t in range(1, H + 1):
        previous = V[t - 1]
        reject = previous[None, :]
        accept = values[:, None] + previous[continuation_index]
        V[t] = probs @ np.where(feasible, np.maximum(reject, accept), reject)

    return DPSolution(V, instance)


def evaluate_static_threshold_exact(
    instance: KnapsackInstance, threshold: float
) -> ThresholdPolicyResult:
    """Exact value of accept-if-value/weight>=threshold, when the item fits."""
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")

    H, C = instance.horizon, instance.capacity
    weights = np.asarray(instance.weights, dtype=np.int32)
    values = np.asarray(instance.values, dtype=np.float64)
    probs = np.asarray(instance.probabilities, dtype=np.float64)
    accepts = values >= threshold * weights

    V = np.zeros((H + 1, C + 1), dtype=np.float64)
    capacities = np.arange(C + 1, dtype=np.int32)[None, :]
    weights_col = weights[:, None]
    feasible_accept = accepts[:, None] & (weights_col <= capacities)
    continuation_index = np.maximum(capacities - weights_col, 0)

    for t in range(1, H + 1):
        previous = V[t - 1]
        reject = previous[None, :]
        accept = values[:, None] + previous[continuation_index]
        V[t] = probs @ np.where(feasible_accept, accept, reject)

    return ThresholdPolicyResult(float(threshold), float(V[H, C]), V)


def exact_threshold_candidates(instance: KnapsackInstance) -> np.ndarray:
    ratios = np.unique(
        np.asarray(instance.values, dtype=float) / np.asarray(instance.weights, dtype=float)
    )
    ratios.sort()
    candidates = [0.0, *(float(x) for x in ratios), float(ratios[-1] + 1.0)]
    candidates.extend(float((a + b) / 2.0) for a, b in zip(ratios[:-1], ratios[1:]))
    return np.array(sorted(set(candidates)), dtype=float)


def optimize_static_threshold(instance: KnapsackInstance) -> ThresholdPolicyResult:
    best = None
    for threshold in exact_threshold_candidates(instance):
        candidate = evaluate_static_threshold_exact(instance, float(threshold))
        if best is None or (candidate.expected_reward, -candidate.threshold) > (
            best.expected_reward,
            -best.threshold,
        ):
            best = candidate
    assert best is not None
    return best


def threshold_action_fn(instance: KnapsackInstance, threshold: float) -> ActionFn:
    def action(_steps: int, capacity: int, item: int) -> int:
        weight = int(instance.weights[item])
        return int(weight <= capacity and float(instance.values[item]) >= threshold * weight)
    return action


def always_accept_feasible_action(instance: KnapsackInstance) -> ActionFn:
    def action(_steps: int, capacity: int, item: int) -> int:
        return int(int(instance.weights[item]) <= capacity)
    return action


def _import_or_gymnasium():
    try:
        import gymnasium as gym
        import or_gymnasium  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install OR-Gymnasium project requirements first.") from exc
    return gym


def make_configured_env(instance: KnapsackInstance):
    """
    Create registered OR-Gymnasium Knapsack-v3, then inject a deterministic
    catalogue while retaining upstream step/reward/mask/termination semantics.
    """
    gym = _import_or_gymnasium()
    env = gym.make("Knapsack-v3")
    base = env.unwrapped
    n = len(instance.weights)
    base.N = n
    base.item_numbers = np.arange(n)
    base.item_weights = np.asarray(instance.weights, dtype=np.int32).copy()
    base.item_values = np.asarray(instance.values, dtype=np.int32).copy()
    base.item_probs = np.asarray(instance.probabilities, dtype=np.float64).copy()
    base.item_limits_init = np.ones(n, dtype=np.int32)
    base.item_limits = base.item_limits_init.copy()
    base.max_weight = int(instance.capacity)
    base.step_limit = int(instance.horizon)
    base.randomize_params_on_reset = False
    return env


def safe_reset_online_env(env, *, seed: int):
    """
    Upstream Knapsack-v3 currently increments step_counter in step() without
    resetting it in the inherited reset path. Restore it before each episode.
    """
    base = env.unwrapped
    if hasattr(base, "step_counter"):
        base.step_counter = 0
    return env.reset(seed=int(seed))


def run_episode(
    env, instance: KnapsackInstance, action_fn: ActionFn, *, seed: int
) -> EpisodeResult:
    safe_reset_online_env(env, seed=seed)
    base = env.unwrapped
    total_reward = 0.0
    accepted = 0
    steps = 0

    while True:
        remaining_steps = int(base.step_limit - base.step_counter)
        remaining_capacity = int(base.max_weight - base.current_weight)
        item = int(base.current_item)
        action = int(action_fn(remaining_steps, remaining_capacity, item))
        if action not in (0, 1):
            raise ValueError("action must be 0 or 1")

        before = int(base.current_weight)
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        accepted += int(int(base.current_weight) > before)
        steps += 1
        if terminated or truncated:
            break

    return EpisodeResult(total_reward, accepted, int(base.current_weight), steps)


def _summarize(policy_name: str, results: Sequence[EpisodeResult]):
    rewards = np.array([x.reward for x in results], dtype=float)
    mean = float(rewards.mean())
    std = float(rewards.std(ddof=1))
    half = float(
        student_t.ppf(0.975, df=len(rewards) - 1) * std / math.sqrt(len(rewards))
    )
    summary = EvaluationSummary(
        policy_name,
        len(results),
        mean,
        std,
        mean - half,
        mean + half,
        float(np.mean([x.final_weight for x in results])),
        float(np.mean([x.accepted_items for x in results])),
    )
    return summary, rewards


def evaluate_policy(
    env,
    instance: KnapsackInstance,
    action_fn: ActionFn,
    *,
    policy_name: str,
    episode_seeds: Sequence[int],
):
    return _summarize(
        policy_name,
        [run_episode(env, instance, action_fn, seed=int(seed)) for seed in episode_seeds],
    )


def evaluate_random_feasible(
    env,
    instance: KnapsackInstance,
    *,
    episode_seeds: Sequence[int],
    action_seed: int = 991,
    accept_probability: float = 0.5,
):
    results = []
    for i, seed in enumerate(episode_seeds):
        rng = np.random.default_rng(action_seed + i)

        def action(_steps: int, capacity: int, item: int) -> int:
            if int(instance.weights[item]) > capacity:
                return 0
            return int(rng.random() < accept_probability)

        results.append(run_episode(env, instance, action, seed=int(seed)))
    return _summarize(f"random_feasible_p={accept_probability:.2f}", results)


def paired_difference_ci95(a: np.ndarray, b: np.ndarray):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean = float(d.mean())
    std = float(d.std(ddof=1))
    half = float(student_t.ppf(0.975, df=len(d) - 1) * std / math.sqrt(len(d)))
    return mean, mean - half, mean + half


def run_experiment(
    *, episodes: int = 3000, instance_seed: int = 20260827, evaluation_seed: int = 10000
) -> ExperimentResult:
    if episodes < 2:
        raise ValueError("episodes must be >= 2")
    instance = make_benchmark_instance(seed=instance_seed)
    exact = solve_exact_online_knapsack_dp(instance)
    threshold = optimize_static_threshold(instance)
    env = make_configured_env(instance)
    seeds = np.arange(evaluation_seed, evaluation_seed + episodes, dtype=np.int64)

    exact_mc, exact_rewards = evaluate_policy(
        env, instance, exact.action, policy_name="exact_dynamic_programming", episode_seeds=seeds
    )
    threshold_mc, threshold_rewards = evaluate_policy(
        env,
        instance,
        threshold_action_fn(instance, threshold.threshold),
        policy_name="optimized_static_density_threshold",
        episode_seeds=seeds,
    )
    always_mc, _ = evaluate_policy(
        env,
        instance,
        always_accept_feasible_action(instance),
        policy_name="always_accept_feasible",
        episode_seeds=seeds,
    )
    random_mc, _ = evaluate_random_feasible(
        env, instance, episode_seeds=seeds, action_seed=evaluation_seed + 500_000
    )
    env.close()

    diff = paired_difference_ci95(exact_rewards, threshold_rewards)
    return ExperimentResult(
        exact.optimal_expected_reward,
        threshold.threshold,
        threshold.expected_reward,
        exact_mc,
        threshold_mc,
        always_mc,
        random_mc,
        *diff,
    )


def self_test() -> None:
    simple = KnapsackInstance(
        np.array([2], dtype=np.int32),
        np.array([5], dtype=np.int32),
        np.array([1.0]),
        capacity=4,
        horizon=3,
    )
    dp = solve_exact_online_knapsack_dp(simple)
    assert math.isclose(dp.optimal_expected_reward, 10.0, abs_tol=1e-12)
    assert dp.action(3, 4, 0) == 0  # Bellman value tie
    assert dp.action(1, 2, 0) == 1

    option_value = KnapsackInstance(
        np.array([2, 2], dtype=np.int32),
        np.array([1, 10], dtype=np.int32),
        np.array([0.0, 1.0]),
        capacity=2,
        horizon=2,
    )
    assert solve_exact_online_knapsack_dp(option_value).action(2, 2, 0) == 0

    benchmark = make_benchmark_instance(seed=123, n_items=20, capacity=30, horizon=8)
    threshold = optimize_static_threshold(benchmark)
    brute = max(
        evaluate_static_threshold_exact(benchmark, float(x)).expected_reward
        for x in exact_threshold_candidates(benchmark)
    )
    assert math.isclose(threshold.expected_reward, brute, abs_tol=1e-10)
    exact = solve_exact_online_knapsack_dp(benchmark)
    assert exact.optimal_expected_reward + 1e-10 >= threshold.expected_reward

    a = make_benchmark_instance(seed=999)
    b = make_benchmark_instance(seed=999)
    assert np.array_equal(a.weights, b.weights)
    assert np.array_equal(a.values, b.values)
    assert np.array_equal(a.probabilities, b.probabilities)
    print("OR-Gymnasium online-knapsack dynamic-programming self-test: OK")


def _print(summary: EvaluationSummary) -> None:
    print(
        f"{summary.policy_name:<36} mean={summary.mean_reward:9.3f} "
        f"95%CI=[{summary.ci95_low:9.3f}, {summary.ci95_high:9.3f}] "
        f"weight={summary.mean_final_weight:7.2f} accepted={summary.mean_accepted_items:6.2f}"
    )


def print_experiment(result: ExperimentResult) -> None:
    print("=" * 104)
    print("OR-GYMNASIUM ONLINE KNAPSACK: EXACT STOCHASTIC DP POLICY OPTIMIZATION")
    print("=" * 104)
    print(f"Exact DP expected optimal reward       : {result.exact_expected_reward:.6f}")
    print(
        f"Best static value-density threshold    : {result.threshold:.6f} "
        f"(exact expected reward {result.threshold_expected_reward:.6f})"
    )
    print()
    for summary in (
        result.exact_mc,
        result.threshold_mc,
        result.always_accept_mc,
        result.random_mc,
    ):
        _print(summary)
    print()
    print(
        "Paired exact-DP minus threshold reward : "
        f"{result.exact_minus_threshold_mean:.3f} "
        f"[95% CI {result.exact_minus_threshold_ci95_low:.3f}, "
        f"{result.exact_minus_threshold_ci95_high:.3f}]"
    )
    print(
        "Exactness: DP is exact for the declared finite-horizon i.i.d. "
        "Knapsack-v3 model and injected benchmark catalogue."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--instance-seed", type=int, default=20260827)
    parser.add_argument("--evaluation-seed", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        print_experiment(
            run_experiment(
                episodes=args.episodes,
                instance_seed=args.instance_seed,
                evaluation_seed=args.evaluation_seed,
            )
        )
