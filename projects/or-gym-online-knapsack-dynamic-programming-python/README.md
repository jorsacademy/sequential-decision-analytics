# OR-Gym Online Knapsack Dynamic Programming Optimization

An exact stochastic dynamic-programming optimizer and benchmark for the online knapsack environment originating in OR-Gym.

The project uses the modern Gymnasium-compatible `or-gymnasium` adaptation rather than forcing the legacy `or-gym` package onto a current Python stack.

## Why `or-gymnasium`

The original OR-Gym project was designed to bridge operations research and reinforcement learning through Gym environments. Its latest PyPI release is `0.5.0` from September 2022, and its package metadata pins the legacy OpenAI Gym line (`gym<=0.21.0`).

The current `JGIoA/or-gymnasium` adaptation updates the environments to the Gymnasium API. At the pinned upstream commit used by this repository it declares Python `>=3.10`, Gymnasium `>=1.2.0`, and registers the classic OR-Gym environments including `Knapsack-v3`.

This repository therefore preserves the OR-Gym problem semantics while using the current Gymnasium interface.

## Problem: `Knapsack-v3`

`Knapsack-v3` is the stochastic online knapsack environment.

At each decision step:

1. one item type is sampled;
2. its weight and value are revealed;
3. the decision maker chooses `reject=0` or `accept=1`;
4. an accepted item earns its value if it fits;
5. the process continues until the horizon is reached or capacity is exactly filled.

The default OR-Gymnasium environment has:

```text
capacity = 200
horizon  = 50 decisions
item types = 200
```

Item arrivals are sampled with replacement according to the environment's `item_probs`.

## Exact finite-horizon dynamic program

Because the current item is observed before the action and the next item is sampled independently, the environment can be solved exactly as a finite-horizon stochastic dynamic program.

Define:

```text
V[t,c]
=
optimal expected future reward
before the next item is revealed

t = decisions remaining
c = remaining capacity
```

For a revealed item `j`:

```text
reject = V[t-1,c]

accept =
value[j] + V[t-1,c-weight[j]]
```

when the item fits.

Therefore:

```text
V[t,c]
=
sum_j p[j] * max(reject, accept_j)
```

with `V[0,c] = 0`.

The benchmark instance requires only:

```text
51 * 201 = 10,251
```

aggregate DP states (`t,c`), while item revelation is integrated exactly inside each Bellman backup.

## Deterministic benchmark catalogue

The upstream environment generates its item catalogue during object construction. To make repository results reproducible across machines, this project creates its own deterministic catalogue from a fixed NumPy seed and injects only:

- item weights;
- item values;
- item arrival probabilities;
- capacity;
- horizon.

The OR-Gymnasium environment still controls episode state transitions, rewards, action masks, termination and item sampling.

Default benchmark seed:

```text
20260827
```

For that deterministic benchmark, the pure DP calculation gives:

```text
exact optimal expected reward = 688.347948
```

## Optimized static heuristic baseline

A second optimizer searches the complete set of distinct static value-density threshold rules:

```text
accept item j iff

value[j] / weight[j] >= lambda
```

and the item fits.

Every distinct threshold-induced accept/reject pattern is evaluated with its own exact expectation recursion.

Default benchmark result:

```text
best lambda                     = 1.749480
exact expected threshold reward = 661.625317
exact DP advantage              = 26.722630
```

The static threshold is optimized exactly within that restricted policy class. It is not the optimal adaptive policy.

## OR-Gymnasium lifecycle audit

Researching the current `Knapsack-v3` implementation exposed an evaluation issue worth handling explicitly.

`OnlineKnapsackEnv.step()` increments `step_counter`, but the inherited reset path does not reset that counter. Reusing one environment across episodes can therefore cause later episodes to truncate immediately.

This repository does not modify the third-party package. `safe_reset_online_env()` restores the episode counter before calling `reset()`, and an integration regression test verifies the behavior.

The project also avoids `OnlineKnapsackEnv.sample_action()` for reproducibility because that helper uses NumPy's global RNG rather than the environment RNG.

## Monte Carlo validation

The exact DP value is a model expectation, not a Monte Carlo estimate.

The environment runner independently replays the DP policy over many Gymnasium episodes. CI checks that the empirical mean remains within a deliberately broad six-standard-error guard around the exact Bellman value.

The same episode seeds are used when comparing DP and heuristic policies so paired differences use Common Random Numbers.

## Baselines

The experiment reports:

- exact dynamic-programming policy;
- optimized static value-density threshold;
- always-accept-if-feasible heuristic;
- seeded random feasible acceptance.

No RL method is claimed to outperform the exact DP oracle. The purpose of the repo is to show how an OR-Gym environment can be solved and benchmarked with a classical exact OR method.

## Validated GitHub Actions run

The first full GitHub Actions run used CPython 3.12.14, installed `or-gymnasium` directly from pinned upstream commit `86ed6e7b5d9c12e9f7767073b58f9ba814ac4f9a`, and passed the pure DP self-test plus all nine regression/integration tests.

The integration suite verified:

- actual `Knapsack-v3` registration through Gymnasium;
- the online action mask;
- the `step_counter` safe-reset workaround;
- feasibility of a DP-controlled environment episode;
- Monte Carlo consistency with the exact Bellman expectation.

The 1000-episode smoke experiment produced:

```text
exact Bellman expected reward          688.347948
best static threshold                    1.749480
static threshold exact expectation     661.625317

exact DP Monte Carlo mean              686.074
95% CI                           [677.876, 694.272]

optimized static threshold mean        660.634
95% CI                           [652.118, 669.150]

always accept if feasible mean         319.754
random feasible p=0.50 mean            273.592

paired DP - threshold mean difference   25.440
95% paired CI                     [22.245, 28.635]
```

The Monte Carlo figures are finite-sample environment evaluations. The `688.347948` DP value and `661.625317` threshold value are exact expectations for the declared benchmark model.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Self-test:

```bash
python or_gym_online_knapsack_dp.py --self-test
```

Full environment experiment:

```bash
python or_gym_online_knapsack_dp.py --episodes 3000
```

Tests:

```bash
python -m unittest discover -s tests -v
```

## Exactness and scope

The dynamic program is exact for the declared finite-horizon i.i.d. online-knapsack model, the injected deterministic catalogue, and the OR-Gymnasium transition semantics.

It does not establish optimality for:

- non-i.i.d. item arrivals;
- hidden/nonstationary item distributions;
- multi-resource knapsack variants;
- continuous capacity;
- arbitrary OR-Gym environments.

The project is an OR/RL benchmark integration, not evidence that dynamic programming is the preferred method for every environment in OR-Gym.
