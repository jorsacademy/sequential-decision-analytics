from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class SensitivityRecord:
    varied_state: str
    varied_probability: float
    probabilities: tuple[float, ...]
    preferred_alternative: str
    preferred_expected_value: float


class DecisionUnderUncertainty:
    """Finite-state static decision analysis under known scenario probabilities."""

    def __init__(
        self,
        alternatives,
        states_of_nature,
        probabilities,
        outcomes,
        *,
        objective="maximize",
    ):
        self.alternatives = tuple(alternatives)
        self.states_of_nature = tuple(states_of_nature)
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.objective = str(objective).lower()

        if not self.alternatives or not self.states_of_nature:
            raise ValueError("alternatives and states_of_nature must be non-empty")
        if len(set(self.alternatives)) != len(self.alternatives):
            raise ValueError("alternatives must be unique")
        if len(set(self.states_of_nature)) != len(self.states_of_nature):
            raise ValueError("states_of_nature must be unique")
        if self.objective not in {"maximize", "minimize"}:
            raise ValueError("objective must be 'maximize' or 'minimize'")
        if self.probabilities.shape != (len(self.states_of_nature),):
            raise ValueError("probability count must match states_of_nature")
        if not np.isfinite(self.probabilities).all() or np.any(self.probabilities < 0):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.isclose(self.probabilities.sum(), 1.0):
            raise ValueError("probabilities must sum to 1")

        rows = []
        for alternative in self.alternatives:
            if alternative not in outcomes:
                raise ValueError(f"missing outcomes for alternative: {alternative}")
            row = np.asarray(outcomes[alternative], dtype=float)
            if row.shape != (len(self.states_of_nature),):
                raise ValueError(
                    f"outcomes for {alternative} must match states_of_nature"
                )
            if not np.isfinite(row).all():
                raise ValueError("outcomes must be finite")
            rows.append(row)

        self.outcomes = np.vstack(rows)

    def expected_values(self) -> dict[str, float]:
        values = self.outcomes @ self.probabilities
        return {
            alternative: float(value)
            for alternative, value in zip(self.alternatives, values)
        }

    def preferred_alternative(self) -> tuple[str, float]:
        values = self.outcomes @ self.probabilities
        index = int(np.argmax(values) if self.objective == "maximize" else np.argmin(values))
        return self.alternatives[index], float(values[index])

    def outcome_risk(self) -> dict[str, float]:
        expected = self.outcomes @ self.probabilities
        variance = ((self.outcomes - expected[:, None]) ** 2) @ self.probabilities
        return {
            alternative: float(np.sqrt(value))
            for alternative, value in zip(self.alternatives, variance)
        }

    def decision_matrix(self) -> np.ndarray:
        """Return alternatives x states outcomes as a copy."""
        return self.outcomes.copy()

    def monte_carlo(self, num_simulations=10_000, *, seed=42):
        if num_simulations < 1:
            raise ValueError("num_simulations must be positive")

        rng = np.random.default_rng(seed)
        states = rng.choice(
            len(self.states_of_nature),
            size=int(num_simulations),
            p=self.probabilities,
        )
        samples = self.outcomes[:, states]

        stats = {}
        for i, alternative in enumerate(self.alternatives):
            values = samples[i]
            stats[alternative] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std(ddof=0)),
                "5th_percentile": float(np.percentile(values, 5)),
                "95th_percentile": float(np.percentile(values, 95)),
            }
        return samples, stats

    def value_of_perfect_information(self) -> float:
        current = self.outcomes @ self.probabilities
        if self.objective == "maximize":
            current_value = float(current.max())
            perfect_by_state = self.outcomes.max(axis=0)
            return float(self.probabilities @ perfect_by_state - current_value)

        current_value = float(current.min())
        perfect_by_state = self.outcomes.min(axis=0)
        return float(current_value - self.probabilities @ perfect_by_state)

    def _probabilities_with_fixed_state(self, state_index, new_probability):
        if not 0.0 <= new_probability <= 1.0:
            raise ValueError("new_probability must be in [0, 1]")

        base = self.probabilities
        revised = np.zeros_like(base)
        revised[state_index] = float(new_probability)

        other_mask = np.arange(len(base)) != state_index
        other_total = float(base[other_mask].sum())
        remaining = 1.0 - float(new_probability)

        if other_total > 0:
            revised[other_mask] = base[other_mask] / other_total * remaining
        elif remaining > 1e-12:
            raise ValueError(
                "cannot redistribute probability because all other states have zero mass"
            )

        return revised

    def sensitivity_analysis(self, probability_shift=0.20, steps=11):
        """Perturb one state probability while preserving relative weights of others."""
        if not 0.0 <= probability_shift <= 1.0:
            raise ValueError("probability_shift must be in [0, 1]")
        if steps < 2:
            raise ValueError("steps must be at least two")

        records = []
        half_range = probability_shift / 2.0

        for state_index, base_probability in enumerate(self.probabilities):
            lower = max(0.0, float(base_probability) - half_range)
            upper = min(1.0, float(base_probability) + half_range)

            for value in np.linspace(lower, upper, int(steps)):
                revised = self._probabilities_with_fixed_state(state_index, float(value))
                expected = self.outcomes @ revised
                preferred_index = int(
                    np.argmax(expected)
                    if self.objective == "maximize"
                    else np.argmin(expected)
                )
                records.append(
                    SensitivityRecord(
                        varied_state=self.states_of_nature[state_index],
                        varied_probability=float(value),
                        probabilities=tuple(float(p) for p in revised),
                        preferred_alternative=self.alternatives[preferred_index],
                        preferred_expected_value=float(expected[preferred_index]),
                    )
                )

        return tuple(records)

    def sensitivity_summary(self, records=None):
        if records is None:
            records = self.sensitivity_analysis()
        if not records:
            raise ValueError("records must be non-empty")

        baseline, _ = self.preferred_alternative()
        summary = {}

        for state in self.states_of_nature:
            state_records = [r for r in records if r.varied_state == state]
            values = np.asarray(
                [r.preferred_expected_value for r in state_records], dtype=float
            )
            preferred = [r.preferred_alternative for r in state_records]
            switches = sum(name != baseline for name in preferred)
            summary[state] = {
                "preferred_value_range": float(values.max() - values.min()),
                "baseline_change_count": int(switches),
                "decision_stability_rate": float(1.0 - switches / len(state_records)),
                "preferred_alternatives": tuple(sorted(set(preferred))),
            }

        return summary

    def visualize_outcomes(self, num_simulations=10_000, *, seed=42):
        samples, _ = self.monte_carlo(num_simulations, seed=seed)

        plt.figure(figsize=(11, 6))
        for i, alternative in enumerate(self.alternatives):
            plt.hist(samples[i], bins=30, alpha=0.4, label=alternative)
        plt.title("Simulated Outcome Distribution by Alternative")
        plt.xlabel("Outcome")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def investment_example():
    alternatives = (
        "Stock Portfolio",
        "Bond Portfolio",
        "Real Estate",
        "Startup Investment",
    )
    states = ("Strong Economy", "Stable Economy", "Weak Economy", "Recession")
    probabilities = (0.25, 0.40, 0.25, 0.10)
    outcomes = {
        "Stock Portfolio": (28, 15, -5, -15),
        "Bond Portfolio": (10, 8, 6, 4),
        "Real Estate": (18, 12, 8, -2),
        "Startup Investment": (45, 20, -10, -30),
    }
    return DecisionUnderUncertainty(
        alternatives,
        states,
        probabilities,
        outcomes,
        objective="maximize",
    )


def main():
    model = investment_example()
    expected = model.expected_values()
    risk = model.outcome_risk()
    preferred, preferred_ev = model.preferred_alternative()

    print("Static decision analysis under economic-state uncertainty")
    print()
    print("Alternative                 Expected value    Std. deviation")
    for alternative in model.alternatives:
        print(
            f"{alternative:<27}"
            f"{expected[alternative]:>14.2f}%"
            f"{risk[alternative]:>18.2f}%"
        )

    print()
    print(f"Expected-value choice: {preferred} ({preferred_ev:.2f}%)")
    print(f"Expected value of perfect information: {model.value_of_perfect_information():.2f}%")

    # Finance-specific diagnostic kept outside the generic model.
    risk_free_rate = 2.0
    print()
    print("Sharpe-like risk-adjusted return diagnostic:")
    for alternative in model.alternatives:
        ratio = (
            (expected[alternative] - risk_free_rate) / risk[alternative]
            if risk[alternative] > 0
            else np.inf
        )
        print(f"{alternative:<27}{ratio:>8.3f}")

    _, stats = model.monte_carlo(100_000, seed=42)
    print()
    print("Monte Carlo check:")
    for alternative in model.alternatives:
        print(
            f"{alternative:<27}"
            f"mean={stats[alternative]['mean']:>7.2f}%  "
            f"p05={stats[alternative]['5th_percentile']:>7.2f}%  "
            f"p95={stats[alternative]['95th_percentile']:>7.2f}%"
        )

    sensitivity = model.sensitivity_analysis(probability_shift=0.20, steps=11)
    summary = model.sensitivity_summary(sensitivity)
    print()
    print("Probability sensitivity:")
    for state, metrics in summary.items():
        print(
            f"{state:<18}"
            f"value range={metrics['preferred_value_range']:>7.3f}  "
            f"stability={100.0 * metrics['decision_stability_rate']:>6.1f}%  "
            f"choices={', '.join(metrics['preferred_alternatives'])}"
        )

    # This frequency is a scenario-stability diagnostic, not a confidence level.
    counts = {alternative: 0 for alternative in model.alternatives}
    for record in sensitivity:
        counts[record.preferred_alternative] += 1

    stable_choice = max(counts, key=counts.get)
    stability_share = counts[stable_choice] / len(sensitivity)

    print()
    print(
        "Most frequently preferred alternative across sensitivity scenarios: "
        f"{stable_choice}"
    )
    print(f"Sensitivity-scenario share: {100.0 * stability_share:.1f}%")
    print("This share is not a statistical confidence level.")


if __name__ == "__main__":
    main()
