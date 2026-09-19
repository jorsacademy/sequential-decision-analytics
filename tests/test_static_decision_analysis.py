from pathlib import Path
import sys
import unittest

import numpy as np


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from static_decision_under_uncertainty import DecisionUnderUncertainty, investment_example


class StaticDecisionAnalysisTests(unittest.TestCase):
    def test_expected_values_risk_and_evpi_hand_check(self):
        model = investment_example()

        expected = model.expected_values()
        risk = model.outcome_risk()
        preferred, value = model.preferred_alternative()

        self.assertAlmostEqual(expected["Stock Portfolio"], 10.25)
        self.assertAlmostEqual(expected["Bond Portfolio"], 7.60)
        self.assertAlmostEqual(expected["Real Estate"], 11.10)
        self.assertAlmostEqual(expected["Startup Investment"], 13.75)
        self.assertAlmostEqual(risk["Real Estate"], 5.63826214360418)
        self.assertEqual(preferred, "Startup Investment")
        self.assertAlmostEqual(value, 13.75)
        self.assertAlmostEqual(model.value_of_perfect_information(), 7.90)

    def test_probability_perturbation_keeps_requested_state_probability(self):
        model = investment_example()
        revised = model._probabilities_with_fixed_state(0, 0.40)

        self.assertAlmostEqual(revised[0], 0.40)
        self.assertAlmostEqual(revised.sum(), 1.0)

        base_others = model.probabilities[1:]
        revised_others = revised[1:]
        np.testing.assert_allclose(
            revised_others / revised_others.sum(),
            base_others / base_others.sum(),
        )

    def test_sensitivity_records_are_valid_probability_vectors(self):
        model = investment_example()
        records = model.sensitivity_analysis(probability_shift=0.20, steps=7)

        self.assertEqual(len(records), 4 * 7)
        for record in records:
            probs = np.asarray(record.probabilities)
            self.assertAlmostEqual(probs.sum(), 1.0)
            self.assertTrue(np.all(probs >= 0.0))
            state_index = model.states_of_nature.index(record.varied_state)
            self.assertAlmostEqual(
                probs[state_index],
                record.varied_probability,
            )

        summary = model.sensitivity_summary(records)
        self.assertEqual(set(summary), set(model.states_of_nature))
        for metrics in summary.values():
            self.assertTrue(0.0 <= metrics["decision_stability_rate"] <= 1.0)
            self.assertGreaterEqual(metrics["preferred_value_range"], 0.0)

    def test_monte_carlo_mean_matches_analytic_expected_value(self):
        model = investment_example()
        _, stats = model.monte_carlo(200_000, seed=123)
        expected = model.expected_values()

        for alternative in model.alternatives:
            self.assertAlmostEqual(
                stats[alternative]["mean"],
                expected[alternative],
                delta=0.15,
            )

    def test_minimization_objective_and_evpi(self):
        model = DecisionUnderUncertainty(
            alternatives=("A", "B"),
            states_of_nature=("Low", "High"),
            probabilities=(0.5, 0.5),
            outcomes={
                "A": (2.0, 10.0),
                "B": (5.0, 6.0),
            },
            objective="minimize",
        )

        preferred, value = model.preferred_alternative()
        self.assertEqual(preferred, "B")
        self.assertAlmostEqual(value, 5.5)

        # Current expected cost is 5.5. With perfect information:
        # choose A in Low (2) and B in High (6), giving 4.0.
        self.assertAlmostEqual(model.value_of_perfect_information(), 1.5)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            DecisionUnderUncertainty(
                alternatives=("A",),
                states_of_nature=("S1", "S2"),
                probabilities=(0.2, 0.2),
                outcomes={"A": (1.0, 2.0)},
            )

        with self.assertRaises(ValueError):
            DecisionUnderUncertainty(
                alternatives=("A",),
                states_of_nature=("S1",),
                probabilities=(1.0,),
                outcomes={"A": (1.0,)},
                objective="median",
            )


if __name__ == "__main__":
    unittest.main()
