import unittest

from manual_tests.beam_sweep_experiment import add_derived_metrics, render_markdown


class BeamSweepExperimentTests(unittest.TestCase):
    def test_derived_metrics_measure_quality_and_runtime_against_smallest_width(self):
        results = [
            {
                "status": "ok",
                "beam_width": 2,
                "candidate_width": 2,
                "makespan": 500,
                "planning_seconds": 10.0,
            },
            {
                "status": "ok",
                "beam_width": 4,
                "candidate_width": 4,
                "makespan": 450,
                "planning_seconds": 20.0,
            },
        ]

        add_derived_metrics(results)

        self.assertEqual(results[1]["timesteps_saved_vs_baseline"], 50)
        self.assertAlmostEqual(results[1]["makespan_improvement_pct_vs_baseline"], 10.0)
        self.assertAlmostEqual(results[1]["runtime_multiplier_vs_baseline"], 2.0)
        self.assertEqual(results[1]["timesteps_saved_vs_previous"], 50)
        self.assertAlmostEqual(results[1]["timesteps_saved_per_extra_second"], 5.0)

    def test_dominated_quality_compute_point_is_not_pareto_optimal(self):
        results = [
            {
                "status": "ok",
                "beam_width": 2,
                "candidate_width": 2,
                "makespan": 500,
                "planning_seconds": 10.0,
            },
            {
                "status": "ok",
                "beam_width": 4,
                "candidate_width": 4,
                "makespan": 480,
                "planning_seconds": 20.0,
            },
            {
                "status": "ok",
                "beam_width": 8,
                "candidate_width": 8,
                "makespan": 490,
                "planning_seconds": 30.0,
            },
        ]

        add_derived_metrics(results)

        self.assertTrue(results[0]["pareto_optimal"])
        self.assertTrue(results[1]["pareto_optimal"])
        self.assertFalse(results[2]["pareto_optimal"])

    def test_markdown_explicitly_calls_sweep_joint_beam_candidate_experiment(self):
        results = [
            {
                "status": "ok",
                "beam_width": 2,
                "candidate_width": 2,
                "makespan": 500,
                "planning_seconds": 10.0,
                "astar_expansions": 100,
            },
            {
                "status": "ok",
                "beam_width": 4,
                "candidate_width": 4,
                "makespan": 450,
                "planning_seconds": 20.0,
                "astar_expansions": 200,
            },
        ]
        add_derived_metrics(results)

        report = render_markdown(results, orders=50, robots=5, candidate_cap=30_000)

        self.assertIn("joint search-breadth experiment", report)
        self.assertIn("not a pure beam-width ablation", report)
        self.assertIn("Best schedule", report)
        self.assertIn("10.00%", report)


if __name__ == "__main__":
    unittest.main()
