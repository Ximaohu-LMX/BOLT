import unittest

from src.CLIQUE_bound import CLIQUEestimator_bound


class _BoundStub:
    def get_bound(self, _query):
        return 17

    def get_lower_bound_for_query(self, _query):
        return 0


class LowerBoundSemanticsTest(unittest.TestCase):
    def setUp(self):
        self.model = _BoundStub()

    def effective_lb(self, query, upper_bound=None, raw_lower_bound=None):
        return CLIQUEestimator_bound.get_effective_lower_bound_for_query(
            self.model, query, upper_bound, raw_lower_bound
        )

    def test_library_miss_remains_zero(self):
        self.assertEqual(self.effective_lb("%missing%", 17, 0), 0)

    def test_zero_lb_uses_same_bounded_head_as_clique(self):
        upper_bound = 17
        sigmoid_output = 0.25
        lower_bound = self.effective_lb("%missing%", upper_bound, 0)
        bounded_prediction = lower_bound + (
            upper_bound - lower_bound
        ) * sigmoid_output
        clique_prediction = upper_bound * sigmoid_output
        self.assertEqual(bounded_prediction, clique_prediction)

    def test_positive_lb_is_clamped_to_upper_bound(self):
        self.assertEqual(self.effective_lb("%known%", 17, 23), 17)

    def test_all_percent_query_keeps_exact_upper_bound(self):
        self.assertEqual(self.effective_lb("%%%", 17, 0), 17)

    def test_negative_raw_lb_cannot_escape_model_range(self):
        self.assertEqual(self.effective_lb("%missing%", 17, -1), 0)

    def test_force_zero_lb_overrides_all_percent_exact_bound(self):
        self.model.force_zero_lower_bound = True
        self.assertEqual(self.effective_lb("%%%", 17, 17), 0)


if __name__ == "__main__":
    unittest.main()
