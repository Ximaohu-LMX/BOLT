import unittest

from src import util


class UnlabeledTrainingGuardTest(unittest.TestCase):
    def test_supervised_training_rejects_negative_card(self):
        with self.assertRaisesRegex(ValueError, "unlabeled rows"):
            util.assert_no_unlabeled_training_targets(
                [("%known%", 3), ("%unknown%", -1)], "test training"
            )

    def test_supervised_training_accepts_zero_card(self):
        util.assert_no_unlabeled_training_targets(
            [("%empty%", 0), ("%known%", 3)]
        )

    def test_packed_targets_are_checked(self):
        with self.assertRaises(ValueError):
            util.assert_no_unlabeled_training_targets(
                [("%packed%", [1, -1, 3])]
            )

    def test_lambda_zero_fallback_keeps_only_labeled_rows(self):
        rows = [("a%", 5), ("b%", -1), ("c%", 0)]
        self.assertEqual(
            util.labeled_training_rows(rows), [("a%", 5), ("c%", 0)]
        )


if __name__ == "__main__":
    unittest.main()
