"""판정 검증 계산 확인: python -m unittest discover -s tests"""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import evaluation  # noqa: E402
from utils.preprocessing import numeric_features, read_csv  # noqa: E402


class Metrics(unittest.TestCase):
    def test_counts_and_rates(self):
        actual = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        predicted = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
        result = evaluation.metrics(actual, predicted)
        self.assertEqual((result["TP"], result["FN"], result["FP"], result["TN"]), (3, 1, 1, 5))
        self.assertAlmostEqual(result["재현율"]["값"], .75)
        self.assertAlmostEqual(result["오경보율"]["값"], 1 / 6)
        self.assertAlmostEqual(result["정밀도"]["값"], .75)

    def test_single_class_is_not_scored_as_zero(self):
        result = evaluation.metrics([0, 0, 0], [0, 1, 0])
        self.assertTrue(math.isnan(result["재현율"]["값"]))
        self.assertTrue(math.isnan(result["MCC"]))
        self.assertAlmostEqual(result["오경보율"]["값"], 1 / 3)
        self.assertTrue(any("실제 이상 행이 없어" in note for note in evaluation.sample_warnings(result)))

    def test_average_precision_matches_perfect_ranking(self):
        curve, average_precision = evaluation.pr_curve([0, 0, 1, 1], [.1, .2, .8, .9])
        self.assertAlmostEqual(average_precision, 1.0)
        self.assertEqual(curve["재현율"].iloc[-1], 1.0)


class Labels(unittest.TestCase):
    def test_case_and_space_are_ignored(self):
        encoded = evaluation.encode_labels(pd.Series(["Normal ", "normal", "ANOMALY"]), "normal", "anomaly")
        self.assertEqual(encoded.tolist(), [0, 0, 1])

    def test_unknown_label_is_reported(self):
        with self.assertRaisesRegex(ValueError, "NA"):
            evaluation.encode_labels(pd.Series(["정상", "이상", "NA"]), "정상", "이상")

    def test_guess(self):
        self.assertEqual(evaluation.guess_labels(["이상", "정상"]), ("정상", "이상"))


class Ranges(unittest.TestCase):
    def setUp(self):
        self.ranges = evaluation.feature_ranges(pd.DataFrame({"f": np.linspace(-6, -4, 200)}))

    def test_outside_rows(self):
        outside, summary = evaluation.range_check(pd.DataFrame({"f": [-5, -3, -7]}), self.ranges)
        self.assertEqual(outside.tolist(), [0, 1, 1])
        self.assertEqual(int(summary["범위 밖 행"].iloc[0]), 2)

    def test_linear_instead_of_log_hint(self):
        _, summary = evaluation.range_check(pd.DataFrame({"f": [1e-5, 2e-5]}), self.ranges)
        self.assertIn("log10", summary["확인할 점"].iloc[0])

    def test_khz_vs_hz_hint(self):
        ranges = evaluation.feature_ranges(pd.DataFrame({"freq": np.linspace(30, 40, 50)}))
        _, summary = evaluation.range_check(pd.DataFrame({"freq": [35_000.0, 36_000.0]}), ranges)
        self.assertIn("1000배", summary["확인할 점"].iloc[0])

    def test_training_overlap(self):
        train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        hashes = evaluation.training_hashes(train)
        check = pd.DataFrame({"a": [1.0, 9.0], "b": [3.0, 4.0]})
        self.assertEqual(evaluation.overlap_with_training(check, hashes).tolist(), [True, False])


class Csv(unittest.TestCase):
    def test_header_spaces_and_na_text(self):
        frame = read_csv(" f , label\n1,NA\n2,정상\n".encode())
        self.assertEqual(list(frame.columns), ["f", "label"])
        self.assertEqual(frame["label"].tolist(), ["NA", "정상"])

    def test_bad_number_shows_example(self):
        frame = read_csv('f\n"1,234"\n2\n'.encode())
        with self.assertRaisesRegex(ValueError, "1,234"):
            numeric_features(frame, ["f"])


if __name__ == "__main__":
    unittest.main()
