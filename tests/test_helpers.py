import ast
import datetime
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LSTMTimeSeries"))
from paths import feature_csv, training_workbook
from io_helpers import str_to_list
from windowing import iter_window_samples


class HelperTests(unittest.TestCase):
    def test_window_order_and_training_tail_are_preserved(self):
        for tail in [0, 200]:
            x, y = list(range(220)), list(range(1000, 1220))
            actual = list(iter_window_samples(x, y, 5, tail))
            expected = [(x[i:i + 5], y[i + 5]) for i in range(len(x) - 5 - tail)]
            self.assertEqual(actual, expected)

    def test_window_errors_are_explicit(self):
        for x, y, window, tail in [([1], [1], 2, 0), ([1, 2], [1], 1, 0), ([1, 2], [1, 2], 0, 0)]:
            with self.subTest(window=window, tail=tail):
                with self.assertRaises(ValueError):
                    list(iter_window_samples(x, y, window, tail))

    def test_paths_share_an_explicit_external_data_root(self):
        with patch.dict(os.environ, {"ML_TIMESERIES_DATA": "fixture-data"}):
            self.assertEqual(Path(training_workbook("12")), Path("fixture-data/train/M012.xlsx"))
            self.assertEqual(Path(training_workbook("12", True)), Path("fixture-data/train/M012_normalized.xlsx"))
            self.assertEqual(Path(feature_csv("12")), Path("fixture-data/NewDataM012.csv"))

    def test_vector_literals_do_not_execute_code(self):
        self.assertEqual(str_to_list("[1, 2.5, 3]"), [1, 2.5, 3])
        with self.assertRaises((ValueError, SyntaxError)):
            str_to_list("__import__('os').getcwd()")

    def test_generating_input1_scales_numpy_array_and_builds_windows(self):
        # Load only the two pure definitions: never import or run a research entry point.
        source = (ROOT / "LSTMTimeSeries/DataSet.py").read_text(encoding="utf-8")
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                 and n.name in ["custom_minmax_scaler", "GeneratingInput1"]]
        namespace = {"np": np, "pd": pd,
                     "DealWithOutliers": lambda _: pd.DataFrame({"放电容量/Ah": [1., 2., 3., 4., 5.]})}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "DataSet.py", "exec"), namespace)
        data = namespace["GeneratingInput1"](2)
        self.assertEqual(len(data), 3)
        self.assertEqual(data.iloc[0]["x"], [[100.], [125.]])
        self.assertEqual(data.iloc[-1]["y"].tolist(), [200.])
        np.testing.assert_array_equal(namespace["custom_minmax_scaler"](np.array([2., 2.]), 100, 200), [100., 100.])


if __name__ == "__main__":
    unittest.main()
