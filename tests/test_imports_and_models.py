from contextlib import ExitStack
import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPERIMENTS = ["LSTMtrain", "LSTMtrain1", "LSTMtrain2", "LSTMPrediction",
               "LSTMReadAndTest", "Seq2seqTrain", "RNNtrain", "RNNtrainNew", "MPLtrain"]


class ImportAndModelTests(unittest.TestCase):
    def test_importing_all_experiments_starts_no_data_or_output_operation(self):
        with ExitStack() as stack:
            for owner, names in [(pd, ["read_csv", "read_excel", "ExcelFile"]),
                                 (torch, ["save", "load"]), (plt, ["show", "savefig"])]:
                for name in names:
                    stack.enter_context(patch.object(owner, name, side_effect=AssertionError("experiment ran on import")))
            for name in ["DataSet", "DataSet1"] + EXPERIMENTS:
                module = importlib.import_module("LSTMTimeSeries." + name)
                self.assertFalse(hasattr(module, "model"), name)

    def test_imported_rnns_support_their_own_dimensions_and_input_dtype(self):
        for name in ["RNNtrain", "RNNtrainNew"]:
            cls = importlib.import_module("LSTMTimeSeries." + name).RNNModel
            for dtype in [torch.float32, torch.float64]:
                for layers in [1, 2]:
                    with self.subTest(module=name, dtype=dtype, layers=layers):
                        model = cls(input_size=3, hidden_size=4, output_size=2, num_layers=layers).to(dtype=dtype)
                        result = model(torch.zeros(2, 5, 3, dtype=dtype))
                        self.assertEqual(tuple(result.shape), (2, 2))
                        self.assertEqual(result.dtype, dtype)
                        self.assertEqual(result.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
