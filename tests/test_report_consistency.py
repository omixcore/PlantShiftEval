import csv
import json
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SummaryConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads((ROOT / "report" / "summary.json").read_text(encoding="utf-8"))
        cls.robustness = json.loads((ROOT / "report" / "robustness_summary.json").read_text(encoding="utf-8"))
        cls.by_model = {row["model"]: row for row in cls.rows}

    def test_summary_matches_tracked_run_metrics(self):
        for model, row in self.by_model.items():
            paths = sorted((ROOT / "runs").glob(f"{model}_s*/metrics.json"))
            self.assertEqual(row["n_seeds"], len(paths), model)
            metrics = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
            self.assertAlmostEqual(row["test_acc"], statistics.mean(m["test_acc"] for m in metrics), places=4, msg=model)
            self.assertAlmostEqual(row["macro_f1"], statistics.mean(m["test_macro_f1"] for m in metrics), places=4, msg=model)
            expected_sd = statistics.stdev(m["test_macro_f1"] for m in metrics) if len(metrics) > 1 else 0.0
            self.assertAlmostEqual(row["macro_f1_sd"], expected_sd, places=4, msg=model)
            self.assertAlmostEqual(row["ece"], statistics.mean(m["ece"] for m in metrics), places=4, msg=model)

    def test_robustness_columns_use_multi_seed_summary(self):
        for model, row in self.by_model.items():
            record = self.robustness["per_model"][model]
            self.assertEqual(row["robustness_n_seeds"], record["n_seeds"], model)
            self.assertAlmostEqual(row["clean"], record["clean"]["mean"], places=4, msg=model)
            self.assertAlmostEqual(row["robust_acc"], record["robust_acc"]["mean"], places=4, msg=model)

    def test_csv_and_json_summaries_agree(self):
        with (ROOT / "report" / "summary.csv").open(encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        csv_by_model = {row["model"]: row for row in csv_rows}
        self.assertEqual(set(csv_by_model), set(self.by_model))
        for model, row in self.by_model.items():
            self.assertAlmostEqual(float(csv_by_model[model]["macro_f1"]), row["macro_f1"], places=4)
            self.assertAlmostEqual(float(csv_by_model[model]["clean"]), row["clean"], places=4)
            self.assertAlmostEqual(float(csv_by_model[model]["robust_acc"]), row["robust_acc"], places=4)


if __name__ == "__main__":
    unittest.main()
