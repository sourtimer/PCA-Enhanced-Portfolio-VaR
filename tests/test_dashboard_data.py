"""Check data schemas, integrity hashes and backtest metrics."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import data_loader as dl  # noqa: E402


class FrozenResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictions = dl.load_predictions()
        cls.summary = dl.load_summary()

    # -- shape and schema -------------------------------------------------
    def test_prediction_shape_and_columns(self) -> None:
        self.assertEqual(len(self.predictions), dl.N_FORECASTS)
        self.assertEqual(list(self.predictions.columns), dl.PREDICTION_COLUMNS)

    def test_summary_shape_and_columns(self) -> None:
        self.assertEqual(len(self.summary), dl.N_SUMMARY_ROWS)
        self.assertEqual(list(self.summary.columns), dl.SUMMARY_COLUMNS)

    def test_every_slice_has_1194_days(self) -> None:
        grouped = self.predictions.groupby(
            ["portfolio", "confidence", "method"], observed=True
        ).size()
        self.assertEqual(len(grouped), 30)
        self.assertTrue((grouped == dl.N_PER_SLICE).all(), grouped.to_dict())

    def test_no_duplicate_forecast_keys(self) -> None:
        keys = ["Date", "portfolio", "confidence", "method"]
        self.assertEqual(int(self.predictions.duplicated(keys).sum()), 0)

    # -- enumerations and dates -------------------------------------------
    def test_enumerations(self) -> None:
        self.assertEqual(set(self.predictions["portfolio"]), set(dl.PORTFOLIOS))
        self.assertEqual(set(self.summary["portfolio"]), set(dl.PORTFOLIOS))
        self.assertEqual(set(self.predictions["method"]), set(dl.METHODS))
        self.assertEqual(set(self.summary["method"]), set(dl.METHODS))
        self.assertEqual(
            set(self.predictions["confidence"].round(4)), set(dl.CONFIDENCES)
        )
        self.assertEqual(set(self.summary["confidence"].round(4)), set(dl.CONFIDENCES))

    def test_date_window(self) -> None:
        start, end = dl.date_bounds(self.predictions)
        self.assertEqual(start, pd.Timestamp("2021-12-28"))
        self.assertEqual(end, pd.Timestamp("2026-09-30"))
        self.assertEqual(dl.n_unique_dates(self.predictions), 1_194)

    # -- accounting identities --------------------------------------------
    def test_violation_matches_loss_above_var(self) -> None:
        expected = (self.predictions["actual_loss"] > self.predictions["VaR"]).astype(int)
        mismatches = int((expected != self.predictions["violation"].astype(int)).sum())
        self.assertEqual(mismatches, 0)

    def test_no_look_ahead(self) -> None:
        bad = int(
            (self.predictions["training_end_date"] >= self.predictions["Date"]).sum()
        )
        self.assertEqual(bad, 0)

    def test_no_missing_key_values(self) -> None:
        key_cols = ["Date", "portfolio", "method", "confidence", "VaR", "actual_loss"]
        self.assertFalse(self.predictions[key_cols].isna().any().any())
        self.assertFalse(self.summary.isna().any().any())

    def test_pca_fields_only_for_pca_method(self) -> None:
        pca = self.predictions.loc[self.predictions["method"] == "PCA_Factor"]
        other = self.predictions.loc[self.predictions["method"] != "PCA_Factor"]
        self.assertFalse(pca["pca_components"].isna().any())
        self.assertTrue(other["pca_components"].isna().all())

    # -- summary reconciliation -------------------------------------------
    def test_summary_reconciles_to_daily_rows(self) -> None:
        grouped = self.predictions.groupby(
            ["portfolio", "confidence", "method"], observed=True
        ).agg(
            days=("violation", "size"),
            breaches=("violation", "sum"),
        )
        for row in self.summary.itertuples(index=False):
            key = (row.portfolio, round(float(row.confidence), 4), row.method)
            cell = grouped.loc[key]
            self.assertEqual(int(row.backtest_days), int(cell["days"]), key)
            self.assertEqual(int(row.breaches), int(cell["breaches"]), key)
            self.assertAlmostEqual(
                float(row.actual_breach_rate),
                float(cell["breaches"]) / float(cell["days"]),
                places=6,
                msg=str(key),
            )

    def test_kupiec_p_value_matches_official_summary(self) -> None:
        worst = 0.0
        for row in self.summary.itertuples(index=False):
            lr, p = dl.kupiec_test(
                int(row.backtest_days), int(row.breaches), float(row.confidence)
            )
            worst = max(worst, abs(lr - float(row.Kupiec_LR)))
            worst = max(worst, abs(p - float(row.Kupiec_p_value)))
        self.assertLess(worst, 1e-9)

    def test_metrics_for_single_slice_matches_summary(self) -> None:
        row = self.summary.iloc[0]
        sl = dl.slice_predictions(
            self.predictions,
            row["portfolio"],
            float(row["confidence"]),
            row["method"],
        )
        metrics = dl.metrics_for(sl, float(row["confidence"]))
        self.assertEqual(metrics["n"], int(row["backtest_days"]))
        self.assertEqual(metrics["breaches"], int(row["breaches"]))
        self.assertAlmostEqual(metrics["actual_rate"], float(row["actual_breach_rate"]), 6)
        self.assertAlmostEqual(metrics["kupiec_p"], float(row["Kupiec_p_value"]), 10)

    # -- PCA evidence ------------------------------------------------------
    def test_full_sample_pca_seven_components(self) -> None:
        variance = dl.load_pca_variance()
        cumulative = variance["cumulative_ratio"].astype(float)
        self.assertAlmostEqual(float(cumulative.iloc[6]), 0.9074241897351636, 12)
        self.assertAlmostEqual(float(variance["explained_ratio"].sum()), 1.0, 9)

    def test_correlation_matrix_shape_and_diagonal(self) -> None:
        corr, labels = dl.load_correlation()
        self.assertEqual(corr.shape, (18, 18))
        self.assertEqual(len(labels), 18)
        self.assertTrue((corr.values.diagonal() == 1.0).all())
        self.assertTrue((corr.values <= 1.0 + 1e-12).all())
        self.assertTrue((corr.values >= -1.0 - 1e-12).all())

    def test_rolling_pca_k_and_variance(self) -> None:
        diag = dl.load_pca_diagnostics()
        self.assertEqual(len(diag), 3_582)
        self.assertEqual(set(diag["pca_components"].unique()), {6, 7, 8})
        self.assertTrue((diag["pca_components"] >= 6).all())
        self.assertTrue((diag["explained_variance"] >= 0.90 - 1e-9).all())
        self.assertTrue(dl.rolling_pca_identical(diag))

    def test_sensitivity_only_fixed_k(self) -> None:
        daily = dl.load_pca_sensitivity_daily()
        summary = dl.load_pca_sensitivity_summary()
        self.assertEqual(len(daily), 21_492)
        self.assertEqual(set(daily["method"]), set(dl.PCA_SENSITIVITY_METHODS))
        self.assertEqual(len(summary), 18)
        counts = daily.groupby("method", observed=True).size()
        self.assertTrue((counts == 7_164).all(), counts.to_dict())

    # -- immutability ------------------------------------------------------
    def test_source_files_are_byte_identical(self) -> None:
        baseline = json.loads(
            (ROOT / "tests" / "baseline_sha256.json").read_text(encoding="utf-8")
        )
        for relative, expected in baseline.items():
            path = ROOT / relative
            self.assertTrue(path.exists(), relative)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, expected, f"{relative} was modified")

    # -- integration sanity ------------------------------------------------
    def test_integrity_report_is_green(self) -> None:
        report = dl.verify_integrity(self.predictions, self.summary)
        failures = [c for c in report if not c["ok"]]
        self.assertEqual(failures, [])

    def test_comparison_table_has_five_methods(self) -> None:
        table = dl.comparison_table(self.summary, "EqualWeight", 0.99)
        self.assertEqual(len(table), 5)
        self.assertEqual(set(table["method"]), set(dl.METHODS))
        self.assertTrue((table["Breaches"] >= 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
