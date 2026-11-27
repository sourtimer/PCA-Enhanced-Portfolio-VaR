"""Check alignment of forecasts used by the presentation figures."""
from __future__ import annotations
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fe_model import TABLE_DIR
from plot_presentation import verify_daily, filter_series

class PresentationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=TABLE_DIR/'monte_carlo'
        cls.daily=pd.read_csv(base/'all_methods_daily_predictions.csv')
        cls.summary=pd.read_csv(base/'all_methods_backtest_summary.csv')

    def test_no_future_data_and_correct_violation_flags(self):
        verify_daily(self.daily)

    def test_all_models_and_dates_present(self):
        df=self.daily
        self.assertEqual(len(df),35820)
        self.assertEqual(df.Date.nunique(),1194)
        self.assertEqual(df.method.nunique(),5)
        self.assertEqual(df.portfolio.nunique(),3)
        self.assertEqual(df.confidence.nunique(),2)
        self.assertTrue((df.groupby(['portfolio','method','confidence']).size()==1194).all())

    def test_summary_uses_actual_daily_data(self):
        df=self.daily
        for _,row in self.summary.iterrows():
            g=df.loc[(df.portfolio==row.portfolio)&(df.method==row.method)&np.isclose(df.confidence,row.confidence)]
            self.assertEqual(len(g),int(row.backtest_days))
            self.assertEqual(int(g.violation.sum()),int(row.breaches))
            self.assertAlmostEqual(float(g.violation.mean()),float(row.actual_breach_rate),places=10)

    def test_forecasts_have_repeated_observed_losses(self):
        g=self.daily.groupby(['Date','portfolio'])['actual_loss'].agg(['min','max'])
        self.assertLess((g['max']-g['min']).abs().max(),1e-9)

    def test_pca_and_parametric_difference_is_grounded(self):
        g=filter_series(self.daily,'EqualWeight',.99).pivot(index='Date',columns='method',values='VaR')
        diffs=(g.PCA_Factor-g.Parametric)*1e4
        self.assertTrue(np.isfinite(diffs).all())
        self.assertGreater(np.abs(diffs).sum(),0)

if __name__=='__main__': unittest.main()
