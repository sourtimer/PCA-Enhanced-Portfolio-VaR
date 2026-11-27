"""Run using: python -m unittest discover -s tests -v"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
import numpy as np
import pandas as pd
from fe_model import (ROOT, load_data, portfolio_weights, min_var_long_only,
                      factor_covariance, compute_one_day_models, summarize_daily,
                      WINDOW, CONFIDENCES)
from monte_carlo_interface import monte_carlo_var, normal_analytic_reference


class TestFinancialPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log, cls.simple = load_data()
        cls.v = cls.simple.to_numpy()
        cls.cols = list(cls.log.columns)

    def test_input_prices_and_returns_consistent(self):
        prices = pd.read_csv(ROOT/'data'/'asset_prices.csv', parse_dates=['Date']).set_index('Date')
        self.assertEqual(len(self.log), len(prices)-1)
        self.assertListEqual(list(prices.columns), list(self.log.columns))
        np.testing.assert_array_equal(prices.index[1:].values, self.log.index.values)
        # Original return files round to about 10 decimal places.
        np.testing.assert_allclose(np.log(prices.iloc[1:].values / prices.iloc[:-1].values),
                                   self.log.values, atol=1e-9, rtol=1e-6)

    def test_weights_and_factor_covariance(self):
        weights=portfolio_weights(self.simple)
        self.assertAlmostEqual(weights['EqualWeight'].sum(),1)
        self.assertAlmostEqual(weights['6040'].sum(),1)
        mv=min_var_long_only(np.cov(self.v[:WINDOW],rowvar=False))
        self.assertGreaterEqual(mv.min(),-1e-10)
        self.assertAlmostEqual(mv.sum(),1)
        cov,k,share=factor_covariance(self.v[:WINDOW],target=.90)
        self.assertGreaterEqual(share,.90-1e-10)
        self.assertEqual(cov.shape,(len(self.cols),len(self.cols)))
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(),-1e-10)

    def test_forecast_has_no_future_access(self):
        w=portfolio_weights(self.simple)['EqualWeight']
        f, d=compute_one_day_models(self.v[:WINDOW],w)
        future_changed=self.v.copy()
        future_changed[WINDOW:] *= 1000  # No change to ANY fitting history
        f2,d2=compute_one_day_models(future_changed[:WINDOW],w)
        for a in CONFIDENCES:
            for method in f[a]:
                self.assertAlmostEqual(f[a][method], f2[a][method], places=12)
        for key in d:
            if isinstance(d[key], float):
                self.assertAlmostEqual(d[key], d2[key], places=12)
            else:
                self.assertEqual(d[key], d2[key])
        for a in CONFIDENCES:
            self.assertEqual(set(f[a]), {'Historical','Parametric','RiskMetrics_EWMA','PCA_Factor'})

    def test_minvariance_first_and_second_rebalances_use_past_only(self):
        tab=pd.read_csv(ROOT/'results'/'tables'/'dynamic_portfolio_weights.csv',parse_dates=['Date'])
        minimum=tab[tab.portfolio=='MinVariance'].reset_index(drop=True)
        for offset in (0,21,42):
            t=WINDOW+offset
            expected=min_var_long_only(np.cov(self.v[t-WINDOW:t],rowvar=False,ddof=1))
            row=minimum.loc[minimum.Date==self.simple.index[t]]
            self.assertEqual(len(row),1)
            actual=row[self.cols].to_numpy().ravel()
            np.testing.assert_allclose(actual,expected,atol=1e-8)

    def test_monte_carlo_interface(self):
        w=portfolio_weights(self.simple)['EqualWeight']
        ref=normal_analytic_reference(self.v[:WINDOW],w)
        mc=monte_carlo_var(self.v[:WINDOW],w,n_scenarios=100000,seed=111)
        for a in CONFIDENCES:
            self.assertLess(abs(mc[a]-ref[a]), 0.002)

    def test_outputs_pooled(self):
        tbl=ROOT/'results'/'tables'
        daily=pd.read_csv(tbl/'var_daily_predictions.csv',parse_dates=['Date','training_end_date'])
        self.assertEqual(len(daily),(len(self.log)-WINDOW)*3*2*4)
        self.assertEqual(daily.Date.min(), self.log.index[WINDOW])
        self.assertEqual(daily.Date.max(), self.log.index[-1])
        self.assertTrue((daily.training_end_date < daily.Date).all())
        self.assertTrue(daily.VaR.ge(0).all())
        self.assertFalse(daily[['VaR','actual_return','actual_loss']].isna().any().any())
        np.testing.assert_allclose(daily.actual_loss.values,-daily.actual_return.values)
        np.testing.assert_array_equal(daily.violation.values, (daily.actual_loss > daily.VaR).astype(int).values)
        weights=pd.read_csv(tbl/'dynamic_portfolio_weights.csv')
        self.assertEqual(len(weights),(len(self.log)-WINDOW)*3)
        self.assertTrue(np.allclose(weights[self.cols].sum(axis=1),1,atol=1e-9))
        self.assertTrue((weights[self.cols].to_numpy()>=-1e-10).all())
        summary=pd.read_csv(tbl/'fe_var_backtest_detail.csv')
        self.assertEqual(len(summary),3*2*4)
        actual=summarize_daily(daily)
        joined=summary.merge(actual,on=['portfolio','confidence','method'],suffixes=('_saved','_calc'))
        self.assertEqual(len(joined),len(summary))
        self.assertTrue((joined.breaches_saved == joined.breaches_calc).all())
        np.testing.assert_allclose(joined.Kupiec_p_value_saved,joined.Kupiec_p_value_calc,atol=1e-9)
        diag=pd.read_csv(tbl/'pca_window_diagnostics.csv')
        self.assertTrue(diag.explained_variance.ge(.90-1e-12).all())
        self.assertTrue(diag.pca_components.between(1,len(self.cols)).all())

if __name__=='__main__':
    unittest.main()
