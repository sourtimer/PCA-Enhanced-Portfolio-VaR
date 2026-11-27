"""Rolling one-day VaR forecasts and fixed-k PCA sensitivity analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fe_model import (load_data, portfolio_weights, min_var_long_only,
                      compute_one_day_models, factor_covariance, summarize_daily,
                      ensure_results_dirs, TABLE_DIR, FIG_DIR,
                      WINDOW, REBALANCE_EVERY, CONFIDENCES, MAIN_METHODS)


def run_rolling_backtest():
    _, returns_simple = load_data()
    ensure_results_dirs()
    values = returns_simple.to_numpy()
    dates = returns_simple.index
    cols = list(returns_simple.columns)
    fixed = portfolio_weights(returns_simple)
    n = len(dates)
    records = []
    weight_records = []
    diag_records = []
    sensitivity_records = []
    minvariance_w = None
    for t in range(WINDOW, n):
        window_simple = values[t - WINDOW:t, :]
        if minvariance_w is None or (t - WINDOW) % REBALANCE_EVERY == 0:
            minvariance_w = min_var_long_only(np.cov(window_simple, rowvar=False, ddof=1))
        weights = {**fixed, "MinVariance": minvariance_w.copy()}
        day = dates[t].strftime('%Y-%m-%d')
        last_training = dates[t - 1].strftime('%Y-%m-%d')
        first_training = dates[t - WINDOW].strftime('%Y-%m-%d')
        for portfolio, w in weights.items():
            # Today's portfolio weights apply to ALL past scenarios and today's
            # realized return; old weights are not spliced into the history.
            forecasts, info = compute_one_day_models(window_simple, w)
            realized = float(values[t] @ w)
            logrealized = float(np.log1p(realized))
            diag_records.append({"Date": day, "portfolio": portfolio,
                                 "pca_components": info['pca_components'],
                                 "explained_variance": info['pca_explained_variance'],
                                 "train_start": first_training, "train_end": last_training})
            weight_records.append({"Date": day, "portfolio": portfolio,
                                   "train_end": last_training,
                                   **{c: float(x) for c, x in zip(cols, w)}})
            for alpha in CONFIDENCES:
                for method, threshold in forecasts[alpha].items():
                    records.append({"Date": day, "portfolio": portfolio,
                                    "method": method, "confidence": alpha,
                                    "VaR": threshold, "actual_return": realized,
                                    "actual_loss": -realized,
                                    "actual_log_return": logrealized,
                                    "violation": int(-realized > threshold),
                                    "pca_components": (info['pca_components'] if method == 'PCA_Factor' else np.nan),
                                    "pca_explained_variance": (info['pca_explained_variance'] if method == 'PCA_Factor' else np.nan),
                                    "training_end_date": last_training})
            # Fixed k=3/7/8 is a sensitivity check ONLY, using that day's
            # 500-day training window (never full-sample fitted loadings).
            for k in (3, 7, 8):
                sigma_f, _, explained = factor_covariance(window_simple, fixed_k=k)
                mu = info['portfolio_mu']
                factor_sd = float(np.sqrt(max(w @ sigma_f @ w, 0.)))
                for alpha in CONFIDENCES:
                    from scipy.stats import norm
                    var = max(float(norm.ppf(alpha)) * factor_sd - mu, 0.)
                    sensitivity_records.append({"Date": day, "portfolio": portfolio,
                        "method": f"PCA_Factor_k{k}", "confidence": alpha,
                        "VaR": var, "actual_return": realized, "actual_loss": -realized,
                        "violation": int(-realized > var), "pca_components": k,
                        "pca_explained_variance": explained,
                        "training_end_date": last_training})
    daily = pd.DataFrame(records)
    weights = pd.DataFrame(weight_records)
    diagnostics = pd.DataFrame(diag_records)
    sens = pd.DataFrame(sensitivity_records)
    detail = summarize_daily(daily)
    sens_summary = summarize_daily(sens)
    daily.to_csv(TABLE_DIR/'var_daily_predictions.csv', index=False, encoding='utf-8-sig', float_format='%.12g')
    weights.to_csv(TABLE_DIR/'dynamic_portfolio_weights.csv', index=False, encoding='utf-8-sig', float_format='%.12g')
    diagnostics.to_csv(TABLE_DIR/'pca_window_diagnostics.csv', index=False, encoding='utf-8-sig', float_format='%.12g')
    detail.to_csv(TABLE_DIR/'fe_var_backtest_detail.csv', index=False, encoding='utf-8-sig', float_format='%.12g')
    sens.to_csv(TABLE_DIR/'pca_sensitivity_daily.csv', index=False, encoding='utf-8-sig', float_format='%.12g')
    sens_summary.to_csv(TABLE_DIR/'pca_sensitivity_backtest_detail.csv', index=False, encoding='utf-8-sig', float_format='%.12g')
    # Stable Chinese reporting table, underlying numeric columns remain in decimals
    chinese = detail.copy()
    chinese['置信水平'] = chinese['confidence'].map(lambda v: f'{v:.0%}')
    chinese['组合'] = chinese['portfolio']
    chinese['方法'] = chinese['method']
    chinese['回测天数'] = chinese['backtest_days']
    chinese['突破次数'] = chinese['breaches']
    chinese['实际突破率(%)'] = (chinese['actual_breach_rate']*100).round(3)
    chinese['理论突破率(%)'] = (chinese['expected_rate']*100).round(3)
    chinese['理论突破次数'] = chinese['expected_breaches'].round(2)
    chinese['Kupiec_LR'] = chinese['Kupiec_LR'].round(4)
    chinese['Kupiec_p值'] = chinese['Kupiec_p_value'].round(5)
    chinese['5%显著性检验'] = np.where(chinese['reject_at_5pct'], '拒绝无条件覆盖原假设', '未拒绝无条件覆盖原假设')
    chinese[['组合','置信水平','方法','回测天数','突破次数','实际突破率(%)',
             '理论突破率(%)','理论突破次数','Kupiec_LR','Kupiec_p值','5%显著性检验']].to_csv(
             TABLE_DIR/'var_backtest_summary_cn.csv', index=False, encoding='utf-8-sig')
    plot_rolling_var(daily)
    plot_pca_counts(diagnostics)
    print(f'Rolling backtest complete: {len(dates)-WINDOW} test days, {len(daily)} daily forecasts.')
    print(detail.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    return daily, detail


def plot_rolling_var(daily):
    # Use the presentation-ready chart; legend stays outside both axes.
    from plot_presentation import plot_rolling_var as _revised_plot
    return _revised_plot(daily)


def plot_pca_counts(diag):
    g=diag[diag.portfolio=='EqualWeight']
    count=g.pca_components.value_counts().sort_index()
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(count.index.astype(str), count.values, color='#628ab2')
    for x,n in enumerate(count.values):
        ax.text(x, n+5, str(n), ha='center', fontsize=9)
    ax.set_xlabel('Number of PCs selected by 90% in 500-day window')
    ax.set_ylabel('Forecast dates')
    ax.set_title('Rolling PCA dimension (no future data)')
    fig.tight_layout();fig.savefig(FIG_DIR/'pca_window_k.png', dpi=180);plt.close(fig)


if __name__=='__main__':
    run_rolling_backtest()
