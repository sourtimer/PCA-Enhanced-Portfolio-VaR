"""Regenerate figures from saved five-model results without simulation."""
from __future__ import annotations
import pandas as pd
from fe_model import TABLE_DIR
from plot_presentation import build_all


def main():
    src = TABLE_DIR/'monte_carlo'
    daily = pd.read_csv(src/'all_methods_daily_predictions.csv',parse_dates=['Date'])
    summary = pd.read_csv(src/'all_methods_backtest_summary.csv')
    outputs = build_all(daily,summary)
    for path in outputs: print('Saved:',path)
    return outputs

if __name__=='__main__':
    main()
