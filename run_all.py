"""Command-line entry point for VaR estimation and figure generation."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))


def run_script(name):
    print(f'\n>>> {name}',flush=True)
    subprocess.run([sys.executable, str(SRC / name)],cwd=ROOT,check=True)


def run_base():
    from fe_var_analysis import run_static_analysis
    from fe_var_factor_backtest import run_rolling_backtest
    run_static_analysis()
    run_rolling_backtest()


def main(argv=None):
    parser=argparse.ArgumentParser(description='ETF portfolio VaR pipeline')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--charts-only',action='store_true',help='regenerate figures from saved forecasts (default)')
    group.add_argument('--base',action='store_true',help='recompute 4-model baseline; MC must be regenerated separately if inputs change')
    group.add_argument('--full',action='store_true',help='recompute base, full rolling Monte Carlo, merge and all presentation figures')
    args=parser.parse_args(argv)
    if args.base or args.full:
        run_base()
    if args.full:
        run_script('monte_carlo_backtest.py')
        run_script('compare_monte_carlo.py')
    if args.charts_only or args.full or not(args.base or args.full):
        run_script('plot_monte_carlo.py')
        if args.full:
            print('\nFull pipeline completed. Final figures in results/figures/',flush=True)
    elif args.base:
        print('Base recomputed. Existing Monte Carlo combined tables were NOT refreshed; do not mix them with changed baseline outputs.',flush=True)


if __name__=='__main__':
    main()
