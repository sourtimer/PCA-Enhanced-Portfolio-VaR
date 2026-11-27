"""Rebuild the price/return inputs; download adjusted prices only on request."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI",
           "TLT", "IEF", "SHY", "GLD", "VNQ", "EEM", "EWU", "EWJ", "AGG"]
START = "2020-01-01"
END = "2026-10-06"
CUTOFF = "2026-09-30"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch adjusted Yahoo Finance prices")
    parser.add_argument("--prices", type=Path, default=ROOT / "data" / "asset_prices.csv")
    args = parser.parse_args()
    output = ROOT / "reproduced" / "data"
    output.mkdir(parents=True, exist_ok=True)

    if args.download:
        import yfinance as yf
        raw = yf.download(tickers=TICKERS, start=START, end=END, interval="1d",
                          auto_adjust=True, group_by="column")
        prices = raw["Close"].reset_index()
    else:
        prices = pd.read_csv(args.prices, parse_dates=["Date"])

    prices = prices.loc[prices["Date"] <= pd.Timestamp(CUTOFF)].copy()
    prices = prices[["Date", *sorted(TICKERS)]]
    if prices.empty or prices.iloc[:, 1:].isna().any().any():
        raise ValueError("Missing ETF prices; check the input or download")
    prices["Date"] = prices["Date"].dt.strftime("%Y/%m/%d")
    prices.to_csv(output / "asset_prices.csv", index=False)

    prices = pd.read_csv(output / "asset_prices.csv", parse_dates=["Date"]).set_index("Date")
    returns = np.log(prices / prices.shift(1)).dropna(how="all").round(10).reset_index()
    returns["Date"] = returns["Date"].dt.strftime("%Y/%m/%d")
    returns.to_csv(output / "asset_returns.csv", index=False)
    print(f"{len(prices)} price rows, {len(returns)} return rows: {output}")


if __name__ == "__main__":
    main()
