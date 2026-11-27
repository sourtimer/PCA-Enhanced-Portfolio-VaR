import numpy as np
import pandas as pd

from fe_model import load_data, WINDOW, TABLE_DIR, summarize_daily
from monte_carlo_interface import monte_carlo_var


_, simple_returns = load_data()

weight_table = pd.read_csv(
    TABLE_DIR / "dynamic_portfolio_weights.csv",
    parse_dates=["Date"],
)

weight_table = weight_table.set_index(["Date", "portfolio"])

if weight_table.index.has_duplicates:
    raise ValueError("权重表存在重复的日期与组合。")


PORTFOLIOS = ("EqualWeight", "6040", "MinVariance")
CONFIDENCES = (0.95, 0.99)

N_SCENARIOS = 20000
BASE_SEED = 2026

records = []
total_days = len(simple_returns) - WINDOW

print(f"开始三种组合的滚动回测，共 {total_days} 个预测日。")


for t in range(WINDOW, len(simple_returns)):
    forecast_date = simple_returns.index[t]
    train = simple_returns.iloc[t - WINDOW:t]
    today_returns = simple_returns.iloc[t].to_numpy()

    for portfolio in PORTFOLIOS:
        weights = weight_table.loc[
            (forecast_date, portfolio),
            simple_returns.columns,
        ].to_numpy(dtype=float)

        if not np.isfinite(weights).all():
            raise ValueError(f"{forecast_date} {portfolio} 权重含无效值。")

        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"{forecast_date} {portfolio} 权重之和不为1。")

        mc_results = monte_carlo_var(
            train.to_numpy(),
            weights,
            n_scenarios=N_SCENARIOS,
            seed=BASE_SEED + t,
        )

        actual_return = float(today_returns @ weights)
        actual_loss = -actual_return

        for confidence in CONFIDENCES:
            var_value = mc_results[confidence]

            records.append({
                "Date": forecast_date.strftime("%Y-%m-%d"),
                "portfolio": portfolio,
                "method": "MonteCarlo_Normal",
                "confidence": confidence,
                "VaR": var_value,
                "actual_return": actual_return,
                "actual_loss": actual_loss,
                "violation": int(actual_loss > var_value),
                "training_end_date": train.index[-1].strftime("%Y-%m-%d"),
            })

    completed = t - WINDOW + 1

    if completed % 100 == 0 or completed == total_days:
        print(f"已完成 {completed}/{total_days} 个预测日")


daily_results = pd.DataFrame(records)

expected_rows = total_days * len(PORTFOLIOS) * len(CONFIDENCES)

if len(daily_results) != expected_rows:
    raise ValueError("总记录数不符合预期。")

keys = ["Date", "portfolio", "method", "confidence"]

if daily_results.duplicated(keys).any():
    raise ValueError("预测结果存在重复记录。")

if not (
    pd.to_datetime(daily_results["training_end_date"])
    < pd.to_datetime(daily_results["Date"])
).all():
    raise ValueError("训练结束日期必须早于预测日。")

if not np.isfinite(daily_results["VaR"]).all():
    raise ValueError("VaR包含无效值。")

if (daily_results["VaR"] < 0).any():
    raise ValueError("VaR不应为负数。")


summary = summarize_daily(daily_results)


output_dir = TABLE_DIR / "monte_carlo"
output_dir.mkdir(parents=True, exist_ok=True)

daily_path = output_dir / "monte_carlo_daily_predictions.csv"
summary_path = output_dir / "monte_carlo_backtest_detail.csv"

daily_results.to_csv(
    daily_path, index=False, encoding="utf-8-sig"
)

summary.to_csv(
    summary_path, index=False, encoding="utf-8-sig"
)


print(f"\n完成！共生成 {len(daily_results)} 条预测记录。")

print("\n各组合、各置信水平的记录数：")
print(
    daily_results.groupby(["portfolio", "confidence"])
    .size()
    .to_string()
)

print("\n回测汇总：")
print(summary.to_string(index=False))

print("\n文件已保存到：")
print(daily_path)
print(summary_path)