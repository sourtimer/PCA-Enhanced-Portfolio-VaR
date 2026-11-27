import numpy as np
import pandas as pd

from fe_model import TABLE_DIR, summarize_daily


mc_dir = TABLE_DIR / "monte_carlo"

baseline = pd.read_csv(
    TABLE_DIR / "var_daily_predictions.csv"
)

mc = pd.read_csv(
    mc_dir / "monte_carlo_daily_predictions.csv"
)

# Parametric与正态Monte Carlo采用相同的分布假设，
# 因此先用它作为逐日核对的参照
reference = baseline.loc[
    baseline["method"] == "Parametric"
].copy()

keys = ["Date", "portfolio", "confidence"]


paired = mc.merge(
    reference,
    on=keys,
    how="outer",
    suffixes=("_mc", "_ref"),
    indicator=True,
    validate="one_to_one",
)

if not (paired["_merge"] == "both").all():
    raise ValueError("两种方法的预测日期、组合或置信水平不一致。")

print("检查通过：所有Monte Carlo记录均有唯一对应的Parametric记录。")


for column in ("actual_return", "actual_loss"):
    if not np.allclose(
        paired[f"{column}_mc"],
        paired[f"{column}_ref"],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(f"{column}与队友结果不一致。")

if not (
    pd.to_datetime(paired["training_end_date_mc"])
    == pd.to_datetime(paired["training_end_date_ref"])
).all():
    raise ValueError("训练结束日期不一致。")

print("检查通过：实际收益、实际损失、训练结束日期一致。")


expected_violation = (
    mc["actual_loss"] > mc["VaR"]
).astype(int)

if not (mc["violation"] == expected_violation).all():
    raise ValueError("Monte Carlo突破标记计算不一致。")

print("检查通过：突破标记正确。")


# 1个基点 = 0.01个百分点 = 小数形式的0.0001
paired["abs_difference_bp"] = (
    paired["VaR_mc"] - paired["VaR_ref"]
).abs() * 10000

difference_summary = (
    paired.groupby(["portfolio", "confidence"])
    ["abs_difference_bp"]
    .agg(["mean", "max"])
    .rename(columns={
        "mean": "mean_abs_difference_bp",
        "max": "max_abs_difference_bp",
    })
    .reset_index()
)

print("\nMC与Parametric的VaR绝对差异，单位：基点")
print(difference_summary.to_string(index=False))


mc["actual_log_return"] = np.log1p(mc["actual_return"])
mc["pca_components"] = np.nan
mc["pca_explained_variance"] = np.nan

combined = pd.concat(
    [baseline, mc.reindex(columns=baseline.columns)],
    ignore_index=True,
)

if combined.duplicated(keys + ["method"]).any():
    raise ValueError("合并后存在重复预测。")

if len(combined) != 35820:
    raise ValueError("合并后的记录数应为35820。")

combined = combined.sort_values(
    ["Date", "portfolio", "confidence", "method"]
).reset_index(drop=True)

combined_summary = summarize_daily(combined)

if len(combined_summary) != 30:
    raise ValueError("五种方法的回测汇总应为30行。")


combined.to_csv(
    mc_dir / "all_methods_daily_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

combined_summary.to_csv(
    mc_dir / "all_methods_backtest_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

difference_summary.to_csv(
    mc_dir / "mc_vs_parametric_difference.csv",
    index=False,
    encoding="utf-8-sig",
)

print(f"\n合并完成：{len(combined)}条逐日预测。")
print(f"回测汇总：{len(combined_summary)}行。")

print("\n等权组合的五种方法对比：")
print(
    combined_summary.loc[
        combined_summary["portfolio"] == "EqualWeight",
        [
            "confidence",
            "method",
            "breaches",
            "actual_breach_rate",
            "Kupiec_p_value",
        ],
    ].to_string(index=False)
)

print("\n比较结果已保存到：")
print(mc_dir)