import csv
import statistics
from pathlib import Path

import numpy as np
import pymupdf as pdf

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT / "data"
OUTPUT_DIR = ROOT / "reproduced" / "eda_reference"

# Part 1. ETF correlation analysis
def run_correlation():
    result_dir = OUTPUT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)

    # Read daily log returns.
    with (data_dir / "asset_returns.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        labels = next(reader)[1:]
        rows = list(reader)

    # Separate dates from returns.
    trade_dates = [row[0] for row in rows]
    returns = [list(map(float, row[1:])) for row in rows]

    # Compute pairwise Pearson correlations.
    etf_cols = list(zip(*returns))
    rho = [[statistics.correlation(a, b) for b in etf_cols] for a in etf_cols]

    # Save the full correlation matrix.
    with (result_dir / "ETF-correlation.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["ETF", *labels])
        for label, row in zip(labels, rho):
            w.writerow([label, *[f"{v:.8f}" for v in row]])

    # Add the heatmap title and sample dates.
    chart = pdf.open()
    page = chart.new_page(width=690, height=700)
    page.insert_text((42, 38), "ETF Daily Return Correlations", fontsize=23)
    page.insert_text(
        (42, 60),
        f"Pearson correlation | {len(returns):,} trading days | {trade_dates[0]} to {trade_dates[-1]}",
        fontsize=10,
    )

    # Set colors by correlation.
    def cell_fill(v):
        target = (0.08, 0.29, 0.61) if v >= 0 else (0.71, 0.19, 0.12)
        return tuple(1 - abs(v) * (1 - t) for t in target)

    # Draw the lower triangle.
    for i, name in enumerate(labels):
        page.insert_textbox(
            pdf.Rect(12, 103 + i * 29, 64, 126 + i * 29), name, fontsize=10, align=2
        )
        page.insert_textbox(
            pdf.Rect(68 + i * 29, 84, 97 + i * 29, 101), name, fontsize=9, align=1
        )
        for k in range(i + 1):
            v = rho[i][k]
            x, y = 68 + k * 29, 100 + i * 29
            page.draw_rect(
                pdf.Rect(x, y, x + 29, y + 29),
                color=(0.94, 0.95, 0.97),
                fill=cell_fill(v),
                width=0.45,
            )
            page.insert_textbox(
                pdf.Rect(x, y + 8, x + 29, y + 26),
                f"{v:.2f}",
                fontsize=8,
                align=1,
                color=(1, 1, 1) if abs(v) > 0.64 else (0.09, 0.12, 0.18),
            )

    # Add the color legend.
    for k, v in enumerate((-1, -0.5, 0, 0.5, 1)):
        x = 68 + k * 102
        page.draw_rect(
            pdf.Rect(x, 645, x + 22, 659),
            color=(0.8, 0.8, 0.8),
            fill=cell_fill(v),
            width=0.5,
        )
        page.insert_text((x + 29, 656), f"{v:+.1f}", fontsize=9)

    # Save the heatmap and print examples.
    page.get_pixmap(matrix=pdf.Matrix(2, 2), alpha=False).save(
        result_dir / "ETF-correlation-heatmap.png"
    )
    print(f"输出目录：{result_dir}")
    print(
        [
            (labels[i], labels[k], round(rho[i][k], 4))
            for i, k in [(6, 14), (6, 8), (5, 9), (8, 9), (4, 8)]
        ]
    )


# Part 2. Principal component analysis (PCA)
def run_pca():
    output = OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)

    # Read daily ETF returns.
    with (data_dir / "asset_returns.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        labels = next(reader)[1:]
        rows = list(reader)
        values = np.array([row[1:] for row in rows], dtype=float)

    # Standardize returns by ETF.
    volatility = values.std(axis=0, ddof=1)
    standardized = (values - values.mean(axis=0)) / volatility

    # Compute the correlation matrix and sort the eigenpairs.
    correlation = np.cov(standardized, rowvar=False)
    eigenvalues, vectors = np.linalg.eigh(correlation)
    eigenvalues = eigenvalues[::-1]
    vectors = vectors[:, ::-1]

    # Use a consistent sign for each component.
    for i in range(len(labels)):
        index = np.abs(vectors[:, i]).argmax()
        if vectors[index, i] < 0:
            vectors[:, i] *= -1

    # Find how many components explain at least 90% of the variance.
    ratios = eigenvalues / eigenvalues.sum()
    cumulative = ratios.cumsum()
    count = int(np.searchsorted(cumulative, 0.90)) + 1

    # Compute correlations between ETFs and component scores.
    loadings = vectors * np.sqrt(eigenvalues)

    # Save the explained variance table.
    with (output / "PCA-explained-variance.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["component", "eigenvalue", "explained_ratio", "cumulative_ratio"]
        )
        for i, (v, r, c) in enumerate(zip(eigenvalues, ratios, cumulative), 1):
            writer.writerow([f"PC{i}", v, r, c])

    # Save the loadings table.
    with (output / "PCA-loadings.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ETF", *[f"PC{i}" for i in range(1, len(labels) + 1)]])
        writer.writerows([label, *row] for label, row in zip(labels, loadings))

    # Print explained variance and the largest loadings.
    print(f"{len(values)} trading days, {len(labels)} ETFs")
    for i in range(count):
        strongest = np.argsort(np.abs(loadings[:, i]))[-4:][::-1]
        print(
            f"PC{i + 1}: {ratios[i]:.2%}, cumulative {cumulative[i]:.2%}",
            [(labels[j], round(float(loadings[j, i]), 3)) for j in strongest],
        )
    print(f"Components to reach 90%: {count}")
    print(output)

    # Plot individual and cumulative explained variance.
    with pdf.open() as chart:
        page = chart.new_page(width=720, height=390)
        page.insert_text((45, 35), "PCA Explained Variance", fontsize=21)
        left, bottom, height, step = 55, 322, 245, 34
        for percent in (0, 25, 50, 75, 90, 100):
            y = bottom - height * percent / 100
            page.draw_line(
                (left, y),
                (left + 18 * step, y),
                color=(0.55, 0.55, 0.55) if percent == 90 else (0.85, 0.85, 0.85),
            )
            page.insert_text((17, y + 3), f"{percent}%", fontsize=9)
        last = None
        for i, (share, total) in enumerate(zip(ratios, cumulative)):
            x = left + step * (i + 0.5)
            page.draw_rect(
                pdf.Rect(x - 10, bottom - share * height, x + 10, bottom),
                fill=(0.12, 0.37, 0.68),
                color=None,
            )
            point = (x, bottom - total * height)
            if last:
                page.draw_line(last, point, color=(0.8, 0.28, 0.08), width=1.6)
            page.draw_circle(point, 2.5, fill=(0.8, 0.28, 0.08), color=None)
            page.insert_text((x - 12, 340), f"PC{i + 1}", fontsize=8)
            last = point

        # Add the legend below the plot.
        legend = [
            (154, (0.12, 0.37, 0.68), "Individual explained variance"),
            (421, (0.8, 0.28, 0.08), "Cumulative explained variance"),
        ]
        for x, color, label in legend:
            page.draw_rect(pdf.Rect(x, 359, x + 13, 372), fill=color, color=None)
            page.insert_text((x + 20, 370), label, fontsize=11)
        chart.save(output / "PCA-explained-variance.pdf")
        page.get_pixmap(matrix=pdf.Matrix(2, 2), alpha=False).save(
            output / "PCA-explained-variance.png"
        )

if __name__ == "__main__":
    run_correlation()
    run_pca()



