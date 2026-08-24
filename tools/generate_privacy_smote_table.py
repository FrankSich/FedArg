from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "experiment_summary.csv"
OUTPUT_PATH = ROOT / "results" / "global" / "privacy_smote_results.png"


def format_metric(summary, metric):
    row = summary.loc[summary["Metric"] == metric].iloc[0]
    return f"{row['Mean']:.2f} +/- {row['Std']:.2f}"


def add_table(axis, title, columns, rows, bbox):
    axis.text(0.02, bbox[1] + bbox[3] + 0.035, title, fontsize=14, weight="bold")
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#e9eef2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")


def generate_table():
    summary = pd.read_csv(SUMMARY_PATH)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"]
    available = [format_metric(summary, metric) for metric in metrics]

    figure, axis = plt.subplots(figsize=(13, 7))
    axis.axis("off")
    figure.text(
        0.05,
        0.96,
        "Table of Results on Privacy and SMOTE",
        ha="left",
        va="top",
        fontsize=17,
        weight="bold",
    )
    figure.text(
        0.05,
        0.915,
        "Available results: 10 runs with DP enabled, SMPC enabled, SMOTE enabled, sigma = 0.002",
        ha="left",
        va="top",
        fontsize=10,
    )

    add_table(
        axis,
        "Privacy configuration",
        ["Setup", "Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"],
        [
            ["Baseline FL\n(not recorded)", "-", "-", "-", "-", "-", "-"],
            ["FL + DP only\n(not recorded)", "-", "-", "-", "-", "-", "-"],
            ["FL + DP + SMPC\n(mean +/- std)", *available],
        ],
        [0.03, 0.50, 0.94, 0.26],
    )

    add_table(
        axis,
        "SMOTE configuration",
        ["Setup", "Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"],
        [
            ["Before SMOTE\n(not recorded)", "-", "-", "-", "-", "-", "-"],
            ["After SMOTE\n(mean +/- std)", *available],
        ],
        [0.03, 0.14, 0.94, 0.22],
    )

    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    generate_table()