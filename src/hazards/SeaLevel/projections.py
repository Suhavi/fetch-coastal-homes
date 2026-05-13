import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

MODEL_COLORS = {
    "K14": "tab:green",
    "DP16": "tab:red",
    "BRICK Slow": "tab:blue",
    "BRICK Fast": "tab:orange",
}


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.hazards.SeaLevel.io_projections import fetch_lsl

sns.set_theme(style="whitegrid", context="paper")


def plot_slr_timeseries(ensembles):
    fig, ax = plt.subplots(figsize=(8, 5))

    for ens in ensembles:
        color = MODEL_COLORS[ens.model]
        mean = np.mean(ens.lsl_ft, axis=0)
        p05 = np.quantile(ens.lsl_ft, 0.05, axis=0)
        p95 = np.quantile(ens.lsl_ft, 0.95, axis=0)

        label = f"{ens.model} RCP {ens.rcp}"

        ax.plot(ens.years, mean, label=label, color = color)
        ax.fill_between(ens.years, p05, p95, alpha=0.35, color = color)

    ax.set_xlabel("Year")
    ax.set_ylabel("Local sea-level rise (ft)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_2100_boxplot(ensembles):
    labels = []
    values = []

    for ens in ensembles:
        idx = np.where(ens.years == 2100)[0][0]
        labels.append(ens.model)
        values.append(ens.lsl_ft[:, idx])

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=values,
        ax=ax,
        showfliers=False,
        width=0.5,
    )    

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Local sea-level rise in 2100 (ft)")
    ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    ensembles = fetch_lsl(syear=2026, eyear=2100)

    os.makedirs("plots/slr", exist_ok=True)

    fig, ax = plot_slr_timeseries(ensembles)
    fig.tight_layout()
    fig.savefig("plots/slr/mean_slr_timeseries.png", dpi=300)

    fig, ax = plot_2100_boxplot(ensembles)
    fig.tight_layout()
    fig.savefig("plots/slr/mean_slr_2100_boxplot.png", dpi=300)

    plt.show()
