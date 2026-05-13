import numpy as np


def _sorted_results(results, action_year=None):
    if action_year is not None:
        results = [r for r in results if r["action_year"] == action_year]

    return sorted(results, key=lambda r: r["delta_h_ft"])


def plot_height_tradeoffs(results, front=None, action_year=None):
    import matplotlib.pyplot as plt

    results = _sorted_results(results, action_year)
    front = [] if front is None else _sorted_results(front, action_year)

    heights = np.array([r["delta_h_ft"] for r in results])
    total_cost = np.array([r["total_cost"] for r in results]) / 1000
    upfront = np.array([r["upfront_cost"] for r in results]) / 1000
    damages = np.array([r["lifetime_expected_damage"] for r in results]) / 1000
    reliability = 100 * np.array([r["reliability"] for r in results])
    benefit_cost = np.array([r["benefit_cost"] for r in results])

    fig, (ax_cost, ax_obj) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1]},
    )

    ax_cost.axvspan(0, 3, facecolor="0.9", hatch="///", edgecolor="0.8")
    ax_cost.text(1.5, np.nanmax(total_cost) * 0.8, "< 3 ft\nnot practical", ha="center")

    ax_cost.plot(heights, total_cost, color="tab:purple", lw=2, label="Total cost")
    ax_cost.plot(heights, damages, color="darkgreen", lw=2, label="Lifetime expected damage")
    ax_cost.plot(heights, upfront, color="black", lw=2, label="Elevation cost")

    if front:
        front_heights = np.array([r["delta_h_ft"] for r in front])
        front_cost = np.array([r["total_cost"] for r in front]) / 1000
        ax_cost.scatter(front_heights, front_cost, color="tab:red", s=45, zorder=3, label="Pareto")

    ax_cost.set_ylabel("Cost (1,000 USD)")
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(ncol=2, fontsize=8)

    ax_obj.plot(heights, reliability, color="tab:orange", lw=2, label="Reliability")
    ax_bcr = ax_obj.twinx()
    ax_bcr.plot(heights, benefit_cost, color="tab:blue", lw=2, label="Benefit-cost ratio")

    if front:
        front_heights = np.array([r["delta_h_ft"] for r in front])
        front_rel = 100 * np.array([r["reliability"] for r in front])
        ax_obj.scatter(front_heights, front_rel, color="tab:red", s=45, zorder=3)

    ax_obj.set_xlabel("Heightening (ft)")
    ax_obj.set_ylabel("Reliability (%)")
    ax_bcr.set_ylabel("Benefit-cost ratio")
    ax_obj.grid(True, alpha=0.3)

    lines, labels = ax_obj.get_legend_handles_labels()
    lines_b, labels_b = ax_bcr.get_legend_handles_labels()
    ax_obj.legend(lines + lines_b, labels + labels_b, fontsize=8)

    title = "Strategy Tradeoffs"
    if action_year is not None:
        title += f" ({action_year})"
    fig.suptitle(title)

    fig.tight_layout()
    return fig, (ax_cost, ax_obj)


def plot_pareto_cost_reliability(results, front):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.scatter(
        [r["total_cost"] for r in results],
        [r["reliability"] for r in results],
        color="0.75",
        s=45,
        label="All strategies",
    )
    ax.scatter(
        [r["total_cost"] for r in front],
        [r["reliability"] for r in front],
        color="tab:red",
        s=70,
        label="Pareto front",
    )

    for r in front:
        ax.annotate(
            f'{r["delta_h_ft"]} ft',
            xy=(r["total_cost"], r["reliability"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Total cost (USD)")
    ax.set_ylabel("Reliability")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax
