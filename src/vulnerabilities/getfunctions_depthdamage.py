import numpy as np
from pathlib import Path

uncertainty_depth_damage = 0.3

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CURVE_FILES = {
    "europa": PROJECT_ROOT / "data/raw/fragility_EUROPA.csv",
    "hazus": PROJECT_ROOT / "data/raw/fragility_hazus.csv",
}


def load_depth_damage_curve(curve="hazus"):
    if curve not in CURVE_FILES:
        raise ValueError(f"Unknown depth-damage curve: {curve}")

    data = np.genfromtxt(
        CURVE_FILES[curve],
        delimiter=",",
        names=True,
    )

    damage_frac = data["damage_frac"]

    return {
        "curve": curve,
        "depth_ft": data["depth_feet"],
        "damage_frac": damage_frac,
        "damage_lb": damage_frac * (1-uncertainty_depth_damage),
        "damage_ub": np.minimum(damage_frac * (1+uncertainty_depth_damage), 1.0),
    }


def get_damage_fraction(depth_ft, curve="hazus"):
    dd = load_depth_damage_curve(curve)

    return np.interp(
        depth_ft,
        dd["depth_ft"],
        dd["damage_frac"],
        left=0.0,
        right=dd["damage_frac"][-1],
    )


def get_damage_fraction_bounds(depth_ft, curve="hazus"):
    dd = load_depth_damage_curve(curve)

    lower = np.interp(
        depth_ft,
        dd["depth_ft"],
        dd["damage_lb"],
        left=0.0,
        right=dd["damage_lb"][-1],
    )
    upper = np.interp(
        depth_ft,
        dd["depth_ft"],
        dd["damage_ub"],
        left=0.0,
        right=dd["damage_ub"][-1],
    )

    return lower, upper


def get_damage_usd(depth_ft, house_value_usd, curve="hazus"):
    return get_damage_fraction(depth_ft, curve) * house_value_usd


# def plot_depth_damage_curves():
#     from src.nature_figures import clean_axes, nature_subplots

#     fig, ax = nature_subplots("double")

#     for curve, label, color in [
#         ("hazus", "HAZUS", "#662506"),
#         ("europa", "EUROPA", "#238b45"),
#     ]:
#         dd = load_depth_damage_curve(curve)

#         ax.fill_between(
#             dd["depth_ft"],
#             dd["damage_lb"],
#             dd["damage_ub"],
#             color=color,
#             alpha=0.15,
#         )

#         ax.plot(
#             dd["depth_ft"],
#             dd["damage_frac"],
#             color=color,
#             marker="o",
#             label=label,
#         )

#     ax.set_xlabel("Flood depth in house (ft)")
#     ax.set_ylabel("Damage fraction")
#     ax.grid(False)
#     ax.legend(frameon=False)
#     clean_axes(ax, y_bounds=(0, 1))

#     fig.tight_layout()
#     return fig, ax


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     fig, ax = plot_depth_damage_curves()
#     plt.show()
#     fig.savefig("plots/vulnerability/depth_damage_uncertainty.png", dpi=300)
