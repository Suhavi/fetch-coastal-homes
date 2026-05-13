import os
import numpy as np


RW_PARAMS = {
    "rho1": 1.7429,
    "rho2": -1.0455,
    "rho3": 0.3010,
    "sigma_sq": 0.0034,
}

MRV_PARAMS = {
    "eta": 3.405,
    "rho1": 1.7371,
    "rho2": -1.0270,
    "rho3": 0.2806,
    "sigma_sq": 0.0034,
}

DRIFT_PARAMS = {
    "intercept": 1.9289,
    "slope": -0.0058,
    "rho1": 1.6965,
    "rho2": -0.9755,
    "rho3": 0.2388,
    "sigma_sq": 0.0033,
}


def fixed_discount_factors(n_years: int, rate: float = 0.04):
    years = np.arange(n_years + 1)
    return np.exp(-rate * years)


def discount_factors_from_rates(rates_percent):
    rates = np.asarray(rates_percent) / 100
    if rates.ndim == 1:
        rates = rates.reshape(1, -1)

    cumulative_rates = np.cumsum(rates, axis=1)
    return np.column_stack([
        np.ones(rates.shape[0]),
        np.exp(-cumulative_rates),
    ])


def present_value(annual_values, factors):
    annual_values = np.asarray(annual_values)
    factors = np.asarray(factors)
    return np.sum(annual_values * factors)


def lifetime_expected_damage(ead: float, factors):
    return ead * np.sum(factors)


def certainty_equivalent_factors(factor_paths):
    return np.mean(factor_paths, axis=0)


def random_walk_discount(log_history, n_years: int = 100, n_sims: int = 10000, seed=None):
    rng = np.random.default_rng(seed)
    d = np.asarray(log_history)

    eps = np.empty((n_sims, n_years + 3))
    eps[:, :3] = d[-3:] - d[~np.isnan(d)][0]

    for year in range(3, n_years + 3):
        eps[:, year] = (
            RW_PARAMS["rho1"] * eps[:, year - 1]
            + RW_PARAMS["rho2"] * eps[:, year - 2]
            + RW_PARAMS["rho3"] * eps[:, year - 3]
            + rng.normal(0, np.sqrt(RW_PARAMS["sigma_sq"]), size=n_sims)
        )

    return np.exp(d[~np.isnan(d)][0] + eps[:, 3:])


def mean_reverting_discount(log_history, n_years: int = 100, n_sims: int = 10000, seed=None):
    rng = np.random.default_rng(seed)
    d = np.asarray(log_history)

    eps = np.empty((n_sims, n_years + 3))
    eps[:, :3] = d[-3:] - np.log(MRV_PARAMS["eta"])

    for year in range(3, n_years + 3):
        eps[:, year] = (
            MRV_PARAMS["rho1"] * eps[:, year - 1]
            + MRV_PARAMS["rho2"] * eps[:, year - 2]
            + MRV_PARAMS["rho3"] * eps[:, year - 3]
            + rng.normal(0, np.sqrt(MRV_PARAMS["sigma_sq"]), size=n_sims)
        )

    return np.exp(np.log(MRV_PARAMS["eta"]) + eps[:, 3:])


def drift_discount(log_history, n_years: int = 100, n_sims: int = 10000, seed=None):
    rng = np.random.default_rng(seed)
    d = np.asarray(log_history)

    offset = len(d) - 1
    trend_years = np.arange(offset - 2, n_years + offset + 1)
    trend = DRIFT_PARAMS["intercept"] + DRIFT_PARAMS["slope"] * trend_years

    eps = np.empty((n_sims, n_years + 3))
    eps[:, :3] = d[-3:] - trend[:3]

    for year in range(3, n_years + 3):
        eps[:, year] = (
            DRIFT_PARAMS["rho1"] * eps[:, year - 1]
            + DRIFT_PARAMS["rho2"] * eps[:, year - 2]
            + DRIFT_PARAMS["rho3"] * eps[:, year - 3]
            + rng.normal(0, np.sqrt(DRIFT_PARAMS["sigma_sq"]), size=n_sims)
        )

    return np.exp(eps[:, 3:] + trend[3:])


def example_log_history():
    return np.log(np.array([4.2, 4.0, 3.8]))


def plot_discount_rates(log_history=None, n_years: int = 100, n_sims: int = 2000):

    if log_history is None:
        log_history = example_log_history()

    years = np.arange(1, n_years + 1)
    models = {
        "Random walk": random_walk_discount(log_history, n_years, n_sims, seed=1),
        "Mean reverting": mean_reverting_discount(log_history, n_years, n_sims, seed=1),
        "Drift": drift_discount(log_history, n_years, n_sims, seed=1),
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    for label, rates in models.items():
        mean = np.mean(rates, axis=0)
        p05 = np.quantile(rates, 0.05, axis=0)
        p95 = np.quantile(rates, 0.95, axis=0)

        ax.plot(years, mean, label=label)
        ax.fill_between(years, p05, p95, alpha=0.15)

    ax.axhline(4.0, color="0.4", linestyle="--", linewidth=1, label="Fixed 4%")
    ax.set_xlabel("Years in future")
    ax.set_ylabel("Discount rate (%/year)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_discount_factors(log_history=None, n_years: int = 100, n_sims: int = 2000):

    if log_history is None:
        log_history = example_log_history()

    years = np.arange(n_years + 1)
    rate_paths = {
        "Random walk": random_walk_discount(log_history, n_years, n_sims, seed=1),
        "Mean reverting": mean_reverting_discount(log_history, n_years, n_sims, seed=1),
        "Drift": drift_discount(log_history, n_years, n_sims, seed=1),
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(years, fixed_discount_factors(n_years, 0.04), color="black", label="Fixed 4%")

    for label, rates in rate_paths.items():
        factors = discount_factors_from_rates(rates)
        mean = np.mean(factors, axis=0)
        p05 = np.quantile(factors, 0.05, axis=0)
        p95 = np.quantile(factors, 0.95, axis=0)

        ax.plot(years, mean, label=label)
        ax.fill_between(years, p05, p95, alpha=0.15)

    ax.set_yscale("log")
    ax.set_xlabel("Years in future")
    ax.set_ylabel("Discount factor")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    os.makedirs("plots/economics", exist_ok=True)

    fig, ax = plot_discount_rates()
    fig.savefig("plots/economics/discount_rates.png", dpi=300)

    fig, ax = plot_discount_factors()
    fig.savefig("plots/economics/discount_factors.png", dpi=300)

    plt.show()
