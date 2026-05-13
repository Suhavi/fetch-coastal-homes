import numpy as np

# definiting base costs acc to CLARA
clara_base_cost = {
                "admin" : 10000, #Project administration and construction management
                "title": 300, #title search 
                "survey" : 470,
                "inspection" : 4300,
                "permitting" : 2175, #estimated permitting cost
                "temp-relocation": 3500 #Temporary occupant relocation costs
            }

BASE_COST = np.sum(np.array([clara_base_cost[c] for c in clara_base_cost]))
HEIGHTS_FT = np.array([3, 5, 8.5, 12, 14])
RATES_PER_SQFT = np.array([80.36, 82.5, 86.25, 103.75, 113.75])


def elevation_cost(delta_h_ft: float, sqft: float):
    if np.isclose(delta_h_ft, 0):
        return 0.0

    if delta_h_ft < 3:
        return np.nan
    
    if delta_h_ft > 14:
        raise ValueError("Elevation must be 0 or between 3 and 14 ft.")

    rate = np.interp(delta_h_ft, HEIGHTS_FT, RATES_PER_SQFT)
    return BASE_COST + rate * sqft


def batch_elevation_cost(delta_h_ft: np.ndarray, sqft: float):
    return np.array([elevation_cost(h, sqft) 
                     for h in delta_h_ft
                     ])

def plot_elevation_cost():
    import matplotlib.pyplot as plt

    heights = np.linspace(0,14,200)
    house_sizes = (1000,1500,2000,2500,3000)

    fig, ax = plt.subplots(figsize=(8,4))

    for sqft in house_sizes:
        costs = []

        for h in heights:
            costs.append(elevation_cost(h,sqft))

        ax.plot(heights, np.array(costs) / 1000, label=f"{sqft:,} sqft")
        
    ax.axvspan(0, 3, color="0.85", alpha=0.6)
    ax.text(1.5, 250, "Not practical", rotation=90, ha="center", va="center")

    ax.set_xlabel("Heightening (ft)")
    ax.set_ylabel("Construction cost (1,000 USD)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(elevation_cost(3, 1000))
    print(elevation_cost(8, 2000))

    fig, ax = plot_elevation_cost()
    fig.savefig("plots/economics/elevation_cost.png", dpi=300)

    plt.show()




