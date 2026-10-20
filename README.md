# FETCH: Flexible Elevation Timing for Coastal Homes

**FETCH** is a Python decision-support framework for evaluating when and how
high to elevate coastal homes under uncertain sea-level rise, storm surge,
flood damage, and adaptation costs.

The model connects coastal flood hazards with building vulnerability and
economic objectives so users can compare home-elevation strategies across
many plausible futures. It is designed for research and planning applications
in climate adaptation, coastal resilience, and flood-risk management.

### What FETCH models

- NOAA observed water levels and annual extremes
- Probabilistic local sea-level-rise projections
- Storm-surge return levels
- Residential depth-damage relationships
- House-elevation costs and discounting
- Elevation height and timing strategies
- Cost, reliability, benefit-cost, and Pareto performance

### Repository structure

```text
src/
  hazards/          Sea-level-rise, NOAA, and storm-surge models
  vulnerabilities/  Residential depth-damage relationships
  economics/        Elevation cost and discounting functions
  objectives/       Cost, reliability, damage, and benefit-cost metrics
  decision/         Strategy generation, optimization, and trade-off analysis
  world/            Houses and states of the world
casestudy/           Case-study inputs
config/              Model configuration
data/                Raw and processed model data
```

### Installation

Create the reproducible Conda environment:

```bash
conda env create -f environment.yml
conda activate fetch-coastal-homes
python -m pip install -e .
```

FETCH currently targets Python 3.11 and is under active development.

### Project links

- Repository: https://github.com/Suhavi/fetch-coastal-homes
- Issues: https://github.com/Suhavi/fetch-coastal-homes/issues
