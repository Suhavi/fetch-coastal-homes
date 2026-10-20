# Processed local sea-level projections

These NetCDF files are generated from `data/raw/msl` by
`src.hazards.SeaLevel.io_projections`.

Each ensemble is annualized, converted to feet, adjusted with the K14
background component when required for BRICK, and expressed relative to the
2026 datum. The files retain individual simulations; means, quantiles, PDFs,
CDFs, and exceedance probabilities are analysis outputs and are not stored
here.

Rebuild all configured projections from the repository root with:

```bash
conda run -n flood python -m src.hazards.SeaLevel.io_projections
```

The filename and NetCDF metadata record the model, RCP, year range, datum,
background adjustment, random seed, raw source file, and processing version.
Do not edit generated NetCDF files by hand.
