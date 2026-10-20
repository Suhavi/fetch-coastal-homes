# Wong (2018) nonstationary storm surge

`Wong_2018_norfolk_MCMC_stormsurge.rds` is the published Norfolk/Sewells Point
return-level ensemble used by Ruckert, Srikrishnan, and Keller (2019).

- Source: `scrim-network/local-coastal-flood-risk`, commit
  `973d7d903d6724e112f954c48776f0074435c803`
- Original path: `Data/Wong_2018_norfolk_MCMC_stormsurge.rds`
- SHA-256: `94f3106e5d8db52e0b922418e5e2e4be568010769b6fc5368c924add896d37df`
- Calibration: Wong (2018), https://doi.org/10.5194/ascmo-4-53-2018
- Site: Sewells Point, Norfolk, Virginia
- Projection year: 2065
- Original units: millimetres above mean sea level

The top-level `bma` matrix is identical to `rl$bma` in Tony Wong's official
`covariates/output/analysis0.RData` at commit
`3888048ebea42de73aa4cb1e2e4884131dd0a8be`.

Run `src/hazards/SeaLevel/convert_wong_bma.R` to reproduce the processed
compressed table used by Python.
