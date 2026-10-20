args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop(
    paste(
      "Usage: Rscript calibrate_random_discount.R",
      "<random|mean|drift> <random-discount-root> <output-dir> <seed>",
      "[map-iterations] [mcmc-iterations]"
    )
  )
}

type <- args[[1]]
source_root <- normalizePath(args[[2]], mustWork = TRUE)
output_dir <- normalizePath(args[[3]], mustWork = FALSE)
seed <- as.integer(args[[4]])
map_iterations <- if (length(args) >= 5) as.integer(args[[5]]) else 5000L
mcmc_iterations <- if (length(args) >= 6) as.integer(args[[6]]) else 100000L

if (!type %in% c("random", "mean", "drift")) {
  stop("type must be one of: random, mean, drift")
}
if (is.na(seed) || is.na(map_iterations) || is.na(mcmc_iterations)) {
  stop("seed and iteration counts must be integers")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

old_working_directory <- setwd(source_root)
on.exit(setwd(old_working_directory), add = TRUE)

suppressPackageStartupMessages({
  library(DEoptim)
  library(adaptMCMC)
  library(readxl)
})

source("R/likelihood.R")
source("R/prior_list.R")

if (type == "random") {
  parnames <- c("rho1", "rho2", "rho3", "sigma_sq")
} else if (type == "mean") {
  parnames <- c("eta", "rho1", "rho2", "rho3", "sigma_sq")
} else {
  parnames <- c("a", "b", "rho1", "rho2", "rho3", "sigma_sq")
}

prior_df <- data.frame(
  name = parnames,
  type = character(length(parnames)),
  lower = numeric(length(parnames)),
  upper = numeric(length(parnames)),
  stringsAsFactors = FALSE
)

for (parameter in parnames) {
  index <- match(parameter, prior_df$name)
  if (grepl("rho", parameter)) {
    prior_df[index, c("type", "lower", "upper")] <- list("normal", -3, 3)
  } else if (parameter == "sigma_sq") {
    prior_df[index, c("type", "lower", "upper")] <- list("log-normal", 0.001, 1)
  } else if (parameter == "eta") {
    prior_df[index, c("type", "lower", "upper")] <- list("normal", 2, 5)
  } else if (parameter == "a") {
    prior_df[index, c("type", "lower", "upper")] <- list("normal", 0, 3)
  } else if (parameter == "b") {
    prior_df[index, c("type", "lower", "upper")] <- list("normal", -0.5, 0.5)
  }
}

priors <- create_prior_list(prior_df)

# The upstream HEAD reads this range with c(read_excel(...)), which produces a
# list with current readxl. The model and its earlier calibration script both
# require a numeric series on the log scale.
rates_percent <- read_excel(
  "data/discount.xlsx",
  range = "Real rates!K7:K231"
)[[1]]
rates_percent <- rates_percent[is.finite(rates_percent) & rates_percent > 0]
dat <- log(rates_percent)

all_parnames <- c("rho1", "rho2", "rho3", "sigma_sq", "eta", "a", "b")
lower_bounds <- c(-2, -2, -2, 0, 0, 0, -2)
upper_bounds <- c(2, 2, 2, 1, 7, 2, 0)

cat(
  sprintf(
    "Calibrating %s with %d observations, seed %d, %d MAP iterations, and %d MCMC iterations.\n",
    type,
    length(dat),
    seed,
    map_iterations,
    mcmc_iterations
  )
)

map_out <- DEoptim::DEoptim(
  neg_log_post,
  lower_bounds[match(parnames, all_parnames)],
  upper_bounds[match(parnames, all_parnames)],
  control = list(
    NP = 20 * length(parnames),
    itermax = map_iterations,
    trace = FALSE
  ),
  parnames = parnames,
  priors = priors,
  dat = dat,
  type = type
)

stepsize <- numeric(length(parnames))
for (parameter in parnames) {
  index <- match(parameter, parnames)
  prior <- priors[[parameter]]
  if (prior$type == "uniform") {
    stepsize[index] <- 0.1 * (prior$max - prior$min) / sqrt(12)
  } else if (prior$type == "normal") {
    stepsize[index] <- 0.1 * prior$sd
  } else if (prior$type == "truncnorm") {
    stepsize[index] <- 0.1 * prior$sd
  } else if (prior$type == "log-normal") {
    stepsize[index] <- 0.1 * sqrt(
      (exp(prior$sdlog^2) - 1) * exp(2 * prior$meanlog + prior$sdlog^2)
    )
  }
}

initial_values <- map_out$optim$bestmem
names(initial_values) <- parnames
acceptance_rate <- 0.234 + (0.44 - 0.234) / length(parnames)
adapt_stop <- min(mcmc_iterations, max(10000L, round(0.1 * mcmc_iterations)))

mcmc_out <- adaptMCMC::MCMC(
  log_post,
  mcmc_iterations,
  init = initial_values,
  adapt = adapt_stop,
  acc.rate = acceptance_rate,
  gamma = 0.51,
  list = TRUE,
  scale = stepsize,
  parnames = parnames,
  priors = priors,
  dat = dat,
  type = type
)

metadata <- list(
  model = type,
  seed = seed,
  data_file = file.path(source_root, "data", "discount.xlsx"),
  data_range = "Real rates!K7:K231",
  observation_years = c(1799L, 2022L),
  observations = length(dat),
  map_iterations = map_iterations,
  mcmc_iterations = mcmc_iterations,
  r_version = R.version.string,
  package_versions = c(
    DEoptim = as.character(packageVersion("DEoptim")),
    adaptMCMC = as.character(packageVersion("adaptMCMC")),
    readxl = as.character(packageVersion("readxl"))
  )
)

map_out$calibration_metadata <- metadata
mcmc_out$calibration_metadata <- metadata

saveRDS(map_out, file.path(output_dir, paste0("map-", type, ".rds")))
saveRDS(mcmc_out, file.path(output_dir, paste0("mcmc-", type, ".rds")))

cat(
  sprintf(
    "Saved %s calibration; MCMC acceptance rate: %.4f\n",
    type,
    mcmc_out$acceptance.rate
  )
)
