args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 5) {
  stop(
    paste(
      "Usage: Rscript simulate_random_discount.R",
      "<random-discount-root> <fit-dir> <output.csv.gz> <seed> <n-samples>"
    )
  )
}

source_root <- normalizePath(args[[1]], mustWork = TRUE)
fit_dir <- normalizePath(args[[2]], mustWork = TRUE)
output_file <- file.path(
  normalizePath(dirname(args[[3]]), mustWork = TRUE),
  basename(args[[3]])
)
seed <- as.integer(args[[4]])
n_samples <- as.integer(args[[5]])

if (is.na(seed) || is.na(n_samples) || n_samples < 1) {
  stop("seed and n-samples must be positive integers")
}

set.seed(seed)
old_working_directory <- setwd(source_root)
on.exit(setwd(old_working_directory), add = TRUE)

suppressPackageStartupMessages(library(readxl))
source("R/random_discount.R")

rates_percent <- read_excel(
  "data/discount.xlsx",
  range = "Real rates!K7:K231"
)[[1]]
rates_percent <- rates_percent[is.finite(rates_percent) & rates_percent > 0]
log_history <- log(rates_percent)

models <- c("random", "mean", "drift")
parameter_names <- list(
  random = c("rho1", "rho2", "rho3", "sigma_sq"),
  mean = c("eta", "rho1", "rho2", "rho3", "sigma_sq"),
  drift = c("a", "b", "rho1", "rho2", "rho3", "sigma_sq")
)
selected_years <- c(2026L, 2056L, 2076L, 2101L)
first_projection_year <- 2023L
n_years <- max(selected_years) - first_projection_year + 1L
selected_indices <- selected_years - first_projection_year + 1L

selected_samples <- list()

for (model in models) {
  fit_file <- file.path(fit_dir, paste0("mcmc-", model, ".rds"))
  if (!file.exists(fit_file)) {
    stop(paste("Missing fitted posterior:", fit_file))
  }

  fit <- readRDS(fit_file)
  sampled_rows <- sample.int(nrow(fit$samples), n_samples, replace = TRUE)
  parameters <- fit$samples[
    sampled_rows,
    parameter_names[[model]],
    drop = FALSE
  ]

  simulate_model <- match.fun(paste0(model, "_discount"))
  paths <- simulate_model(
    log_history,
    N = n_years,
    n = n_samples,
    pars = parameters,
    parnames = parameter_names[[model]]
  )

  for (index in seq_along(selected_years)) {
    column_name <- paste(model, selected_years[[index]], sep = "__")
    selected_samples[[column_name]] <- paths[, selected_indices[[index]]]
  }

  rm(paths, parameters, fit)
  invisible(gc())
}

output_connection <- gzfile(output_file, open = "wt")
write.csv(as.data.frame(selected_samples), output_connection, row.names = FALSE)
close(output_connection)

cat(
  sprintf(
    "Saved %d posterior-predictive samples for %d models and %d years to %s\n",
    n_samples,
    length(models),
    length(selected_years),
    output_file
  )
)
