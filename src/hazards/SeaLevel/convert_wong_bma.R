# Convert Tony Wong's published R object to a Python-readable table.
# Wong (2018), Sections 2.2 and 2.4; https://doi.org/10.5194/ascmo-4-53-2018

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript convert_wong_bma.R <input.rds> <output.csv.gz>")
}

storm_surge <- readRDS(args[1])
return_levels <- storm_surge$bma
return_periods <- as.integer(sub("y", "", colnames(return_levels)))

stopifnot(
  nrow(return_levels) == 8021,
  identical(return_periods, c(2L, 3L, 5L, 10L, 20L, 30L, 50L, 80L, 100L,
                              150L, 200L, 250L, 300L, 350L, 400L, 450L,
                              500L, 1000L)),
  all(apply(return_levels, 1, function(x) all(diff(x) > 0)))
)

output <- data.frame(member = seq_len(nrow(return_levels)), return_levels)
connection <- gzfile(args[2], "wt")
write.csv(output, connection, row.names = FALSE, quote = FALSE)
close(connection)
