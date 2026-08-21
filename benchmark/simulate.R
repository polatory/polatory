#!/usr/bin/env Rscript

library(gstat)

args <- commandArgs(trailingOnly = TRUE)

data_file <- args[1]

data <- read.table(data_file, col.names = c("x", "y", "z"))

set.seed(0)
k <- krige(
  formula = value ~ 1,
  locations = ~ x + y + z,
  data = NULL,
  newdata = data,
  model = vgm(1.0, "Exp", 0.02),
  beta = 0.0,
  nmax = 20,
  nsim = 1,
  dummy = TRUE
)

write.table(
  k[, c("x", "y", "z", "sim1")],
  data_file,
  row.names = FALSE,
  col.names = FALSE
)
