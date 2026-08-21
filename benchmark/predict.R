#!/usr/bin/env Rscript

library(gstat)

args <- commandArgs(trailingOnly = TRUE)

data_file <- args[1]
pred_pts_file <- args[2]
pred_file <- args[3]

data <- read.table(data_file, col.names = c("x", "y", "z", "value"))
pred <- read.table(pred_pts_file, col.names = c("x", "y", "z"))

system.time(
  k <- krige(
    formula = value ~ 1,
    locations = ~ x + y + z,
    data = data,
    newdata = pred,
    model = vgm(1.0, "Exp", 0.02),
    beta = 0.0
  )
)

write.table(
  k[, c("x", "y", "z", "var1.pred")],
  pred_file,
  row.names = FALSE,
  col.names = FALSE
)
