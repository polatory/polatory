#!/usr/bin/env Rscript

library(gstat)

args <- commandArgs(trailingOnly=T)

dataFile <- args[1]
dataValueFile <- args[2]

data <- read.table(dataFile, col.names=c("x", "y", "z"))

g <- gstat(formula=value~1, loc=~x+y+z,
           model=vgm(1.0, "Exp", 0.02), beta=0.0,
           nmax=20, dummy=T)
sim <- predict(g, newdata=data, nsim=1)
write.table(sim[, c("sim1")], dataValueFile, row.names=F, col.names=F)
