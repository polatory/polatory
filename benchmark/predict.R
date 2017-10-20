#!/usr/bin/env Rscript

library(gstat)

args <- commandArgs(trailingOnly=T)

dataFile <- args[1]
dataValueFile <- args[2]
newdataFile <- args[3]
newdataValueFile <- args[4]

data <- read.table(dataFile, col.names=c("x", "y", "z"))
data <- cbind(data, read.table(dataValueFile, col.names=c('value')))
newdata <- read.table(newdataFile, col.names=c("x", "y", "z"))

model <- vgm(1.0, "Exp", 0.02)

# Ordinary kriging.
k <- krige(value~1, loc=~x+y+z,
           data=data, newdata=newdata,
           model=model)

write.table(k[, c("var1.pred")], newdataValueFile, row.names=F, col.names=F)
