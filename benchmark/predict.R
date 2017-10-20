#!/usr/bin/env Rscript

library(gstat)

my.predict <- function(dataFile, newdataFile) {
  cat(paste('data:', dataFile, 'newdata:', newdataFile), '\n')
  
  data <- read.table(dataFile, col.names=c('x', 'y', 'z'))
  data <- cbind(data, read.table(paste(dataFile, '.val', sep=''), col.names=c('value')))
  newdata <- read.table(newdataFile, col.names=c('x', 'y', 'z'))
  
  model <- vgm(1.0, "Exp", 0.02)
  
  # Ordinary kriging.
  k <- krige(value~1, loc=~x+y+z,
             data=data, newdata=newdata,
             model=model)
}

system.time(my.predict('1k.txt', '1k_predict.txt'))
system.time(my.predict('1k.txt', '10k_predict.txt'))
system.time(my.predict('1k.txt', '100k_predict.txt'))
system.time(my.predict('10k.txt', '1k_predict.txt'))
system.time(my.predict('10k.txt', '10k_predict.txt'))
