#!/usr/bin/env Rscript

library(gstat)

my.simulate <- function(file) {
  data <- read.table(file, col.names=c('x', 'y', 'z'))

  g <- gstat(formula=value~1, loc=~x+y+z,
             model=vgm(1.0, 'Exp', 0.02), beta=0.0,
             nmax=20, dummy=T)
  sim <- predict(g, newdata=data, nsim=1)
  write.table(sim[, c('sim1')], paste(file, '.val', sep=''), row.names=F, col.names=F)
}

my.simulate('1k.txt')
my.simulate('10k.txt')
my.simulate('100k.txt')
my.simulate('1M.txt')
