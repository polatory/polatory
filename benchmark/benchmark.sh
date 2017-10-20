#!/bin/bash

./points 1000 0 > 1k.txt
./points 10000 0 > 10k.txt
./points 100000 0 > 100k.txt
./points 1000000 0 > 1M.txt

./points 1000 1 > 1k_predict.txt
./points 10000 1 > 10k_predict.txt
./points 100000 1 > 100k_predict.txt
./points 1000000 1 > 1M_predict.txt

#./simulate.R

#./predict.R

#time ./predict 1k.txt 1k.txt.val 1k_predict.txt
#time ./predict 1k.txt 1k.txt.val 10k_predict.txt
#time ./predict 1k.txt 1k.txt.val 100k_predict.txt
#time ./predict 1k.txt 1k.txt.val 1M_predict.txt
