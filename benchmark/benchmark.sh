#!/bin/sh

set -eux

./points 1000 0 1k.txt
./points 10000 0 10k.txt
./points 100000 0 100k.txt
./points 1000000 0 1M.txt

./points 1000 1 1k_test.txt
./points 10000 1 10k_test.txt
./points 100000 1 100k_test.txt
./points 1000000 1 1M_test.txt

./simulate.R   1k.txt
./simulate.R  10k.txt
./simulate.R 100k.txt
./simulate.R   1M.txt

mkdir -p result_gstat
./predict.R   1k.txt   1k_test.txt result_gstat/1k_1k.txt
./predict.R   1k.txt  10k_test.txt result_gstat/1k_10k.txt
./predict.R   1k.txt 100k_test.txt result_gstat/1k_100k.txt
./predict.R   1k.txt   1M_test.txt result_gstat/1k_1M.txt
./predict.R  10k.txt   1k_test.txt result_gstat/10k_1k.txt
./predict.R  10k.txt  10k_test.txt result_gstat/10k_10k.txt

mkdir -p result_polatory
./predict   1k.txt   1k_test.txt result_polatory/1k_1k.txt
./predict   1k.txt  10k_test.txt result_polatory/1k_10k.txt
./predict   1k.txt 100k_test.txt result_polatory/1k_100k.txt
./predict   1k.txt   1M_test.txt result_polatory/1k_1M.txt
./predict  10k.txt   1k_test.txt result_polatory/10k_1k.txt
./predict  10k.txt  10k_test.txt result_polatory/10k_10k.txt
./predict  10k.txt 100k_test.txt result_polatory/10k_100k.txt
./predict  10k.txt   1M_test.txt result_polatory/10k_1M.txt
./predict 100k.txt   1k_test.txt result_polatory/100k_1k.txt
./predict 100k.txt  10k_test.txt result_polatory/100k_10k.txt
./predict 100k.txt 100k_test.txt result_polatory/100k_100k.txt
./predict 100k.txt   1M_test.txt result_polatory/100k_1M.txt
./predict   1M.txt   1k_test.txt result_polatory/1M_1k.txt
./predict   1M.txt  10k_test.txt result_polatory/1M_10k.txt
./predict   1M.txt 100k_test.txt result_polatory/1M_100k.txt
./predict   1M.txt   1M_test.txt result_polatory/1M_1M.txt
