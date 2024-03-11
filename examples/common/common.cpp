#include "common.hpp"

const char* const cov_list =
    R"(Covariance function, one of:
Arguments        Name
--------------------------------
ca3 PSILL RANGE  cov_cauchy3
ca5 PSILL RANGE  cov_cauchy5
ca7 PSILL RANGE  cov_cauchy7
ca9 PSILL RANGE  cov_cauchy9
exp PSILL RANGE  cov_exponential
gau PSILL RANGE  cov_gaussian
sp3 PSILL RANGE  cov_spheroidal3
sp5 PSILL RANGE  cov_spheroidal5
sp7 PSILL RANGE  cov_spheroidal7
sp9 PSILL RANGE  cov_spheroidal9)";

const char* const rbf_cov_list =
    R"(Basic function, one of:
Arguments        Name                 --deg >=
----------------------------------------------
             Polyharmonic splines
bh2 SCALE        biharmonic2d           1
bh3 SCALE        biharmonic3d           0
th2 SCALE        triharmonic2d          2
th3 SCALE        triharmonic3d          1
          Generalized multiquadrics
imq1 SCALE C     inverse_multiquadric1  -1
mq1 SCALE C      multiquadric1          0
mq3 SCALE C      multiquadric3          1
             Covariance functions
ca3 PSILL RANGE  cov_cauchy3            -1
ca5 PSILL RANGE  cov_cauchy5            -1
ca7 PSILL RANGE  cov_cauchy7            -1
ca9 PSILL RANGE  cov_cauchy9            -1
exp PSILL RANGE  cov_exponential        -1
gau PSILL RANGE  cov_gaussian           -1
sp3 PSILL RANGE  cov_spheroidal3        -1
sp5 PSILL RANGE  cov_spheroidal5        -1
sp7 PSILL RANGE  cov_spheroidal7        -1
sp9 PSILL RANGE  cov_spheroidal9        -1)";
