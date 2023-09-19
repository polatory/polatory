#include "common.hpp"

const char* const cov_list =
    R"(Covariance function, one of:
Positive definite
  exp VAR SCALE  (cov_exponential)
  sp3 VAR SCALE  (cov_spheroidal3)
  sp5 VAR SCALE  (cov_spheroidal5)
  sp7 VAR SCALE  (cov_spheroidal7)
  sp9 VAR SCALE  (cov_spheroidal9))";

const char* const rbf_cov_list =
    R"(RBF/covariance function, one of:
Positive definite
  exp VAR SCALE  (cov_exponential)
  sp3 VAR SCALE  (cov_spheroidal3)
  sp5 VAR SCALE  (cov_spheroidal5)
  sp7 VAR SCALE  (cov_spheroidal7)
  sp9 VAR SCALE  (cov_spheroidal9)
Conditionally positive definite of order 1
  bh3 VAR        (biharmonic3d)
  mq1 VAR C      (multiquadric1)
Conditionally positive definite of order 2
  bh2 VAR        (biharmonic2d))";
