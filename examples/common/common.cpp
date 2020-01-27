#include "common.hpp"

const std::string cov_list = R"(
== Positive definite ==
  exp <psill> <range>: cov_exponential
  sp3 <psill> <range>: cov_spheroidal3
  sp5 <psill> <range>: cov_spheroidal5
  sp7 <psill> <range>: cov_spheroidal7
  sp9 <psill> <range>: cov_spheroidal9)";

const std::string spline_list = R"(
== Conditionally positive definite of order 0 ==
  bh3 <var>          : biharmonic3d
== Conditionally positive definite of order 1 ==
  bh2 <var>          : biharmonic2d)";
