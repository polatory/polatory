#pragma once

#include <boost/program_options.hpp>
#include <string>
#include <vector>

struct model_options {
  std::vector<std::string> rbf_args;
  double nugget{};
  int poly_degree{};
};

inline boost::program_options::options_description make_model_options_description(
    model_options& opts) {
  namespace po = boost::program_options;

  po::options_description opts_desc("Model options", 80, 50);
  opts_desc.add_options()  //
      ("rbf",
       po::value(&opts.rbf_args)
           ->multitoken()
           ->required()
           ->value_name("NAME PARAMS [aniso A_11 A_12 ... A_dd] ..."),
       R"(Basic function(s), one of:
  NAME  PARAMS       Full Name        --deg >=
  --------------------------------------------
                 Polyharmonic splines
  bh2   SCALE        biharmonic2d            1
  bh3   SCALE        biharmonic3d            0
  th2   SCALE        triharmonic2d           2
  th3   SCALE        triharmonic3d           1
              Generalized multiquadrics
  imq1  SCALE C      inverse_multiquadric1  -1
  mq1   SCALE C      multiquadric1           0
  mq3   SCALE C      multiquadric3           1
                 Covariance functions
  ca3   PSILL RANGE  cov_cauchy3            -1
  ca5   PSILL RANGE  cov_cauchy5            -1
  ca7   PSILL RANGE  cov_cauchy7            -1
  ca9   PSILL RANGE  cov_cauchy9            -1
  cub   PSILL RANGE  cov_cubic              -1
  exp   PSILL RANGE  cov_exponential        -1
  gau   PSILL RANGE  cov_gaussian           -1
  sp3   PSILL RANGE  cov_spheroidal3        -1
  sp5   PSILL RANGE  cov_spheroidal5        -1
  sp7   PSILL RANGE  cov_spheroidal7        -1
  sp9   PSILL RANGE  cov_spheroidal9        -1
  sph   PSILL RANGE  cov_spherical          -1)")  //
      ("nug", po::value(&opts.nugget)->default_value(0.0, "0.0")->value_name("NUG"),
       "Nugget of the model")  //
      ("deg",
       po::value(&opts.poly_degree)
           ->default_value(polatory::model<1>::kMinRequiredPolyDegree, "AUTO")
           ->value_name("-1|0|1|2"),
       "Degree of the polynomial trend")  //
      ;

  return opts_desc;
}
