#pragma once

#include <boost/program_options.hpp>
#include <string>
#include <vector>

struct ModelOptions {
  std::vector<std::string> rbf_args;
  double nugget{};
  int poly_degree{};
};

inline boost::program_options::options_description make_model_options_description(
    ModelOptions& opts) {
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
  bh2   [SCALE [C]]  Biharmonic2D            1
  bh3   [SCALE [C]]  Biharmonic3D            0
  th2   [SCALE [C]]  Triharmonic2D           2
  th3   [SCALE [C]]  Triharmonic3D           1
              Covariance functions
  cub   PSILL RANGE  CovCubic               -1
  exp   PSILL RANGE  CovExponential         -1
  gau   PSILL RANGE  CovGaussian            -1
  gc3   PSILL RANGE  CovGeneralizedCauchy3  -1
  gc5   PSILL RANGE  CovGeneralizedCauchy5  -1
  gc7   PSILL RANGE  CovGeneralizedCauchy7  -1
  gc9   PSILL RANGE  CovGeneralizedCauchy9  -1
  sp3   PSILL RANGE  CovSpheroidal3         -1
  sp5   PSILL RANGE  CovSpheroidal5         -1
  sp7   PSILL RANGE  CovSpheroidal7         -1
  sp9   PSILL RANGE  CovSpheroidal9         -1
  sph   PSILL RANGE  CovSpherical           -1)")  //
      ("nug", po::value(&opts.nugget)->default_value(0.0, "0.0")->value_name("NUG"),
       "Nugget of the model")  //
      ("deg",
       po::value(&opts.poly_degree)
           ->default_value(polatory::Model<1>::kMinRequiredPolyDegree, "AUTO")
           ->value_name("-1|0|1|2"),
       "Degree of the polynomial trend")  //
      ;

  return opts_desc;
}
