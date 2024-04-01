#pragma once

#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <optional>
#include <string>
#include <vector>

struct rbf_options {
  std::string name;
  std::vector<double> params;
  std::vector<double> aniso;
};

rbf_options parse_rbf_options(const std::vector<std::string>& values) {
  namespace po = boost::program_options;

  if (values.size() < 1) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  auto it = values.begin();
  auto end = values.end();

  rbf_options opts;
  opts.name = *it++;

  while (it != end && *it != "aniso") {
    opts.params.push_back(polatory::numeric::to_double(*it++));
  }

  if (it != end) {
    it++;  // Skip "aniso".
    if (it == end) {
      throw po::validation_error(po::validation_error::invalid_option_value);
    }

    while (it != end) {
      opts.aniso.push_back(polatory::numeric::to_double(*it++));
    }
  }

  return opts;
}

inline void validate(boost::any& v, const std::vector<std::string>& values, rbf_options*, int) {
  v = parse_rbf_options(values);
}

inline void validate(boost::any& v, const std::vector<std::string>& values,
                     std::optional<rbf_options>*, int) {
  v = std::optional{parse_rbf_options(values)};
}

struct model_options {
  rbf_options rbf;
  std::optional<rbf_options> rbf2;
  double nugget{};
  int poly_degree{};
};

boost::program_options::options_description make_model_options_description(model_options& opts) {
  namespace po = boost::program_options;

  po::options_description opts_desc("Model options");
  opts_desc.add_options()  //
      ("rbf",
       po::value(&opts.rbf)->multitoken()->required()->value_name(
           "... [aniso A_11 A_12 ... A_dd]"),  //
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
cub PSILL RANGE  cov_cubic              -1
exp PSILL RANGE  cov_exponential        -1
gau PSILL RANGE  cov_gaussian           -1
sp3 PSILL RANGE  cov_spheroidal3        -1
sp5 PSILL RANGE  cov_spheroidal5        -1
sp7 PSILL RANGE  cov_spheroidal7        -1
sp9 PSILL RANGE  cov_spheroidal9        -1
sph PSILL RANGE  cov_spherical          -1)")  //
      ("rbf2",
       po::value(&opts.rbf2)
           ->multitoken()
           ->default_value(std::nullopt, "NONE")
           ->value_name("... [aniso A_11 A_12 ... A_dd]"),                            //
       "Second basic function to be added")                                           //
      ("nug", po::value(&opts.nugget)->default_value(0.0, "0.0")->value_name("NUG"),  //
       "Nugget of the model")                                                         //
      ("deg",
       po::value(&opts.poly_degree)
           ->default_value(polatory::model<1>::kMinRequiredPolyDegree, "AUTO")
           ->value_name("-1|0|1|2"),  //
       "Degree of the polynomial trend");

  return opts_desc;
}
