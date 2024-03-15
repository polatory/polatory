#pragma once

#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <memory>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace Eigen {

inline void validate(boost::any& v, const std::vector<std::string>& values,
                     polatory::geometry::matrix3d*, int) {
  namespace po = boost::program_options;

  if (values.size() != 9) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  polatory::geometry::matrix3d aniso;
  aniso << polatory::numeric::to_double(values[0]), polatory::numeric::to_double(values[1]),
      polatory::numeric::to_double(values[2]), polatory::numeric::to_double(values[3]),
      polatory::numeric::to_double(values[4]), polatory::numeric::to_double(values[5]),
      polatory::numeric::to_double(values[6]), polatory::numeric::to_double(values[7]),
      polatory::numeric::to_double(values[8]);

  v = aniso;
}

}  // namespace Eigen

namespace polatory::geometry {

inline void validate(boost::any& v, const std::vector<std::string>& values, bbox3d*, int) {
  namespace po = boost::program_options;

  if (values.size() != 6) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  v = bbox3d({polatory::numeric::to_double(values[0]), polatory::numeric::to_double(values[1]),
              polatory::numeric::to_double(values[2])},
             {polatory::numeric::to_double(values[3]), polatory::numeric::to_double(values[4]),
              polatory::numeric::to_double(values[5])});
}

}  // namespace polatory::geometry

template <int Dim>
polatory::rbf::RbfPtr<Dim> make_rbf(const std::string& name, const std::vector<double>& params) {
  if (name == "bh2") {
    return std::make_unique<polatory::rbf::biharmonic2d<Dim>>(params);
  } else if (name == "bh3") {
    return std::make_unique<polatory::rbf::biharmonic3d<Dim>>(params);
  } else if (name == "ca3") {
    return std::make_unique<polatory::rbf::cov_cauchy3<Dim>>(params);
  } else if (name == "ca5") {
    return std::make_unique<polatory::rbf::cov_cauchy5<Dim>>(params);
  } else if (name == "ca7") {
    return std::make_unique<polatory::rbf::cov_cauchy7<Dim>>(params);
  } else if (name == "ca9") {
    return std::make_unique<polatory::rbf::cov_cauchy9<Dim>>(params);
  } else if (name == "cub") {
    return std::make_unique<polatory::rbf::cov_cubic<Dim>>(params);
  } else if (name == "exp") {
    return std::make_unique<polatory::rbf::cov_exponential<Dim>>(params);
  } else if (name == "gau") {
    return std::make_unique<polatory::rbf::cov_gaussian<Dim>>(params);
  } else if (name == "imq1") {
    return std::make_unique<polatory::rbf::inverse_multiquadric1<Dim>>(params);
  } else if (name == "mq1") {
    return std::make_unique<polatory::rbf::multiquadric1<Dim>>(params);
  } else if (name == "mq3") {
    return std::make_unique<polatory::rbf::multiquadric3<Dim>>(params);
  } else if (name == "sp3") {
    return std::make_unique<polatory::rbf::cov_spheroidal3<Dim>>(params);
  } else if (name == "sp5") {
    return std::make_unique<polatory::rbf::cov_spheroidal5<Dim>>(params);
  } else if (name == "sp7") {
    return std::make_unique<polatory::rbf::cov_spheroidal7<Dim>>(params);
  } else if (name == "sp9") {
    return std::make_unique<polatory::rbf::cov_spheroidal9<Dim>>(params);
  } else if (name == "sph") {
    return std::make_unique<polatory::rbf::cov_spherical<Dim>>(params);
  } else if (name == "th2") {
    return std::make_unique<polatory::rbf::triharmonic2d<Dim>>(params);
  } else if (name == "th3") {
    return std::make_unique<polatory::rbf::triharmonic3d<Dim>>(params);
  } else {
    throw std::runtime_error("Unknown RBF name: " + name);
  }
}

#define MAIN_IMPL(NAME, DIM, PARAMS, OPTS)                                       \
  switch (DIM) {                                                                 \
    case 1:                                                                      \
      main_impl(make_rbf<1>(NAME, PARAMS), OPTS);                                \
      break;                                                                     \
    case 2:                                                                      \
      main_impl(make_rbf<2>(NAME, PARAMS), OPTS);                                \
      break;                                                                     \
    case 3:                                                                      \
      main_impl(make_rbf<3>(NAME, PARAMS), OPTS);                                \
      break;                                                                     \
    default:                                                                     \
      throw std::runtime_error("Unsupported dimension: " + std::to_string(DIM)); \
  }

extern const char* const cov_list;
extern const char* const rbf_cov_list;
