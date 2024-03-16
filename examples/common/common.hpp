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
polatory::rbf::rbf_proxy<Dim> make_rbf(const std::string& name, const std::vector<double>& params) {
#define CASE(SHORT_NAME, RBF_NAME)               \
  if (name == SHORT_NAME) {                      \
    return polatory::rbf::RBF_NAME<Dim>(params); \
  }

  CASE("bh2", biharmonic2d);
  CASE("bh3", biharmonic3d);
  CASE("ca3", cov_cauchy3);
  CASE("ca5", cov_cauchy5);
  CASE("ca7", cov_cauchy7);
  CASE("ca9", cov_cauchy9);
  CASE("cub", cov_cubic);
  CASE("exp", cov_exponential);
  CASE("gau", cov_gaussian);
  CASE("imq1", inverse_multiquadric1);
  CASE("mq1", multiquadric1);
  CASE("mq3", multiquadric3);
  CASE("sp3", cov_spheroidal3);
  CASE("sp5", cov_spheroidal5);
  CASE("sp7", cov_spheroidal7);
  CASE("sp9", cov_spheroidal9);
  CASE("sph", cov_spherical);
  CASE("th2", triharmonic2d);
  CASE("th3", triharmonic3d);

#undef CASE

  throw std::runtime_error("Unknown RBF name: " + name);
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
