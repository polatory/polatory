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

#define MAIN_IMPL_DIM(NAME, DIM, PARAMS, OPTS)                          \
  if (NAME == "bh2") {                                                  \
    main_impl(polatory::rbf::biharmonic2d<DIM>(PARAMS), OPTS);          \
  } else if (NAME == "bh3") {                                           \
    main_impl(polatory::rbf::biharmonic3d<DIM>(PARAMS), OPTS);          \
  } else if (NAME == "ca3") {                                           \
    main_impl(polatory::rbf::cov_cauchy3<DIM>(PARAMS), OPTS);           \
  } else if (NAME == "ca5") {                                           \
    main_impl(polatory::rbf::cov_cauchy5<DIM>(PARAMS), OPTS);           \
  } else if (NAME == "ca7") {                                           \
    main_impl(polatory::rbf::cov_cauchy7<DIM>(PARAMS), OPTS);           \
  } else if (NAME == "ca9") {                                           \
    main_impl(polatory::rbf::cov_cauchy9<DIM>(PARAMS), OPTS);           \
  } else if (NAME == "exp") {                                           \
    main_impl(polatory::rbf::cov_exponential<DIM>(PARAMS), OPTS);       \
  } else if (NAME == "imq1") {                                          \
    main_impl(polatory::rbf::inverse_multiquadric1<DIM>(PARAMS), OPTS); \
  } else if (NAME == "mq1") {                                           \
    main_impl(polatory::rbf::multiquadric1<DIM>(PARAMS), OPTS);         \
  } else if (NAME == "mq3") {                                           \
    main_impl(polatory::rbf::multiquadric3<DIM>(PARAMS), OPTS);         \
  } else if (NAME == "sp3") {                                           \
    main_impl(polatory::rbf::cov_spheroidal3<DIM>(PARAMS), OPTS);       \
  } else if (NAME == "sp5") {                                           \
    main_impl(polatory::rbf::cov_spheroidal5<DIM>(PARAMS), OPTS);       \
  } else if (NAME == "sp7") {                                           \
    main_impl(polatory::rbf::cov_spheroidal7<DIM>(PARAMS), OPTS);       \
  } else if (NAME == "sp9") {                                           \
    main_impl(polatory::rbf::cov_spheroidal9<DIM>(PARAMS), OPTS);       \
  } else if (NAME == "th2") {                                           \
    main_impl(polatory::rbf::triharmonic2d<DIM>(PARAMS), OPTS);         \
  } else if (NAME == "th3") {                                           \
    main_impl(polatory::rbf::triharmonic3d<DIM>(PARAMS), OPTS);         \
  } else {                                                              \
    throw std::runtime_error("Unknown RBF name: " + NAME);              \
  }

#define MAIN_IMPL(NAME, DIM, PARAMS, OPTS)                                       \
  switch (DIM) {                                                                 \
    case 1:                                                                      \
      MAIN_IMPL_DIM(NAME, 1, PARAMS, OPTS)                                       \
      break;                                                                     \
    case 2:                                                                      \
      MAIN_IMPL_DIM(NAME, 2, PARAMS, OPTS)                                       \
      break;                                                                     \
    case 3:                                                                      \
      MAIN_IMPL_DIM(NAME, 3, PARAMS, OPTS)                                       \
      break;                                                                     \
    default:                                                                     \
      throw std::runtime_error("Unsupported dimension: " + std::to_string(DIM)); \
  }

extern const char* const cov_list;
extern const char* const rbf_cov_list;
