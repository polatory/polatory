// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <string>
#include <memory>
#include <stdexcept>
#include <vector>

#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <polatory/polatory.hpp>

namespace Eigen {

inline
void validate(boost::any& v, const std::vector<std::string>& values,  // NOLINT(runtime/references)
  polatory::geometry::linear_transformation3d*, int) {
  namespace po = boost::program_options;

  if (values.size() != 9) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  polatory::geometry::linear_transformation3d aniso;
  aniso <<
    boost::lexical_cast<double>(values[0]),
    boost::lexical_cast<double>(values[1]),
    boost::lexical_cast<double>(values[2]),
    boost::lexical_cast<double>(values[3]),
    boost::lexical_cast<double>(values[4]),
    boost::lexical_cast<double>(values[5]),
    boost::lexical_cast<double>(values[6]),
    boost::lexical_cast<double>(values[7]),
    boost::lexical_cast<double>(values[8]);

  v = aniso;
}

}  // namespace Eigen

namespace polatory {
namespace geometry {

inline
void validate(boost::any& v, const std::vector<std::string>& values,  // NOLINT(runtime/references)
  bbox3d*, int) {
  namespace po = boost::program_options;

  if (values.size() != 6) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  v = bbox3d(
    { boost::lexical_cast<double>(values[0]), boost::lexical_cast<double>(values[1]), boost::lexical_cast<double>(values[2]) },
    { boost::lexical_cast<double>(values[3]), boost::lexical_cast<double>(values[4]), boost::lexical_cast<double>(values[5]) });
}

}  // namespace geometry
}  // namespace polatory

inline
std::unique_ptr<polatory::rbf::rbf_base> make_rbf(const std::string& name, const std::vector<double>& params) {
  if (name == "bh2") {
    return std::make_unique<polatory::rbf::biharmonic2d>(params);
  }

  if (name == "bh3") {
    return std::make_unique<polatory::rbf::biharmonic3d>(params);
  }

  if (name == "exp") {
    return std::make_unique<polatory::rbf::cov_exponential>(params);
  }

  if (name == "sp3") {
    return std::make_unique<polatory::rbf::cov_spheroidal3>(params);
  }

  if (name == "sp5") {
    return std::make_unique<polatory::rbf::cov_spheroidal5>(params);
  }

  if (name == "sp7") {
    return std::make_unique<polatory::rbf::cov_spheroidal7>(params);
  }

  if (name == "sp9") {
    return std::make_unique<polatory::rbf::cov_spheroidal9>(params);
  }

  throw std::invalid_argument("Unknown RBF name: " + name);
}

extern const char *const cov_list;
extern const char *const rbf_cov_list;
