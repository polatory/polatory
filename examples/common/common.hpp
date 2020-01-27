#pragma once

#include <string>
#include <memory>
#include <stdexcept>
#include <vector>

#include <polatory/polatory.hpp>

inline
std::unique_ptr<polatory::rbf::rbf_base> make_rbf(const std::string& name, const std::vector<double>& params) {
  using namespace polatory::rbf;

  if (name == "bh2") {
    return std::make_unique<biharmonic2d>(params);
  }

  if (name == "bh3") {
    return std::make_unique<biharmonic3d>(params);
  }

  if (name == "exp") {
    return std::make_unique<cov_exponential>(params);
  }

  if (name == "sp3") {
    return std::make_unique<cov_spheroidal3>(params);
  }

  if (name == "sp5") {
    return std::make_unique<cov_spheroidal5>(params);
  }

  if (name == "sp7") {
    return std::make_unique<cov_spheroidal7>(params);
  }

  if (name == "sp9") {
    return std::make_unique<cov_spheroidal9>(params);
  }

  throw std::invalid_argument("Unknown RBF name: " + name);
}

extern const std::string cov_list;
extern const std::string spline_list;
