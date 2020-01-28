// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <string>
#include <memory>
#include <stdexcept>
#include <vector>

#include <polatory/polatory.hpp>

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
