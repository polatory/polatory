// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <polatory/kriging/empirical_variogram.hpp>
#include <polatory/kriging/weight_functions.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace kriging {

class variogram_fitting {
public:
  variogram_fitting(const empirical_variogram& emp_variog, const rbf::rbf& rbf,
                    rbf::weight_function weight_fn);

  const std::vector<double>& parameters() const;

private:
  std::vector<double> params_;
};

}  // namespace kriging
}  // namespace polatory
