// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <polatory/kriging/empirical_variogram.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace kriging {

enum class variogram_fitting_weights {
  cressie,
  equal,
  npairs
};

class variogram_fitting {
public:
  variogram_fitting(const empirical_variogram& emp_variog, const rbf::rbf& rbf,
                    variogram_fitting_weights weights = variogram_fitting_weights::equal);

  const std::vector<double>& parameters() const;

private:
  std::vector<double> params_;
};

} // namespace kriging
} // namespace polatory
