// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <utility>
#include <vector>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace interpolation {

class rbf_fitter {
public:
  rbf_fitter(const rbf::rbf& rbf, int poly_dimension, int poly_degree,
             const geometry::points3d& points);

  common::valuesd fit(const common::valuesd& values, double absolute_tolerance) const;

private:
  const rbf::rbf rbf_;
  const int poly_dimension_;
  const int poly_degree_;
  const geometry::points3d& points_;
};

}  // namespace interpolation
}  // namespace polatory
