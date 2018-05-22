// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>

namespace polatory {
namespace interpolation {

class rbf_fitter {
public:
  rbf_fitter(const model& model, const geometry::points3d& points);

  common::valuesd fit(const common::valuesd& values, double absolute_tolerance) const;

private:
  const model model_;
  const geometry::points3d& points_;
};

}  // namespace interpolation
}  // namespace polatory
