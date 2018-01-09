// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <functional>
#include <utility>
#include <vector>

#include <polatory/common/types.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace interpolation {

class rbf_inequality_fitter {
public:
  rbf_inequality_fitter(const rbf::rbf& rbf, int poly_dimension, int poly_degree,
                        const geometry::points3d& points);

  std::pair<std::vector<size_t>, common::valuesd>
  fit(const common::valuesd& values, const common::valuesd& values_lb, const common::valuesd& values_ub,
      double absolute_tolerance) const;

private:
  static std::vector<size_t> arg_where(const common::valuesd& v,
                                       std::function<bool(double)> predicate);

  const rbf::rbf rbf_;
  const int poly_dimension_;
  const int poly_degree_;
  const geometry::points3d& points_;

  const size_t n_points_;
  const size_t n_poly_basis_;

  const geometry::bbox3d bbox_;
};

}  // namespace interpolation
}  // namespace polatory
