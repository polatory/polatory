// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <utility>
#include <vector>

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace interpolation {

class rbf_inequality_fitter {
public:
  rbf_inequality_fitter(const model& model, const geometry::points3d& points);

  std::pair<std::vector<index_t>, common::valuesd>
  fit(const common::valuesd& values, const common::valuesd& values_lb, const common::valuesd& values_ub,
      double absolute_tolerance) const;

private:
  template <class Predicate>
  static std::vector<index_t> arg_where(const common::valuesd& v, Predicate predicate) {
    std::vector<index_t> idcs;

    auto size = static_cast<index_t>(v.size());
    for (index_t i = 0; i < size; i++) {
      if (predicate(v(i))) {
        idcs.push_back(i);
      }
    }

    return idcs;
  }

  const model& model_;
  const geometry::points3d& points_;

  const index_t n_points_;
  const index_t n_poly_basis_;

  const geometry::bbox3d bbox_;
};

}  // namespace interpolation
}  // namespace polatory
