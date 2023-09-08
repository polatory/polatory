#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

class rbf_fitter {
 public:
  rbf_fitter(const model& model, const geometry::points3d& points);

  common::valuesd fit(const common::valuesd& values, double absolute_tolerance, int max_iter) const;

 private:
  const model& model_;
  const geometry::points3d& points_;
};

}  // namespace polatory::interpolation
