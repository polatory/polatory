#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace kriging {

common::valuesd k_fold_cross_validation(const model& model, const geometry::points3d& points,
                                        const common::valuesd& values, double absolute_tolerance,
                                        index_t k);

}  // namespace kriging
}  // namespace polatory
