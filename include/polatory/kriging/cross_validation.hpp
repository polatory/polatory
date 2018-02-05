// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>

namespace polatory {
namespace kriging {

common::valuesd k_fold_cross_validation(const model& model,
                                        const geometry::points3d& points, const common::valuesd& values,
                                        double absolute_tolerance,
                                        size_t k);

}  // namespace kriging
}  // namespace polatory
