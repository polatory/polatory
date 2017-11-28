// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf_base.hpp>

namespace polatory {
namespace kriging {

common::valuesd k_fold_cross_validation(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                                        const geometry::points3d& points, const common::valuesd& values,
                                        double absolute_tolerance,
                                        int k);

} // namespace kriging
} // namespace polatory
