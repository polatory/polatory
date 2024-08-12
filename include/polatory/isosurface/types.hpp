#pragma once

#include <polatory/types.hpp>

namespace polatory::isosurface {

using face = Eigen::Matrix<index_t, 1, 3>;
using faces = Eigen::Matrix<index_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

}  // namespace polatory::isosurface
