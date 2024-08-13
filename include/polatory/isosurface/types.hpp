#pragma once

#include <polatory/types.hpp>

namespace polatory::isosurface {

using Face = Eigen::Matrix<Index, 1, 3>;
using Faces = Eigen::Matrix<Index, Eigen::Dynamic, 3, Eigen::RowMajor>;

}  // namespace polatory::isosurface
