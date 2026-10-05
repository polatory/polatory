#pragma once

#include <jizai/types.hpp>

namespace jizai::isosurface {

using Face = Eigen::Matrix<Index, 1, 3>;
using Faces = Eigen::Matrix<Index, Eigen::Dynamic, 3, Eigen::RowMajor>;

}  // namespace jizai::isosurface
