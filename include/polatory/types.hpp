#pragma once

#include <Eigen/Core>

namespace polatory {

using Index = Eigen::Index;

template <int M, int N = M>
using Mat = Eigen::Matrix<double, M, N, N == 1 ? Eigen::ColMajor : Eigen::RowMajor>;

using Mat1 = Mat<1>;
using Mat2 = Mat<2>;
using Mat3 = Mat<3>;
using MatX = Mat<Eigen::Dynamic, Eigen::Dynamic>;

template <int N>
using Vec = Mat<N, 1>;

using VecX = Vec<Eigen::Dynamic>;

}  // namespace polatory
