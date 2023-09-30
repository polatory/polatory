#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace polatory::geometry {

template <int Dim>
using vectorNd = Eigen::Matrix<double, 1, Dim>;

template <int Dim>
using matrixNd = Eigen::Matrix<double, Dim, Dim, Eigen::RowMajor>;

template <int Dim>
using vectorsNd = Eigen::Matrix<double, Eigen::Dynamic, Dim, Eigen::RowMajor>;

using vector2d = vectorNd<2>;

using point2d = vector2d;

using vector3d = vectorNd<3>;

using point3d = vector3d;

using vectors3d = vectorsNd<3>;

using points3d = vectors3d;

using matrix3d = matrixNd<3>;

using vectorXd = Eigen::RowVectorXd;

using matrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using linear_transformation3d = matrix3d;

template <class T>
linear_transformation3d to_linear_transformation3d(T t) {
  return Eigen::Transform<double, 3, Eigen::Affine, Eigen::RowMajor>(t).linear();
}

inline point3d transform_point(const linear_transformation3d& t, const point3d& p) {
  return t * p.transpose();
}

inline vector3d transform_vector(const linear_transformation3d& t, const vector3d& v) {
  return t * v.transpose();
}

}  // namespace polatory::geometry
