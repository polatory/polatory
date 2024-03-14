#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace polatory::geometry {

template <int Dim>
using vectorNd = Eigen::Matrix<double, 1, Dim>;

using vector1d = vectorNd<1>;
using vector2d = vectorNd<2>;
using vector3d = vectorNd<3>;

template <int Dim>
using vectorsNd =
    Eigen::Matrix<double, Eigen::Dynamic, Dim, Dim == 1 ? Eigen::ColMajor : Eigen::RowMajor>;

using vectors1d = vectorsNd<1>;
using vectors2d = vectorsNd<2>;
using vectors3d = vectorsNd<3>;

template <int Dim>
using pointNd = Eigen::Matrix<double, 1, Dim>;

using point1d = pointNd<1>;
using point2d = pointNd<2>;
using point3d = pointNd<3>;

template <int Dim>
using pointsNd =
    Eigen::Matrix<double, Eigen::Dynamic, Dim, Dim == 1 ? Eigen::ColMajor : Eigen::RowMajor>;

using points1d = pointsNd<1>;
using points2d = pointsNd<2>;
using points3d = pointsNd<3>;

template <int Dim>
using matrixNd = Eigen::Matrix<double, Dim, Dim, Dim == 1 ? Eigen::ColMajor : Eigen::RowMajor>;

using matrix1d = matrixNd<1>;
using matrix2d = matrixNd<2>;
using matrix3d = matrixNd<3>;

template <class T>
matrix3d to_matrix3d(T t) {
  return Eigen::Transform<double, 3, Eigen::Affine, Eigen::RowMajor>(t).linear();
}

template <int Dim, class DerivedT, class DerivedPoint>
pointNd<Dim> transform_point(const Eigen::MatrixBase<DerivedT>& t,
                             const Eigen::MatrixBase<DerivedPoint>& point) {
  return (t * point.transpose()).transpose();
}

template <int Dim, class DerivedT, class DerivedPoints>
pointsNd<Dim> transform_points(const Eigen::MatrixBase<DerivedT>& t,
                               const Eigen::MatrixBase<DerivedPoints>& points) {
  return (t * points.transpose()).transpose();
}

template <int Dim, class DerivedT, class DerivedVector>
vectorNd<Dim> transform_vector(const Eigen::MatrixBase<DerivedT>& t,
                               const Eigen::MatrixBase<DerivedVector>& vector) {
  return (t * vector.transpose()).transpose();
}

template <int Dim, class DerivedT, class DerivedVectors>
vectorsNd<Dim> transform_vectors(const Eigen::MatrixBase<DerivedT>& t,
                                 const Eigen::MatrixBase<DerivedVectors>& vectors) {
  return (t * vectors.transpose()).transpose();
}

}  // namespace polatory::geometry
