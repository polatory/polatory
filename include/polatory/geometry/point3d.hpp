#pragma once

#include <Eigen/Core>
#include <polatory/types.hpp>

namespace polatory::geometry {

template <int Dim>
using Vector = Mat<1, Dim>;

using Vector1 = Vector<1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template <int Dim>
using Vectors = Mat<Eigen::Dynamic, Dim>;

using Vectors1 = Vectors<1>;
using Vectors2 = Vectors<2>;
using Vectors3 = Vectors<3>;

template <int Dim>
using Point = Mat<1, Dim>;

using Point1 = Point<1>;
using Point2 = Point<2>;
using Point3 = Point<3>;

template <int Dim>
using Points = Mat<Eigen::Dynamic, Dim>;

using Points1 = Points<1>;
using Points2 = Points<2>;
using Points3 = Points<3>;

template <int Dim, class DerivedT, class DerivedPoint>
Point<Dim> transform_point(const Eigen::MatrixBase<DerivedT>& t,
                           const Eigen::MatrixBase<DerivedPoint>& point) {
  return point * t.transpose();
}

template <int Dim, class DerivedT, class DerivedPoints>
Points<Dim> transform_points(const Eigen::MatrixBase<DerivedT>& t,
                             const Eigen::MatrixBase<DerivedPoints>& points) {
  return points * t.transpose();
}

template <int Dim, class DerivedT, class DerivedVector>
Vector<Dim> transform_vector(const Eigen::MatrixBase<DerivedT>& t,
                             const Eigen::MatrixBase<DerivedVector>& vector) {
  return vector * t.transpose();
}

template <int Dim, class DerivedT, class DerivedVectors>
Vectors<Dim> transform_vectors(const Eigen::MatrixBase<DerivedT>& t,
                               const Eigen::MatrixBase<DerivedVectors>& vectors) {
  return vectors * t.transpose();
}

}  // namespace polatory::geometry
