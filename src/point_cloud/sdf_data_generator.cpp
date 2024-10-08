#include <Eigen/LU>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/point_cloud/sdf_data_generator.hpp>
#include <stdexcept>
#include <vector>

namespace polatory::point_cloud {

SdfDataGenerator::SdfDataGenerator(const geometry::Points3& points,
                                   const geometry::Vectors3& normals, double min_distance,
                                   double max_distance)
    : SdfDataGenerator(points, normals, min_distance, max_distance, Mat3::Identity()) {}

SdfDataGenerator::SdfDataGenerator(const geometry::Points3& points,
                                   const geometry::Vectors3& normals, double min_distance,
                                   double max_distance, const Mat3& aniso) {
  if (normals.rows() != points.rows()) {
    throw std::invalid_argument("normals.rows() must be equal to points.rows()");
  }

  if (!(min_distance <= max_distance)) {
    throw std::invalid_argument("min_distance must be less than or equal to max_distance");
  }

  if (!(aniso.determinant() > 0.0)) {
    throw std::invalid_argument("aniso must have a positive determinant");
  }

  if (aniso.isIdentity()) {
    auto [sdf_points, sdf_values] = estimate_impl(points, normals, min_distance, max_distance);
    sdf_points_ = sdf_points;
    sdf_values_ = sdf_values;
  } else {
    auto a_points = geometry::transform_points<3>(aniso, points);
    auto a_normals = geometry::transform_vectors<3>(aniso.inverse().transpose(), normals);
    for (auto n : a_normals.rowwise()) {
      if (!n.isZero()) {
        n = n.normalized();
      }
    }
    auto [sdf_points, sdf_values] = estimate_impl(a_points, a_normals, min_distance, max_distance);
    sdf_points_ = geometry::transform_points<3>(aniso.inverse(), sdf_points);
    sdf_values_ = sdf_values;
  }
}

std::pair<geometry::Points3, VecX> SdfDataGenerator::estimate_impl(
    const geometry::Points3& points, const geometry::Vectors3& normals, double min_distance,
    double max_distance) {
  KdTree tree(points);

  auto n_points = points.rows();
  auto n_max_sdf_points = 3 * n_points;
  auto n_sdf_points = n_points;

  geometry::Points3 sdf_points(n_max_sdf_points, 3);
  sdf_points.topRows(n_points) = points;
  VecX sdf_values = VecX::Zero(n_max_sdf_points);

  std::vector<Index> nn_indices;
  std::vector<double> nn_distances;

  for (auto sign : {-1.0, 1.0}) {
    for (Index i = 0; i < n_points; i++) {
      auto p = points.row(i);
      auto n = normals.row(i);

      if (n.isZero()) {
        continue;
      }

      auto d = max_distance;
      geometry::Point3 q = p + sign * d * n;

      tree.knn_search(q, 1, nn_indices, nn_distances);
      auto i_nearest = nn_indices.at(0);

      while (i_nearest != i) {
        auto p_nearest = points.row(i_nearest);
        auto r = (p_nearest - p).norm() / 2.0;
        auto cos = (q - p).normalized().dot((p_nearest - p).normalized());

        d = 0.99 * r / cos;
        q = p + sign * d * n;

        if (d < min_distance) {
          break;
        }

        tree.knn_search(q, 1, nn_indices, nn_distances);
        i_nearest = nn_indices.at(0);
      }

      if (d < min_distance) {
        continue;
      }

      sdf_points.row(n_sdf_points) = q;
      sdf_values(n_sdf_points) = sign * d;
      n_sdf_points++;
    }
  }

  sdf_points.conservativeResize(n_sdf_points, 3);
  sdf_values.conservativeResize(n_sdf_points);

  return {sdf_points, sdf_values};
}

const geometry::Points3& SdfDataGenerator::sdf_points() const { return sdf_points_; }

const VecX& SdfDataGenerator::sdf_values() const { return sdf_values_; }

}  // namespace polatory::point_cloud
