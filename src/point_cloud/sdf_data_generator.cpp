#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/point_cloud/sdf_data_generator.hpp>
#include <stdexcept>
#include <tuple>

namespace polatory::point_cloud {

sdf_data_generator::sdf_data_generator(const geometry::points3d& points,
                                       const geometry::vectors3d& normals, double min_distance,
                                       double max_distance, double multiplication)
    : points_(points), normals_(normals) {
  if (normals.rows() != points.rows()) {
    throw std::invalid_argument("normals.rows() must be equal to points.rows().");
  }

  if (min_distance > max_distance) {
    throw std::invalid_argument("min_distance must be less than or equal to max_distance.");
  }

  if (multiplication <= 1.0 || multiplication > 3.0) {
    throw std::invalid_argument("multiplication must be within (1.0, 3.0].");
  }

  kdtree tree(points, true);

  auto n_points = points.rows();
  auto n_reduced_points =
      static_cast<index_t>(((multiplication - 1.0) / 2.0) * static_cast<double>(n_points));
  auto n_max_sdf_points = n_points + 2 * n_reduced_points;
  auto n_sdf_points = n_points;

  sdf_points_ = geometry::points3d(n_max_sdf_points, 3);
  sdf_points_.topRows(n_points) = points_;
  sdf_values_ = common::valuesd::Zero(n_max_sdf_points);

  for (auto sign : {1.0, -1.0}) {
    for (index_t i = 0; i < n_reduced_points; i++) {
      auto p = points.row(i);
      auto n = normals.row(i);

      if (n == geometry::vector3d::Zero()) {
        continue;
      }

      auto d = max_distance;
      geometry::point3d q = p + sign * d * n;

      auto [nn_indices, nn_distances] = tree.knn_search(q, 1);
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

        std::tie(nn_indices, nn_distances) = tree.knn_search(q, 1);
        i_nearest = nn_indices.at(0);
      }

      if (d < min_distance) {
        continue;
      }

      sdf_points_.row(n_sdf_points) = q;
      sdf_values_(n_sdf_points) = sign * d;
      n_sdf_points++;
    }
  }

  sdf_points_.conservativeResize(n_sdf_points, 3);
  sdf_values_.conservativeResize(n_sdf_points);
}

const geometry::points3d& sdf_data_generator::sdf_points() const { return sdf_points_; }

const common::valuesd& sdf_data_generator::sdf_values() const { return sdf_values_; }

}  // namespace polatory::point_cloud
