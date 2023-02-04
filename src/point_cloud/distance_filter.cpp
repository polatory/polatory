#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <unordered_set>

namespace polatory::point_cloud {

distance_filter::distance_filter(const geometry::points3d& points, double distance)
    : distance_filter(points, distance, trivial_indices(points.rows())) {}

distance_filter::distance_filter(const geometry::points3d& points, double distance,
                                 const std::vector<index_t>& indices)
    : n_points_(points.rows()) {
  if (distance <= 0.0) {
    throw std::invalid_argument("distance must be greater than 0.0.");
  }

  kdtree tree(points, true);

  std::unordered_set<index_t> indices_to_remove;

  std::vector<index_t> nn_indices;
  std::vector<double> nn_distances;

  for (auto i : indices) {
    if (indices_to_remove.contains(i)) {
      continue;
    }

    geometry::point3d p = points.row(i);
    tree.radius_search(p, distance, nn_indices, nn_distances);

    for (auto j : nn_indices) {
      if (j != i) {
        indices_to_remove.insert(j);
      }
    }
  }

  for (auto i : indices) {
    if (!indices_to_remove.contains(i)) {
      filtered_indices_.push_back(i);
    }
  }
}

}  // namespace polatory::point_cloud
