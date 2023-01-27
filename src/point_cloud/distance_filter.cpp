#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <set>
#include <tuple>

namespace polatory {
namespace point_cloud {

distance_filter::distance_filter(const geometry::points3d& points, double distance)
    : n_points_(static_cast<index_t>(points.rows())) {
  if (distance <= 0.0) throw std::invalid_argument("distance must be greater than 0.0.");

  kdtree tree(points, true);

  std::vector<index_t> nn_indices;
  std::vector<double> nn_distances;

  std::set<index_t> indices_to_remove;

  for (index_t i = 0; i < n_points_; i++) {
    if (indices_to_remove.find(i) != indices_to_remove.end()) continue;

    geometry::point3d p = points.row(i);
    std::tie(nn_indices, nn_distances) = tree.radius_search(p, distance);

    for (auto j : nn_indices) {
      if (j != i) {
        indices_to_remove.insert(j);
      }
    }
  }

  for (index_t i = 0; i < n_points_; i++) {
    if (indices_to_remove.find(i) == indices_to_remove.end()) {
      filtered_indices_.push_back(i);
    }
  }
}

}  // namespace point_cloud
}  // namespace polatory
