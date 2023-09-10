#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/point_cloud/offset_points_generator.hpp>
#include <polatory/types.hpp>
#include <stdexcept>

namespace polatory::point_cloud {

offset_points_generator::offset_points_generator(const geometry::points3d& points,
                                                 const geometry::vectors3d& normals, double offset)
    : offset_points_generator(points, normals, common::valuesd::Constant(points.rows(), offset)) {}

offset_points_generator::offset_points_generator(const geometry::points3d& points,
                                                 const geometry::vectors3d& normals,
                                                 const common::valuesd& offsets) {
  if (normals.rows() != points.rows()) {
    throw std::invalid_argument("normals.rows() must be equal to points.rows().");
  }

  if (offsets.size() != points.rows()) {
    throw std::invalid_argument("offsets.size() must be equal to points.rows().");
  }

  kdtree tree(points, true);

  auto n_points = points.rows();
  auto n_new_points = 0;

  new_points_ = geometry::points3d(n_points, 3);
  new_normals_ = geometry::vectors3d(n_points, 3);

  std::vector<index_t> nn_indices;
  std::vector<double> nn_distances;

  for (index_t i = 0; i < n_points; i++) {
    auto p = points.row(i);
    auto n = normals.row(i);

    if (n == geometry::vector3d::Zero()) {
      continue;
    }

    auto d = offsets(i);
    geometry::point3d q = p + d * n;

    tree.knn_search(q, 1, nn_indices, nn_distances);
    auto i_nearest = nn_indices.at(0);

    if (i_nearest != i) {
      continue;
    }

    new_points_.row(n_new_points) = q;
    new_normals_.row(n_new_points) = n;
    n_new_points++;
  }

  new_points_.conservativeResize(n_new_points, 3);
  new_normals_.conservativeResize(n_new_points, 3);
}

const geometry::points3d& offset_points_generator::new_points() const { return new_points_; }

const geometry::vectors3d& offset_points_generator::new_normals() const { return new_normals_; }

}  // namespace polatory::point_cloud
