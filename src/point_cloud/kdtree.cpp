#include <cmath>
#include <flann/flann.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <stdexcept>

namespace polatory {
namespace point_cloud {

class kdtree::impl {
  using FlannIndex = flann::Index<flann::L2<double>>;

 public:
  using indices_and_distances = std::pair<std::vector<index_t>, std::vector<double>>;

  impl(const geometry::points3d& points, bool use_exact_search) {
    flann::Matrix<double> points_mat(const_cast<double*>(points.data()), points.rows(),
                                     3);  // NOLINT(cppcoreguidelines-pro-type-const-cast)

    flann_index_ = std::make_unique<FlannIndex>(points_mat, flann::KDTreeSingleIndexParams());
    flann_index_->buildIndex();

    if (use_exact_search) {
      params_knn_.checks = flann::FLANN_CHECKS_UNLIMITED;
      params_radius_.checks = flann::FLANN_CHECKS_UNLIMITED;
    }
  }

  indices_and_distances knn_search(const geometry::point3d& point, index_t k) const {
    flann::Matrix<double> point_mat(const_cast<double*>(point.data()), 1,
                                    3);  // NOLINT(cppcoreguidelines-pro-type-const-cast)
    std::vector<std::vector<size_t>> indices_v;
    std::vector<std::vector<double>> distances_v;

    (void)flann_index_->knnSearch(point_mat, indices_v, distances_v, k, params_knn_);

    std::vector<index_t> indices;
    indices.reserve(indices_v[0].size());

    for (auto i : indices_v[0]) {
      indices.push_back(static_cast<index_t>(i));
    }

    for (auto& d : distances_v[0]) {
      d = std::sqrt(d);
    }

    return {std::move(indices), std::move(distances_v[0])};
  }

  indices_and_distances radius_search(const geometry::point3d& point, double radius) const {
    flann::Matrix<double> point_mat(const_cast<double*>(point.data()), 1,
                                    3);  // NOLINT(cppcoreguidelines-pro-type-const-cast)
    std::vector<std::vector<size_t>> indices_v;
    std::vector<std::vector<double>> distances_v;

    auto radius_sq = static_cast<float>(radius * radius);
    (void)flann_index_->radiusSearch(point_mat, indices_v, distances_v, radius_sq, params_radius_);

    std::vector<index_t> indices;
    indices.reserve(indices_v[0].size());

    for (auto i : indices_v[0]) {
      indices.push_back(static_cast<index_t>(i));
    }

    for (auto& d : distances_v[0]) {
      d = std::sqrt(d);
    }

    return {std::move(indices), std::move(distances_v[0])};
  }

 private:
  flann::SearchParams params_knn_;
  flann::SearchParams params_radius_;
  std::unique_ptr<FlannIndex> flann_index_;
};

kdtree::kdtree(const geometry::points3d& points, bool use_exact_search)
    : pimpl_(points.rows() == 0 ? nullptr : std::make_unique<impl>(points, use_exact_search)) {}

kdtree::~kdtree() = default;

kdtree::indices_and_distances kdtree::knn_search(const geometry::point3d& point, index_t k) const {
  if (k <= 0) throw std::invalid_argument("k must be greater than 0.");

  if (!pimpl_) return {};

  return pimpl_->knn_search(point, k);
}

kdtree::indices_and_distances kdtree::radius_search(const geometry::point3d& point,
                                                    double radius) const {
  if (radius <= 0.0) throw std::invalid_argument("radius must be greater than 0.0.");

  if (!pimpl_) return {};

  return pimpl_->radius_search(point, radius);
}

}  // namespace point_cloud
}  // namespace polatory
