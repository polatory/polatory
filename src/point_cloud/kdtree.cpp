// https://github.com/flann-lib/flann/issues/386
using pop_t = unsigned long long;

#include <algorithm>
#include <cmath>
#include <flann/flann.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <stdexcept>

namespace polatory::point_cloud {

class kdtree::impl {
  using FlannIndex = flann::Index<flann::L2<double>>;

 public:
  impl(const geometry::points3d& points, bool use_exact_search) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    flann::Matrix<double> points_mat(const_cast<double*>(points.data()), points.rows(), 3);

    flann_index_ = std::make_unique<FlannIndex>(points_mat, flann::KDTreeSingleIndexParams());
    flann_index_->buildIndex();

    if (use_exact_search) {
      params_knn_.checks = flann::FLANN_CHECKS_UNLIMITED;
      params_radius_.checks = flann::FLANN_CHECKS_UNLIMITED;
    }
  }

  void knn_search(const geometry::point3d& point, index_t k, std::vector<index_t>& indices,
                  std::vector<double>& distances) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    flann::Matrix<double> point_mat(const_cast<double*>(point.data()), 1, 3);

    (void)flann_index_->knnSearch(point_mat, indices_v_, distances_v_, k, params_knn_);

    indices.resize(indices_v_[0].size());
    distances.resize(distances_v_[0].size());

    std::transform(indices_v_[0].begin(), indices_v_[0].end(), indices.begin(),
                   [](auto i) { return static_cast<index_t>(i); });

    std::transform(distances_v_[0].begin(), distances_v_[0].end(), distances.begin(),
                   [](auto d) { return std::sqrt(d); });
  }

  void radius_search(const geometry::point3d& point, double radius, std::vector<index_t>& indices,
                     std::vector<double>& distances) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    flann::Matrix<double> point_mat(const_cast<double*>(point.data()), 1, 3);

    auto radius_sq = static_cast<float>(radius * radius);
    (void)flann_index_->radiusSearch(point_mat, indices_v_, distances_v_, radius_sq,
                                     params_radius_);

    indices.resize(indices_v_[0].size());
    distances.resize(distances_v_[0].size());

    std::transform(indices_v_[0].begin(), indices_v_[0].end(), indices.begin(),
                   [](auto i) { return static_cast<index_t>(i); });

    std::transform(distances_v_[0].begin(), distances_v_[0].end(), distances.begin(),
                   [](auto d) { return std::sqrt(d); });
  }

 private:
  static thread_local inline std::vector<std::vector<std::size_t>> indices_v_;
  static thread_local inline std::vector<std::vector<double>> distances_v_;
  flann::SearchParams params_knn_;
  flann::SearchParams params_radius_;
  std::unique_ptr<FlannIndex> flann_index_;
};

kdtree::kdtree(const geometry::points3d& points, bool use_exact_search)
    : pimpl_(points.rows() == 0 ? nullptr : std::make_unique<impl>(points, use_exact_search)) {}

kdtree::~kdtree() = default;

void kdtree::knn_search(const geometry::point3d& point, index_t k, std::vector<index_t>& indices,
                        std::vector<double>& distances) const {
  if (k <= 0) {
    throw std::invalid_argument("k must be greater than 0.");
  }

  if (!pimpl_) {
    return;
  }

  pimpl_->knn_search(point, k, indices, distances);
}

void kdtree::radius_search(const geometry::point3d& point, double radius,
                           std::vector<index_t>& indices, std::vector<double>& distances) const {
  if (radius <= 0.0) {
    throw std::invalid_argument("radius must be greater than 0.0.");
  }

  if (!pimpl_) {
    return;
  }

  pimpl_->radius_search(point, radius, indices, distances);
}

}  // namespace polatory::point_cloud
