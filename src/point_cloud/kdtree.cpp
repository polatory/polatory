// https://github.com/flann-lib/flann/issues/386
using pop_t = unsigned long long;

#include <algorithm>
#include <cmath>
#include <flann/flann.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <stdexcept>

namespace polatory::point_cloud {

template <int Dim>
class KdTree<Dim>::Impl {
  using FlannIndex = flann::Index<flann::L2<double>>;
  using Point = geometry::Point<Dim>;
  using Points = geometry::Points<Dim>;

 public:
  explicit Impl(const Points& points) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    flann::Matrix<double> points_mat(const_cast<double*>(points.data()), points.rows(), Dim);

    flann_index_ = std::make_unique<FlannIndex>(points_mat, flann::KDTreeSingleIndexParams());
    flann_index_->buildIndex();

    search_params_.checks = flann::FLANN_CHECKS_UNLIMITED;
    search_params_.sorted = false;
  }

  void knn_search(const Point& point, Index k, std::vector<Index>& indices,
                  std::vector<double>& distances) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    flann::Matrix<double> point_mat(const_cast<double*>(point.data()), 1, Dim);

    (void)flann_index_->knnSearch(point_mat, indices_v_, distances_v_, k, search_params_);

    indices.resize(indices_v_[0].size());
    distances.resize(distances_v_[0].size());

    std::transform(indices_v_[0].begin(), indices_v_[0].end(), indices.begin(),
                   [](auto i) { return static_cast<Index>(i); });

    std::transform(distances_v_[0].begin(), distances_v_[0].end(), distances.begin(),
                   [](auto d) { return std::sqrt(d); });
  }

  void radius_search(const Point& point, double radius, std::vector<Index>& indices,
                     std::vector<double>& distances) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    flann::Matrix<double> point_mat(const_cast<double*>(point.data()), 1, Dim);

    auto radius_sq = static_cast<float>(radius * radius);
    (void)flann_index_->radiusSearch(point_mat, indices_v_, distances_v_, radius_sq,
                                     search_params_);

    indices.resize(indices_v_[0].size());
    distances.resize(distances_v_[0].size());

    std::transform(indices_v_[0].begin(), indices_v_[0].end(), indices.begin(),
                   [](auto i) { return static_cast<Index>(i); });

    std::transform(distances_v_[0].begin(), distances_v_[0].end(), distances.begin(),
                   [](auto d) { return std::sqrt(d); });
  }

 private:
  static thread_local inline std::vector<std::vector<std::size_t>> indices_v_;
  static thread_local inline std::vector<std::vector<double>> distances_v_;
  flann::SearchParams search_params_;
  std::unique_ptr<FlannIndex> flann_index_;
};

template <int Dim>
KdTree<Dim>::KdTree(const Points& points)
    : impl_(points.rows() == 0 ? nullptr : std::make_unique<Impl>(points)) {}

template <int Dim>
KdTree<Dim>::~KdTree() = default;

template <int Dim>
void KdTree<Dim>::knn_search(const Point& point, Index k, std::vector<Index>& indices,
                             std::vector<double>& distances) const {
  if (k <= 0) {
    throw std::invalid_argument("k must be positive");
  }

  if (!impl_) {
    return;
  }

  impl_->knn_search(point, k, indices, distances);
}

template <int Dim>
void KdTree<Dim>::radius_search(const Point& point, double radius, std::vector<Index>& indices,
                                std::vector<double>& distances) const {
  if (!(radius >= 0.0)) {
    throw std::invalid_argument("radius must be non-negative");
  }

  if (!impl_) {
    return;
  }

  impl_->radius_search(point, radius, indices, distances);
}

template class KdTree<1>;
template class KdTree<2>;
template class KdTree<3>;

}  // namespace polatory::point_cloud
