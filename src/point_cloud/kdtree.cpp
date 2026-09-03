#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <nanoflann.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <stdexcept>
#include <vector>

namespace polatory::point_cloud {

template <int Dim>
class KdTree<Dim>::Impl {
  using Point = geometry::Point<Dim>;
  using Points = geometry::Points<Dim>;
  using NanoflannIndex =
      nanoflann::KDTreeEigenMatrixAdaptor<Points, Dim, nanoflann::metric_L2_Simple>;

 public:
  explicit Impl(const Points& points) : points_{points}, nf_index_{Dim, std::cref(points_)} {}

  void knn_search(const Point& point, Index k, std::vector<Index>& indices,
                  std::vector<double>& distances) const {
    KNNResultSet rs{static_cast<std::size_t>(k), indices, distances};
    (void)nf_index_.index_->findNeighbors(rs, point.data());
  }

  void radius_search(const Point& point, double radius, std::vector<Index>& indices,
                     std::vector<double>& distances) const {
    RadiusResultSet rs{radius, indices, distances};
    (void)nf_index_.index_->findNeighbors(rs, point.data());
  }

 private:
  class KNNResultSet {
   public:
    using DistanceType = double;
    using IndexType = Index;

    KNNResultSet(std::size_t k, std::vector<IndexType>& indices,
                 std::vector<DistanceType>& distances)
        : k_{k},
          wd_{k_ > 0 ? std::numeric_limits<DistanceType>::infinity() : 0.0},
          indices_{indices},
          distances_{distances} {
      indices_.clear();
      distances_.clear();
    }

    bool addPoint(DistanceType dist, IndexType index) {
      dist = std::sqrt(dist);
      auto it = std::ranges::upper_bound(distances_, dist);
      auto i = static_cast<std::size_t>(std::distance(distances_.begin(), it));
      if (i < k_) {
        if (full()) {
          indices_.pop_back();
          distances_.pop_back();
        }
        indices_.insert(indices_.begin() + i, index);
        distances_.insert(it, dist);
        if (full()) {
          wd_ = distances_.back() * distances_.back();
        }
      }
      return true;
    }

    bool empty() const noexcept { return indices_.empty(); }

    bool full() const noexcept { return indices_.size() == k_; }

    std::size_t size() const noexcept { return indices_.size(); }

    void sort() {
      // no-op.
    }

    DistanceType worstDist() const noexcept { return wd_; }

   private:
    const std::size_t k_;
    DistanceType wd_;
    std::vector<IndexType>& indices_;
    std::vector<DistanceType>& distances_;
  };

  class RadiusResultSet {
   public:
    using DistanceType = double;
    using IndexType = Index;

    explicit RadiusResultSet(DistanceType radius, std::vector<IndexType>& indices,
                             std::vector<DistanceType>& distances)
        : radius_{radius},
          wd_{std::nextafter(radius * radius, std::numeric_limits<DistanceType>::infinity())},
          indices_{indices},
          distances_{distances} {
      indices_.clear();
      distances_.clear();
    }

    bool addPoint(DistanceType dist, IndexType index) {
      dist = std::sqrt(dist);
      if (dist <= radius_) {
        indices_.push_back(index);
        distances_.push_back(dist);
      }
      return true;
    }

    bool empty() const noexcept { return indices_.empty(); }

    bool full() const noexcept { return true; }

    std::size_t size() const noexcept { return indices_.size(); }

    void sort() {
      // no-op.
    }

    DistanceType worstDist() const noexcept { return wd_; }

   private:
    const DistanceType radius_;
    const DistanceType wd_;
    std::vector<IndexType>& indices_;
    std::vector<DistanceType>& distances_;
  };

  Points points_;
  NanoflannIndex nf_index_;
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
