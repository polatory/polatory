#pragma once

#include <memory>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::point_cloud {

template <int Dim>
class KdTree {
  using Point = geometry::Point<Dim>;
  using Points = geometry::Points<Dim>;

 public:
  explicit KdTree(const Points& points);

  ~KdTree();

  KdTree(const KdTree&) = delete;
  KdTree(KdTree&&) = delete;
  KdTree& operator=(const KdTree&) = delete;
  KdTree& operator=(KdTree&&) = delete;

  void knn_search(const Point& point, Index k, std::vector<Index>& indices,
                  std::vector<double>& distances) const;

  void radius_search(const Point& point, double radius, std::vector<Index>& indices,
                     std::vector<double>& distances) const;

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

}  // namespace polatory::point_cloud
