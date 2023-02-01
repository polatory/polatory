#pragma once

#include <polatory/geometry/point3d.hpp>

namespace polatory::geometry {

class cuboid3d {
 public:
  cuboid3d() : min_(point3d::Zero()), max_(point3d::Ones()) {}

  cuboid3d(const point3d& min, const point3d& max) : min_(min), max_(max) {}

  bool operator==(const cuboid3d& other) const = default;

  const point3d& max() const { return max_; }

  const point3d& min() const { return min_; }

 private:
  const point3d min_;
  const point3d max_;
};

}  // namespace polatory::geometry
