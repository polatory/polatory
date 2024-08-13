#pragma once

#include <polatory/geometry/point3d.hpp>

namespace polatory::geometry {

class Cuboid3 {
 public:
  Cuboid3() : min_(Point3::Zero()), max_(Point3::Ones()) {}

  Cuboid3(const Point3& min, const Point3& max) : min_(min), max_(max) {}

  bool operator==(const Cuboid3& other) const = default;

  const Point3& max() const { return max_; }

  const Point3& min() const { return min_; }

 private:
  const Point3 min_;
  const Point3 max_;
};

}  // namespace polatory::geometry
