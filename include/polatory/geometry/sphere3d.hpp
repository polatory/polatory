#pragma once

#include <polatory/geometry/point3d.hpp>

namespace polatory::geometry {

class sphere3d {
 public:
  sphere3d() : center_(point3d::Zero()) {}

  sphere3d(const point3d& center, double radius) : center_(center), radius_(radius) {}

  bool operator==(const sphere3d& other) const = default;

  const point3d& center() const { return center_; }

  double radius() const { return radius_; }

 private:
  const point3d center_;
  const double radius_{1.0};
};

}  // namespace polatory::geometry
