#pragma once

#include <polatory/geometry/point3d.hpp>

namespace polatory {
namespace geometry {

class sphere3d {
 public:
  sphere3d() : center_(point3d::Zero()) {}

  sphere3d(const point3d& center, double radius) : center_(center), radius_(radius) {}

  bool operator==(const sphere3d& other) const {
    return center_ == other.center_ && radius_ == other.radius_;
  }

  const point3d& center() const { return center_; }

  double radius() const { return radius_; }

 private:
  const point3d center_;
  const double radius_{1.0};
};

}  // namespace geometry
}  // namespace polatory
