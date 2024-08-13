#pragma once

#include <polatory/geometry/point3d.hpp>

namespace polatory::geometry {

class Sphere3 {
 public:
  Sphere3() : center_(Point3::Zero()) {}

  Sphere3(const Point3& center, double radius) : center_(center), radius_(radius) {}

  bool operator==(const Sphere3& other) const = default;

  const Point3& center() const { return center_; }

  double radius() const { return radius_; }

 private:
  const Point3 center_;
  const double radius_{1.0};
};

}  // namespace polatory::geometry
