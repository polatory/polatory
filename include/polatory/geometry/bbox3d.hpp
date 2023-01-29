#pragma once

#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>

namespace polatory::geometry {

class bbox3d {
 public:
  bbox3d();

  bbox3d(const point3d& min, const point3d& max);

  bool operator==(const bbox3d& other) const;

  point3d center() const;

  bool contains(const point3d& p) const;

  const point3d& max() const;

  const point3d& min() const;

  vector3d size() const;

  bbox3d transform(const linear_transformation3d& t) const;  // NOLINT(build/include_what_you_use)

  bbox3d union_hull(const bbox3d& other) const;

  static bbox3d from_points(const points3d& points);

  template <class InputIterator>
  static bbox3d from_points(InputIterator points_begin, InputIterator points_end) {
    bbox3d ret;

    if (points_begin == points_end) {
      return ret;
    }

    auto it = points_begin;
    auto first_pt = *it;
    ret.min_(0) = ret.max_(0) = first_pt(0);
    ret.min_(1) = ret.max_(1) = first_pt(1);
    ret.min_(2) = ret.max_(2) = first_pt(2);
    ++it;

    for (; it != points_end; ++it) {
      auto pt = *it;
      if (ret.min_(0) > pt(0)) [[unlikely]] {
        ret.min_(0) = pt(0);
      }
      if (ret.max_(0) < pt(0)) [[unlikely]] {
        ret.max_(0) = pt(0);
      }
      if (ret.min_(1) > pt(1)) [[unlikely]] {
        ret.min_(1) = pt(1);
      }
      if (ret.max_(1) < pt(1)) [[unlikely]] {
        ret.max_(1) = pt(1);
      }
      if (ret.min_(2) > pt(2)) [[unlikely]] {
        ret.min_(2) = pt(2);
      }
      if (ret.max_(2) < pt(2)) [[unlikely]] {
        ret.max_(2) = pt(2);
      }
    }

    return ret;
  }

 private:
  point3d min_;
  point3d max_;
};

}  // namespace polatory::geometry
