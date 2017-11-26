// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <polatory/common/likely.hpp>
#include <polatory/geometry/affine_transform3d.hpp>
#include <polatory/geometry/point3d.hpp>

namespace polatory {
namespace geometry {

class bbox3d {
public:
  bbox3d();

  bbox3d(const point3d& min, const point3d& max);

  bool operator==(const bbox3d& other) const;

  point3d center() const;

  const point3d& max() const;

  const point3d& min() const;

  vector3d size() const;

  bbox3d transform(const affine_transform3d& affine) const;

  bbox3d union_hull(const bbox3d& other) const;

  static bbox3d from_points(const points3d& points);

  template <class InputIterator>
  static bbox3d from_points(InputIterator points_begin, InputIterator points_end) {
    bbox3d ret;

    if (points_begin == points_end)
      return ret;

    auto it = points_begin;
    auto pt = *it;
    ret.min_(0) = ret.max_(0) = pt(0);
    ret.min_(1) = ret.max_(1) = pt(1);
    ret.min_(2) = ret.max_(2) = pt(2);
    ++it;

    for (; it != points_end; ++it) {
      auto pt = *it;
      if (UNLIKELY(ret.min_(0) > pt(0))) ret.min_(0) = pt(0);
      if (UNLIKELY(ret.max_(0) < pt(0))) ret.max_(0) = pt(0);
      if (UNLIKELY(ret.min_(1) > pt(1))) ret.min_(1) = pt(1);
      if (UNLIKELY(ret.max_(1) < pt(1))) ret.max_(1) = pt(1);
      if (UNLIKELY(ret.min_(2) > pt(2))) ret.min_(2) = pt(2);
      if (UNLIKELY(ret.max_(2) < pt(2))) ret.max_(2) = pt(2);
    }

    return ret;
  }

private:
  point3d min_;
  point3d max_;
};

} // namespace geometry
} // namespace polatory
