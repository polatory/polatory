// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <iterator>

#include <Eigen/Core>

#include "affine_transform3d.hpp"
#include "polatory/common/likely.hpp"

namespace polatory {
namespace geometry {

class bbox3d {
public:
  bbox3d();

  bbox3d(const Eigen::Vector3d& min, const Eigen::Vector3d& max);

  Eigen::Vector3d center() const;

  const Eigen::Vector3d& max() const;

  const Eigen::Vector3d& min() const;

  Eigen::Vector3d size() const;

  bbox3d transform(const affine_transform3d& affine) const;

  template <class Container>
  static bbox3d from_points(const Container& points) {
    using std::begin;
    using std::end;

    return from_points(begin(points), end(points));
  }

  template <class InputIterator>
  static bbox3d from_points(InputIterator points_begin, InputIterator points_end) {
    bbox3d ret;

    if (points_begin == points_end)
      return ret;

    auto it = points_begin;
    const auto& pt = *it;
    ret.min_(0) = ret.max_(0) = pt(0);
    ret.min_(1) = ret.max_(1) = pt(1);
    ret.min_(2) = ret.max_(2) = pt(2);
    ++it;

    for (; it != points_end; ++it) {
      const auto& pt = *it;
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
  Eigen::Vector3d min_;
  Eigen::Vector3d max_;
};

} // namespace geometry
} // namespace polatory
