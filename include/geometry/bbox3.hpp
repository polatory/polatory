// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <iterator>

#include <Eigen/Core>

#include "../common/array.hpp"
#include "../common/likely.hpp"

namespace polatory {
namespace geometry {

template<class Point>
class bbox3_base {
public:
   Point min;
   Point max;

   bbox3_base()
      : min()
      , max()
   {
   }

   bbox3_base(const Point& min, const Point& max)
      : min(min)
      , max(max)
   {
   }

   int longest_axis() const
   {
      auto lengths = common::make_array(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
      return std::distance(lengths.begin(), std::max_element(lengths.begin(), lengths.end()));
   }

   template<typename Container>
   static bbox3_base from_points(const Container& points)
   {
      using std::begin;
      using std::end;

      return from_points(begin(points), end(points));
   }

   template<typename InputIterator>
   static bbox3_base from_points(InputIterator points_begin, InputIterator points_end)
   {
      bbox3_base ret;

      if (points_begin == points_end)
         return ret;

      auto it = points_begin;
      const auto& pt = *it;
      ret.min[0] = ret.max[0] = pt[0];
      ret.min[1] = ret.max[1] = pt[1];
      ret.min[2] = ret.max[2] = pt[2];
      ++it;

      for (; it != points_end; ++it) {
         const auto& pt = *it;
         if (UNLIKELY(ret.min[0] > pt[0])) ret.min[0] = pt[0];
         if (UNLIKELY(ret.max[0] < pt[0])) ret.max[0] = pt[0];
         if (UNLIKELY(ret.min[1] > pt[1])) ret.min[1] = pt[1];
         if (UNLIKELY(ret.max[1] < pt[1])) ret.max[1] = pt[1];
         if (UNLIKELY(ret.min[2] > pt[2])) ret.min[2] = pt[2];
         if (UNLIKELY(ret.max[2] < pt[2])) ret.max[2] = pt[2];
      }

      return ret;
   }
};

using bbox3d = bbox3_base<Eigen::Vector3d>;

} // namespace geometry
} // namespace polatory
