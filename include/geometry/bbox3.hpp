// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <iterator>

#include <Eigen/Core>

#include "affine_transform.hpp"
#include "../common/array.hpp"
#include "../common/likely.hpp"

namespace polatory {
namespace geometry {

class bbox3d {
public:
   Eigen::Vector3d min;
   Eigen::Vector3d max;

   bbox3d()
      : min()
      , max()
   {
   }

   bbox3d(const Eigen::Vector3d& min, const Eigen::Vector3d& max)
      : min(min)
      , max(max)
   {
   }

   bbox3d affine_transform(const Eigen::Matrix4d& m) const
   {
      Eigen::Vector3d center = (min + max) / 2.0;
      Eigen::Vector3d v1 = Eigen::Vector3d(min[0], max[1], max[2]) - center;
      Eigen::Vector3d v2 = Eigen::Vector3d(max[0], min[1], max[2]) - center;
      Eigen::Vector3d v3 = Eigen::Vector3d(max[0], max[1], min[2]) - center;

      center = affine_transform_point(center, m);
      v1 = affine_transform_vector(v1, m);
      v2 = affine_transform_vector(v2, m);
      v3 = affine_transform_vector(v3, m);

      Eigen::MatrixXd vertices(3, 8);
      vertices.col(0) = -v1 - v2 - v3;    // min, min, min
      vertices.col(1) = -v1;              // max, min, min
      vertices.col(2) = v3;               // max, max, min
      vertices.col(3) = -v2;              // min, max, min
      vertices.col(4) = v1;               // min, max, max
      vertices.col(5) = -v3;              // min, min, max
      vertices.col(6) = v2;               // max, min, max
      vertices.col(7) = v1 + v2 + v3;     // max, max, max

      Eigen::Vector3d min = center + Eigen::Vector3d(
         vertices.row(0).minCoeff(),
         vertices.row(1).minCoeff(),
         vertices.row(2).minCoeff()
      );
      Eigen::Vector3d max = center + Eigen::Vector3d(
         vertices.row(0).maxCoeff(),
         vertices.row(1).maxCoeff(),
         vertices.row(2).maxCoeff()
      );

      return bbox3d(min, max);
   }

   int longest_axis() const
   {
      auto lengths = common::make_array(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
      return std::distance(lengths.begin(), std::max_element(lengths.begin(), lengths.end()));
   }

   template<typename Container>
   static bbox3d from_points(const Container& points)
   {
      using std::begin;
      using std::end;

      return from_points(begin(points), end(points));
   }

   template<typename InputIterator>
   static bbox3d from_points(InputIterator points_begin, InputIterator points_end)
   {
      bbox3d ret;

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

} // namespace geometry
} // namespace polatory
