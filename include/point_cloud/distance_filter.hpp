// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <set>
#include <vector>

#include <Eigen/Core>

#include "common/vector_view.hpp"
#include "kdtree.hpp"

namespace polatory {
namespace point_cloud {

class distance_filter {
   const std::vector<Eigen::Vector3d>& points;
   const size_t n_points;

   std::vector<size_t> filtered_indices;

public:
   distance_filter(const std::vector<Eigen::Vector3d>& points, double distance)
      : points(points)
      , n_points(points.size())
   {
      kdtree tree(points);
      tree.set_exact_search();

      std::vector<size_t> nn_indices;
      std::vector<double> nn_distances;

      std::set<size_t> indices_to_remove;

      for (size_t i = 0; i < n_points; i++) {
         auto found = tree.radius_search(points[i], distance, nn_indices, nn_distances);

         for (int k = 0; k < found; k++) {
            auto j = nn_indices[k];

            if (j != i) {
               indices_to_remove.insert(j);
            }
         }
      }

      for (size_t i = 0; i < n_points; i++) {
         if (indices_to_remove.count(i) == 0) {
            filtered_indices.push_back(i);
         }
      }
   }

   std::vector<Eigen::Vector3d> filtered_points() const
   {
      auto view = common::make_view(points, filtered_indices);

      return std::vector<Eigen::Vector3d>(view.begin(), view.end());
   }

   template<typename Derived>
   Eigen::VectorXd filter_values(Eigen::MatrixBase<Derived>& values) const
   {
      assert(values.size() == n_points);

      Eigen::VectorXd filtered(filtered_indices.size());

      for (size_t i = 0; i < filtered_indices.size(); i++) {
         filtered(i) = values(filtered_indices[i]);
      }

      return filtered;
   }
};

} // namespace point_cloud
} // namespace polatory
