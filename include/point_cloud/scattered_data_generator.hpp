// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <vector>

#include <boost/range/combine.hpp>

#include <Eigen/Core>

#include "kdtree.hpp"

namespace polatory {
namespace point_cloud {

class scattered_data_generator {
   const std::vector<Eigen::Vector3d>& points;
   const std::vector<Eigen::Vector3d>& normals;

   std::vector<size_t> ext_indices;
   std::vector<size_t> int_indices;

   std::vector<double> ext_distances;
   std::vector<double> int_distances;

   size_t total_size() const
   {
      return points.size() + ext_indices.size() + int_indices.size();
   }

public:
   scattered_data_generator(const std::vector<Eigen::Vector3d>& points, const std::vector<Eigen::Vector3d>& normals, double min_distance, double max_distance)
      : points(points)
      , normals(normals)
   {
      assert(points.size() == normals.size());

      kdtree tree(points);
      tree.set_exact_search();

      std::vector<size_t> nn_indices;
      std::vector<double> nn_distances;

      std::vector<size_t> reduced_indices;
      for (size_t i = 0; i < points.size(); i += 2) {
         reduced_indices.push_back(i);
      }

      for (auto i : reduced_indices) {
         const auto& p = points[i];
         const auto& n = normals[i];

         auto d = max_distance;

         Eigen::Vector3d q = p + d * n;
         tree.knn_search(q, 1, nn_indices, nn_distances);
         while (nn_indices[0] != i && nn_distances[0] > 0.0) {
            d = 0.99 * (points[nn_indices[0]] - p).norm() / 2.0;
            q = p + d * n;
            tree.knn_search(q, 1, nn_indices, nn_distances);
         }

         if (d < min_distance)
            continue;

         ext_indices.push_back(i);
         ext_distances.push_back(d);
      }

      for (auto i : reduced_indices) {
         const auto& p = points[i];
         const auto& n = normals[i];

         auto d = max_distance;

         Eigen::Vector3d q = p - d * n;
         tree.knn_search(q, 1, nn_indices, nn_distances);
         while (nn_indices[0] != i && nn_distances[0] > 0.0) {
            d = 0.99 * (points[nn_indices[0]] - p).norm() / 2.0;
            q = p - d * n;
            tree.knn_search(q, 1, nn_indices, nn_distances);
         }

         if (d < min_distance)
            continue;

         int_indices.push_back(i);
         int_distances.push_back(d);
      }
   }

   std::vector<Eigen::Vector3d> scattered_points() const
   {
      std::vector<Eigen::Vector3d> scattered_points(points);
      scattered_points.reserve(total_size());

      for (auto i_d : boost::combine(ext_indices, ext_distances)) {
         size_t i;
         double d;
         boost::tie(i, d) = i_d;

         const auto& p = points[i];
         const auto& n = normals[i];
         scattered_points.push_back(p + d * n);
      }

      for (auto i_d : boost::combine(int_indices, int_distances)) {
         size_t i;
         double d;
         boost::tie(i, d) = i_d;

         const auto& p = points[i];
         const auto& n = normals[i];
         scattered_points.push_back(p - d * n);
      }

      return scattered_points;
   }

   Eigen::VectorXd scattered_values() const
   {
      Eigen::VectorXd values = Eigen::VectorXd::Zero(total_size());

      values.segment(points.size(), ext_indices.size()) =
         Eigen::Map<const Eigen::VectorXd>(ext_distances.data(), ext_indices.size());

      values.tail(int_indices.size()) =
         -Eigen::Map<const Eigen::VectorXd>(int_distances.data(), int_indices.size());

      return values;
   }
};

} // namespace point_cloud
} // namespace polatory
