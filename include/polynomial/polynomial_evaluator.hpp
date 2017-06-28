// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <cassert>
#include <iterator>
#include <vector>

#include <Eigen/Core>

namespace polatory {
namespace polynomial {

template<class Basis>
class polynomial_evaluator {
   const Basis basis;

   std::vector<Eigen::Vector3d> points;
   Eigen::VectorXd weights;

public:
   explicit polynomial_evaluator(int degree)
      : basis(degree)
      , weights(Eigen::VectorXd::Zero(basis.dimension()))
   {
   }

   Eigen::VectorXd evaluate() const
   {
      Eigen::MatrixXd pt = basis.evaluate_points(points);

      return pt.transpose() * weights;
   }

   template<typename Container>
   void set_field_points(const Container& points)
   {
      this->points.clear();
      this->points.reserve(points.size());
      std::copy(points.begin(), points.end(), std::back_inserter(this->points));
   }

   void set_weights(const Eigen::VectorXd& weights)
   {
      assert(weights.size() == basis.dimension());

      this->weights = weights;
   }

   size_t size() const
   {
      return basis.dimension();
   }
};

} // namespace polynomial
} // namespace polatory
