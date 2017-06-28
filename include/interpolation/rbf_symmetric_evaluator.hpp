// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "../fmm.hpp"
#include "../geometry/bbox3.hpp"
#include "../polynomial.hpp"
#include "../rbf/rbf_base.hpp"

namespace polatory {
namespace interpolation {

template<int Order = 10>
class rbf_symmetric_evaluator {
   typedef polynomial::polynomial_evaluator<polynomial::monomial_basis<>> poly_eval;

   const rbf::rbf_base& rbf;
   const int poly_degree;
   const size_t n_points;
   const size_t n_polynomials;

   std::unique_ptr<fmm::fmm_operator<Order>> a;
   std::unique_ptr<poly_eval> p;

   Eigen::VectorXd weights;

public:
   rbf_symmetric_evaluator(const rbf::rbf_base& rbf, int poly_degree,
      const std::vector<Eigen::Vector3d>& points)
      : rbf(rbf)
      , poly_degree(poly_degree)
      , n_points(points.size())
      , n_polynomials(polynomial::basis_base::dimension(poly_degree))
   {
      auto bounds = geometry::bbox3d::from_points(points);

      Eigen::Vector3d bounds_size = bounds.max - bounds.min;
      bounds.min -= (1.0 + 1.0 / 64.0) * bounds_size;
      bounds.max += (1.0 + 1.0 / 64.0) * bounds_size;

      a = std::make_unique<fmm::fmm_operator<Order>>(rbf, fmm::tree_height(points.size()), bounds);
      a->set_points(points);

      if (poly_degree >= 0) {
         p = std::make_unique<poly_eval>(poly_degree);
         p->set_field_points(points);
      }
   }

   Eigen::VectorXd evaluate() const
   {
      auto rbf_at_center = rbf.evaluate(0.0);
      Eigen::VectorXd y = weights.head(n_points) * rbf_at_center;

      y += a->evaluate();

      if (poly_degree >= 0) {
         // Add polynomial terms.
         y += p->evaluate();
      }

      return y;
   }

   template<typename Derived>
   void set_weights(const Eigen::MatrixBase<Derived>& weights)
   {
      assert(weights.size() == n_points + n_polynomials);

      this->weights = weights;

      a->set_weights(weights.head(n_points));

      if (poly_degree >= 0) {
         p->set_weights(weights.tail(n_polynomials));
      }
   }
};

} // namespace interpolation
} // namespace polatory
