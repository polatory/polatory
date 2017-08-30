// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <vector>

#include "../numeric/sum_accumulator.hpp"
#include "../polynomial.hpp"
#include "../rbf/rbf_base.hpp"

namespace polatory {
namespace interpolation {

class rbf_direct_symmetric_evaluator {
   typedef polynomial::polynomial_evaluator<polynomial::monomial_basis<>> poly_eval;

   const rbf::rbf_base& rbf;
   const int poly_degree;
   const size_t n_points;
   const size_t n_polynomials;

   std::unique_ptr<poly_eval> p;

   const std::vector<Eigen::Vector3d> points;
   Eigen::VectorXd weights;

public:
   rbf_direct_symmetric_evaluator(const rbf::rbf_base& rbf, int poly_degree,
      const std::vector<Eigen::Vector3d>& points)
      : rbf(rbf)
      , poly_degree(poly_degree)
      , n_points(points.size())
      , n_polynomials(polynomial::basis_base::dimension(poly_degree))
      , points(points)
   {
      if (poly_degree >= 0) {
         p = std::make_unique<poly_eval>(poly_degree);
         p->set_field_points(points);
      }
   }

   Eigen::VectorXd evaluate() const
   {
      auto y_accum = std::vector<numeric::kahan_sum_accumulator<double>>(n_points);

      auto rbf_at_center = rbf.evaluate(0.0);
      for (size_t i = 0; i < n_points; i++) {
         y_accum[i] += weights(i) * rbf_at_center;
      }
      for (size_t i = 0; i < n_points - 1; i++) {
         for (size_t j = i + 1; j < n_points; j++) {
            auto a_ij = rbf.evaluate(points[i], points[j]);
            y_accum[i] += weights(j) * a_ij;
            y_accum[j] += weights(i) * a_ij;
         }
      }

      if (poly_degree >= 0) {
         // Add polynomial terms.
         auto poly_val = p->evaluate();
         for (size_t i = 0; i < n_points; i++) {
            y_accum[i] += poly_val(i);
         }
      }

      Eigen::VectorXd y(n_points);
      for (size_t i = 0; i < n_points; i++) {
         y(i) = y_accum[i].get();
      }

      return y;
   }

   template<typename Derived>
   void set_weights(const Eigen::MatrixBase<Derived>& weights)
   {
      assert(weights.size() == n_points + n_polynomials);

      this->weights = weights;

      if (poly_degree >= 0) {
         p->set_weights(weights.tail(n_polynomials));
      }
   }
};

} // namespace interpolation
} // namespace polatory
