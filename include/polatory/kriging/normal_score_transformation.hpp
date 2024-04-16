#pragma once

#include <algorithm>
#include <boost/math/special_functions/erf.hpp>
#include <cmath>
#include <numbers>
#include <numeric>
#include <polatory/types.hpp>

namespace polatory::kriging {

class normal_score_transformation {
 public:
  normal_score_transformation(int order = 30) : order_(order) {
    if (order < 0) {
      throw std::invalid_argument("order must be non-negative");
    }
  }

  vectord transform(const vectord& z) {
    auto n = z.size();
    std::vector<index_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](auto i, auto j) { return z(i) < z(j); });

    vectord y(n);
    auto inv_den = 1.0 / (2.0 * static_cast<double>(n));
    for (index_t i = 0; i < n; ++i) {
      auto p = (2.0 * static_cast<double>(i) + 1.0) * inv_den;
      y(indices[i]) = inv_normal_cdf(p);
    }

    compute_phi(z(indices), y(indices));

    transformed_ = true;
    return y;
  }

  vectord back_transform(const vectord& y) const {
    throw_if_not_transformed();

    auto n = y.size();

    vectord h0y = vectord::Ones(n);
    vectord h1y = -y;

    vectord z = phi_(0) * h0y;

    for (auto p = 1; p <= order_; ++p) {
      auto pd = static_cast<double>(p);

      z += phi_(p) * h1y;

      vectord h2y = -1.0 / std::sqrt(pd + 1.0) * (y.array() * h1y.array()) -
                    std::sqrt(pd / (pd + 1.0)) * h0y.array();
      h0y = h1y;
      h1y = h2y;
    }

    return z;
  }

  double back_transform_gamma(double gamma_y) const {
    throw_if_not_transformed();

    auto gamma_z = 0.0;

    auto one_minus_gamma_p = 1.0 - gamma_y;
    for (auto p = 1; p <= order_; ++p) {
      auto phi = phi_(p);
      gamma_z += phi * phi * (1.0 - one_minus_gamma_p);

      one_minus_gamma_p *= 1.0 - gamma_y;
    }

    return gamma_z;
  }

 private:
  // Precondition: z and y must be sorted.
  void compute_phi(const vectord& z, const vectord& y) {
    auto n = z.size();
    vectord gy = 1.0 / std::sqrt(2.0 * std::numbers::pi) * (-0.5 * y.array().square()).exp();

    phi_ = vectord::Zero(order_ + 1);
    phi_(0) = z.mean();

    vectord a = vectord::Zero(n);
    for (index_t i = 0; i < n; ++i) {
      if (i == 0) {
        a(i) = z(i) - z(i + 1);
      } else if (i == n - 1) {
        a(i) = z(i - 1) - z(i);
      } else {
        a(i) = 0.5 * (z(i - 1) - z(i + 1));
      }
    }

    vectord h0y = vectord::Ones(n);
    vectord h1y = -y;

    for (auto p = 1; p <= order_; ++p) {
      auto pd = static_cast<double>(p);
      phi_(p) = 1.0 / std::sqrt(pd) * (a.array() * h0y.array() * gy.array()).sum();

      vectord h2y = -1.0 / std::sqrt(pd + 1.0) * (y.array() * h1y.array()) -
                    std::sqrt(pd / (pd + 1.0)) * h0y.array();
      h0y = h1y;
      h1y = h2y;
    }
  }

  static double inv_normal_cdf(double p) {
    return -std::numbers::sqrt2 * boost::math::erfc_inv(2.0 * p);
  }

  void throw_if_not_transformed() const {
    if (!transformed_) {
      throw std::runtime_error("normal score transformation has not been computed");
    }
  }

  int order_;
  bool transformed_{};
  vectord phi_;  // Coefficients for the Hermite polynomials.
};

}  // namespace polatory::kriging
