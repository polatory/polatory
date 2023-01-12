// This is an example program that estimates the relative accuracy of
// the fast evaluation of an interpolant.

#include <exception>
#include <iostream>

#include <Eigen/Core>

#include <polatory/geometry/sphere3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/polynomial/orthonormal_basis.hpp>
#include <polatory/rbf/biharmonic2d.hpp>
#include <polatory/rbf/biharmonic3d.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/cov_spheroidal3.hpp>
#include <polatory/rbf/cov_spheroidal5.hpp>
#include <polatory/rbf/cov_spheroidal7.hpp>
#include <polatory/rbf/cov_spheroidal9.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/types.hpp>

using polatory::index_t;
using polatory::common::valuesd;
using polatory::geometry::sphere3d;
using polatory::geometry::points3d;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_evaluator;
using polatory::model;
using polatory::point_cloud::random_points;
using polatory::polynomial::orthonormal_basis;
using polatory::rbf::biharmonic2d;
using polatory::rbf::biharmonic3d;
using polatory::rbf::cov_exponential;
using polatory::rbf::cov_spheroidal3;
using polatory::rbf::cov_spheroidal5;
using polatory::rbf::cov_spheroidal7;
using polatory::rbf::cov_spheroidal9;
using polatory::rbf::rbf_base;

double estimate_accuracy(const rbf_base& rbf) {
  auto n_points = 32768;
  auto n_eval_points = 1024;
  auto points = random_points(sphere3d(), n_points);
  points3d eval_points = points.topRows(n_eval_points);

  auto poly_degree = rbf.cpd_order() - 1;
  model model(rbf, 3, poly_degree);

  valuesd weights = valuesd::Zero(n_points + model.poly_basis_size());
  weights.head(n_points) = valuesd::Random(n_points);

  if (poly_degree >= 0) {
    orthonormal_basis poly(model.poly_dimension(), model.poly_degree(), points);
    Eigen::MatrixXd p = poly.evaluate(points).transpose();

    // Orthogonalize weights against P.
    auto n_cols = static_cast<index_t>(p.cols());
    for (index_t i = 0; i < n_cols; i++) {
      auto dot = p.col(i).dot(weights.head(n_points));
      weights.head(n_points) -= dot * p.col(i);
    }
  }

  rbf_direct_evaluator direct_eval(model, points);
  direct_eval.set_weights(weights);
  direct_eval.set_field_points(eval_points);

  rbf_evaluator<> eval(model, points);
  eval.set_weights(weights);
  eval.set_field_points(eval_points);

  auto direct_values = direct_eval.evaluate();
  auto values = eval.evaluate();

  auto max_residual = (values - direct_values).lpNorm<Eigen::Infinity>();
  auto scale = direct_values.lpNorm<Eigen::Infinity>();

  return max_residual / scale;
}

int main() {
  try {
    std::cout << "biharmonic2d: " << estimate_accuracy(biharmonic2d({ 1.0 })) << std::endl;
    std::cout << "biharmonic3d: " << estimate_accuracy(biharmonic3d({ 1.0 })) << std::endl;
    std::cout << "cov_exponential[scale=0.01]: " << estimate_accuracy(cov_exponential({ 1.0, 0.01 })) << std::endl;
    std::cout << "cov_exponential[scale=0.1]: " << estimate_accuracy(cov_exponential({ 1.0, 0.1 })) << std::endl;
    std::cout << "cov_exponential[scale=1.]: " << estimate_accuracy(cov_exponential({ 1.0, 1.0 })) << std::endl;
    std::cout << "cov_exponential[scale=10.]: " << estimate_accuracy(cov_exponential({ 1.0, 10.0 })) << std::endl;
    std::cout << "cov_spheroidal3[scale=0.01]: " << estimate_accuracy(cov_spheroidal3({ 1.0, 0.01 })) << std::endl;
    std::cout << "cov_spheroidal3[scale=0.1]: " << estimate_accuracy(cov_spheroidal3({ 1.0, 0.1 })) << std::endl;
    std::cout << "cov_spheroidal3[scale=1.]: " << estimate_accuracy(cov_spheroidal3({ 1.0, 1.0 })) << std::endl;
    std::cout << "cov_spheroidal3[scale=10.]: " << estimate_accuracy(cov_spheroidal3({ 1.0, 10.0 })) << std::endl;
    std::cout << "cov_spheroidal5[scale=0.01]: " << estimate_accuracy(cov_spheroidal5({ 1.0, 0.01 })) << std::endl;
    std::cout << "cov_spheroidal5[scale=0.1]: " << estimate_accuracy(cov_spheroidal5({ 1.0, 0.1 })) << std::endl;
    std::cout << "cov_spheroidal5[scale=1.]: " << estimate_accuracy(cov_spheroidal5({ 1.0, 1.0 })) << std::endl;
    std::cout << "cov_spheroidal5[scale=10.]: " << estimate_accuracy(cov_spheroidal5({ 1.0, 10.0 })) << std::endl;
    std::cout << "cov_spheroidal7[scale=0.01]: " << estimate_accuracy(cov_spheroidal7({ 1.0, 0.01 })) << std::endl;
    std::cout << "cov_spheroidal7[scale=0.1]: " << estimate_accuracy(cov_spheroidal7({ 1.0, 0.1 })) << std::endl;
    std::cout << "cov_spheroidal7[scale=1.]: " << estimate_accuracy(cov_spheroidal7({ 1.0, 1.0 })) << std::endl;
    std::cout << "cov_spheroidal7[scale=10.]: " << estimate_accuracy(cov_spheroidal7({ 1.0, 10.0 })) << std::endl;
    std::cout << "cov_spheroidal9[scale=0.01]: " << estimate_accuracy(cov_spheroidal9({ 1.0, 0.01 })) << std::endl;
    std::cout << "cov_spheroidal9[scale=0.1]: " << estimate_accuracy(cov_spheroidal9({ 1.0, 0.1 })) << std::endl;
    std::cout << "cov_spheroidal9[scale=1.]: " << estimate_accuracy(cov_spheroidal9({ 1.0, 1.0 })) << std::endl;
    std::cout << "cov_spheroidal9[scale=10.]: " << estimate_accuracy(cov_spheroidal9({ 1.0, 10.0 })) << std::endl;

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
