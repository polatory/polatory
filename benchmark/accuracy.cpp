// This is an example program that estimates the relative accuracy of
// the fast evaluation of an interpolant.

#include <Eigen/Core>
#include <exception>
#include <iostream>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/polatory.hpp>
#include <polatory/rbf/reference/cov_gaussian.hpp>
#include <polatory/rbf/reference/triharmonic3d.hpp>

using polatory::index_t;
using polatory::model;
using polatory::precision;
using polatory::common::orthonormalize_cols;
using polatory::common::valuesd;
using polatory::geometry::pointsNd;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_evaluator;
using polatory::polynomial::monomial_basis;
using polatory::rbf::biharmonic2d;
using polatory::rbf::biharmonic3d;
using polatory::rbf::cov_exponential;
using polatory::rbf::cov_spheroidal3;
using polatory::rbf::cov_spheroidal5;
using polatory::rbf::cov_spheroidal7;
using polatory::rbf::cov_spheroidal9;
using polatory::rbf::multiquadric1;
using polatory::rbf::rbf_base;
using polatory::rbf::reference::cov_gaussian;
using polatory::rbf::reference::triharmonic3d;

template <class Rbf>
double estimate_accuracy(const std::vector<double>& rbf_params) {
  constexpr int kDim = Rbf::kDim;
  using Model = model<Rbf>;
  using Points = pointsNd<kDim>;

  auto n_points = 32768;
  auto n_eval_points = 1024;
  Points points = Points::Random(n_points, kDim);
  Points eval_points = points.topRows(n_eval_points);

  Rbf rbf(rbf_params);

  auto poly_degree = rbf.cpd_order() - 1;
  Model model(rbf, poly_degree);

  valuesd weights = valuesd::Zero(n_points + model.poly_basis_size());
  weights.head(n_points) = valuesd::Random(n_points);

  if (poly_degree >= 0) {
    monomial_basis<3> poly(model.poly_degree());
    Eigen::MatrixXd p = poly.evaluate(points).transpose();
    orthonormalize_cols(p);

    // Orthogonalize weights against P.
    auto n_cols = p.cols();
    for (index_t i = 0; i < n_cols; i++) {
      auto dot = p.col(i).dot(weights.head(n_points));
      weights.head(n_points) -= dot * p.col(i);
    }
  }

  rbf_direct_evaluator<Model> direct_eval(model, points, Points(0, kDim));
  direct_eval.set_weights(weights);
  direct_eval.set_target_points(eval_points, Points(0, kDim));

  rbf_evaluator<Model> eval(model, points, precision::kPrecise);
  eval.set_weights(weights);
  eval.set_target_points(eval_points);

  auto direct_values = direct_eval.evaluate();
  auto values = eval.evaluate();

  auto max_residual = (values - direct_values).template lpNorm<Eigen::Infinity>();
  auto scale = direct_values.template lpNorm<Eigen::Infinity>();

  return max_residual / scale;
}

int main() {
  try {
    std::cout << "cov_gaussian[scale=0.001]: " << estimate_accuracy<cov_gaussian<3>>({1.0, 0.01})
              << std::endl;
    std::cout << "cov_gaussian[scale=0.01]: " << estimate_accuracy<cov_gaussian<3>>({1.0, 0.01})
              << std::endl;
    std::cout << "cov_gaussian[scale=0.1]: " << estimate_accuracy<cov_gaussian<3>>({1.0, 0.1})
              << std::endl;
    std::cout << "cov_gaussian[scale=1.]: " << estimate_accuracy<cov_gaussian<3>>({1.0, 1.0})
              << std::endl;
    std::cout << "cov_gaussian[scale=10.]: " << estimate_accuracy<cov_gaussian<3>>({1.0, 10.0})
              << std::endl;
    std::cout << "triharmonic3d: " << estimate_accuracy<triharmonic3d<3>>({1.0}) << std::endl;
    std::cout << "biharmonic2d: " << estimate_accuracy<biharmonic2d<3>>({1.0}) << std::endl;
    std::cout << "biharmonic3d: " << estimate_accuracy<biharmonic3d<3>>({1.0}) << std::endl;
    std::cout << "multiquadric1[scale=0.01]: " << estimate_accuracy<multiquadric1<3>>({1.0, 0.01})
              << std::endl;
    std::cout << "multiquadric1[scale=0.1]: " << estimate_accuracy<multiquadric1<3>>({1.0, 0.1})
              << std::endl;
    std::cout << "multiquadric1[scale=1.]: " << estimate_accuracy<multiquadric1<3>>({1.0, 1.0})
              << std::endl;
    std::cout << "multiquadric1[scale=10.]: " << estimate_accuracy<multiquadric1<3>>({1.0, 10.0})
              << std::endl;
    std::cout << "cov_exponential[scale=0.01]: "
              << estimate_accuracy<cov_exponential<3>>({1.0, 0.01}) << std::endl;
    std::cout << "cov_exponential[scale=0.1]: " << estimate_accuracy<cov_exponential<3>>({1.0, 0.1})
              << std::endl;
    std::cout << "cov_exponential[scale=1.]: " << estimate_accuracy<cov_exponential<3>>({1.0, 1.0})
              << std::endl;
    std::cout << "cov_exponential[scale=10.]: "
              << estimate_accuracy<cov_exponential<3>>({1.0, 10.0}) << std::endl;
    std::cout << "cov_spheroidal3[scale=0.01]: "
              << estimate_accuracy<cov_spheroidal3<3>>({1.0, 0.01}) << std::endl;
    std::cout << "cov_spheroidal3[scale=0.1]: " << estimate_accuracy<cov_spheroidal3<3>>({1.0, 0.1})
              << std::endl;
    std::cout << "cov_spheroidal3[scale=1.]: " << estimate_accuracy<cov_spheroidal3<3>>({1.0, 1.0})
              << std::endl;
    std::cout << "cov_spheroidal3[scale=10.]: "
              << estimate_accuracy<cov_spheroidal3<3>>({1.0, 10.0}) << std::endl;
    std::cout << "cov_spheroidal5[scale=0.01]: "
              << estimate_accuracy<cov_spheroidal5<3>>({1.0, 0.01}) << std::endl;
    std::cout << "cov_spheroidal5[scale=0.1]: " << estimate_accuracy<cov_spheroidal5<3>>({1.0, 0.1})
              << std::endl;
    std::cout << "cov_spheroidal5[scale=1.]: " << estimate_accuracy<cov_spheroidal5<3>>({1.0, 1.0})
              << std::endl;
    std::cout << "cov_spheroidal5[scale=10.]: "
              << estimate_accuracy<cov_spheroidal5<3>>({1.0, 10.0}) << std::endl;
    std::cout << "cov_spheroidal7[scale=0.01]: "
              << estimate_accuracy<cov_spheroidal7<3>>({1.0, 0.01}) << std::endl;
    std::cout << "cov_spheroidal7[scale=0.1]: " << estimate_accuracy<cov_spheroidal7<3>>({1.0, 0.1})
              << std::endl;
    std::cout << "cov_spheroidal7[scale=1.]: " << estimate_accuracy<cov_spheroidal7<3>>({1.0, 1.0})
              << std::endl;
    std::cout << "cov_spheroidal7[scale=10.]: "
              << estimate_accuracy<cov_spheroidal7<3>>({1.0, 10.0}) << std::endl;
    std::cout << "cov_spheroidal9[scale=0.01]: "
              << estimate_accuracy<cov_spheroidal9<3>>({1.0, 0.01}) << std::endl;
    std::cout << "cov_spheroidal9[scale=0.1]: " << estimate_accuracy<cov_spheroidal9<3>>({1.0, 0.1})
              << std::endl;
    std::cout << "cov_spheroidal9[scale=1.]: " << estimate_accuracy<cov_spheroidal9<3>>({1.0, 1.0})
              << std::endl;
    std::cout << "cov_spheroidal9[scale=10.]: "
              << estimate_accuracy<cov_spheroidal9<3>>({1.0, 10.0}) << std::endl;

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
