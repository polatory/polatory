// This is an example program that estimates the relative accuracy of
// the fast evaluation method of RBF interpolants.

#include <Eigen/Core>
#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <utility>

#include "../common/common.hpp"
#include "parse_options.hpp"

using polatory::index_t;
using polatory::model;
using polatory::common::valuesd;
using polatory::geometry::bboxNd;
using polatory::geometry::pointNd;
using polatory::geometry::pointsNd;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_evaluator;
using polatory::numeric::relative_error;

template <int Dim>
void main_impl(model<Dim>&& model, const options& opts) {
  static constexpr int kDim = Dim;
  using Bbox = bboxNd<kDim>;
  using DirectEvaluator = rbf_direct_evaluator<kDim>;
  using Evaluator = rbf_evaluator<kDim>;
  using Point = pointNd<kDim>;
  using Points = pointsNd<kDim>;

  auto mu = opts.n_points;
  auto sigma = opts.n_grad_points;
  auto m = mu + kDim * sigma;
  auto l = model.poly_basis_size();

  Points points = Points::Random(mu, kDim);
  Points grad_points = Points::Random(sigma, kDim);
  Points eval_points = Points::Random(opts.n_eval_points, kDim);
  Points grad_eval_points = Points::Random(opts.n_grad_eval_points, kDim);

  valuesd weights = valuesd::Zero(m + l);
  weights.head(m) = valuesd::Random(m);

  Bbox bbox{-Point::Ones(), Point::Ones()};
  Evaluator eval(model, bbox, opts.order);
  eval.set_source_points(points, grad_points);
  eval.set_weights(weights);
  eval.set_target_points(eval_points, grad_eval_points);

  if (opts.perf) {
    eval.evaluate();
    return;
  }

  DirectEvaluator direct_eval(model, points, grad_points);
  direct_eval.set_weights(weights);
  direct_eval.set_target_points(eval_points, grad_eval_points);

  auto values = eval.evaluate();
  auto direct_values = direct_eval.evaluate();

  std::cout << "Relative error (L1): " << relative_error<1>(values, direct_values) << std::endl;
  std::cout << "Relative error (L2): " << relative_error<2>(values, direct_values) << std::endl;
  std::cout << "Relative error (L-infinity): "
            << relative_error<Eigen::Infinity>(values, direct_values) << std::endl;
}

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);
    switch (opts.dim) {
      case 1:
        main_impl(make_model<1>(opts.model_opts), opts);
        break;
      case 2:
        main_impl(make_model<2>(opts.model_opts), opts);
        break;
      case 3:
        main_impl(make_model<3>(opts.model_opts), opts);
        break;
      default:
        throw std::runtime_error("Unsupported dimension: " + std::to_string(opts.dim));
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
