// This is an example program that estimates the relative accuracy of
// the fast evaluation method of RBF interpolants.

#include <Eigen/Core>
#include <exception>
#include <format>
#include <iostream>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>

#include "../common/make_model.hpp"
#include "parse_options.hpp"

using polatory::index_t;
using polatory::matrixd;
using polatory::model;
using polatory::read_table;
using polatory::vectord;
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

  index_t mu{};
  Points points;
  if (!opts.in_file.empty()) {
    if (opts.n_points != 0 || opts.n_grad_points != 0) {
      throw std::runtime_error("--n or --grad-n cannot be specified with --in");
    }

    matrixd table = read_table(opts.in_file);
    mu = table.rows();
    points = table.leftCols(kDim);
  } else {
    mu = opts.n_points;
    points = Points::Random(mu, kDim);
  }

  index_t sigma{};
  Points grad_points;
  if (!opts.grad_in_file.empty()) {
    if (opts.n_points != 0 || opts.n_grad_points != 0) {
      throw std::runtime_error("--n or --grad-n cannot be specified with --grad-in");
    }

    matrixd table = read_table(opts.grad_in_file);
    sigma = table.rows();
    grad_points = table.leftCols(kDim);
  } else {
    sigma = opts.n_grad_points;
    grad_points = Points::Random(sigma, kDim);
  }

  Bbox bbox{-Point::Ones(), Point::Ones()};
  Points eval_points = Points::Random(opts.n_eval_points, kDim);
  Points grad_eval_points = Points::Random(opts.n_grad_eval_points, kDim);
  if (!opts.in_file.empty() || !opts.grad_in_file.empty()) {
    bbox = Bbox::from_points(points).convex_hull(Bbox::from_points(grad_points));
    for (auto p : eval_points.rowwise()) {
      p = (bbox.min() + bbox.width().cwiseProduct(p)).eval();
    }
    for (auto p : grad_eval_points.rowwise()) {
      p = (bbox.min() + bbox.width().cwiseProduct(p)).eval();
    }
  }

  auto m = mu + kDim * sigma;
  auto l = model.poly_basis_size();

  vectord weights = vectord::Zero(m + l);
  weights.head(m) = vectord::Random(m);

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
        throw std::runtime_error(std::format("unsupported dimension: {}", opts.dim));
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "unknown error" << std::endl;
    return 1;
  }
}
