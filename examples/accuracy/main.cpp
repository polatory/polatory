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
using polatory::interpolant;
using polatory::model;
using polatory::geometry::bboxNd;
using polatory::geometry::pointsNd;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_evaluator;

template <int Dim>
void main_impl(const options& opts) {
  static constexpr int kDim = Dim;
  using DirectEvaluator = rbf_direct_evaluator<kDim>;
  using Evaluator = rbf_evaluator<kDim>;
  using Interpolant = interpolant<kDim>;
  using Points = pointsNd<kDim>;

  auto inter = Interpolant::load(opts.in_file);
  const auto& bbox = inter.bbox();
  const auto& points = inter.centers();
  const auto& grad_points = inter.grad_centers();
  const auto& model = inter.model();
  const auto& weights = inter.weights();

  Points eval_points = Points::Random(opts.n_eval_points, kDim);
  Points grad_eval_points = Points::Random(opts.n_grad_eval_points, kDim);

  for (auto p : eval_points.rowwise()) {
    p = (bbox.min().array() + bbox.width().array() * (p.array() + 1.0) / 2.0).eval();
  }
  for (auto p : grad_eval_points.rowwise()) {
    p = (bbox.min().array() + bbox.width().array() * (p.array() + 1.0) / 2.0).eval();
  }

  Evaluator eval(model, bbox, opts.order);
  eval.set_source_points(points, grad_points);
  eval.set_weights(weights);
  eval.set_target_points(eval_points, grad_eval_points);

  auto test_size = std::min(eval_points.rows(), index_t{1024});
  auto grad_test_size = std::min(grad_eval_points.rows(), index_t{1024});

  DirectEvaluator direct_eval(model, points, grad_points);
  direct_eval.set_weights(weights);
  direct_eval.set_target_points(eval_points.topRows(test_size),
                                grad_eval_points.topRows(grad_test_size));

  auto values = eval.evaluate();
  auto direct_values = direct_eval.evaluate();

  std::cout << "Absolute error: " << (values.head(test_size) - direct_values).cwiseAbs().maxCoeff()
            << std::endl;
}

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);
    switch (opts.dim) {
      case 1:
        main_impl<1>(opts);
        break;
      case 2:
        main_impl<2>(opts);
        break;
      case 3:
        main_impl<3>(opts);
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
