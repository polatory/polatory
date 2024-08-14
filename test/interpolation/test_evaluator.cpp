#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/direct_evaluator.hpp>
#include <polatory/interpolation/evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>
#include <utility>

#include "../utility.hpp"

using polatory::Index;
using polatory::Model;
using polatory::VecX;
using polatory::geometry::Bbox;
using polatory::geometry::Point;
using polatory::geometry::Points;
using polatory::interpolation::DirectEvaluator;
using polatory::interpolation::Evaluator;
using polatory::numeric::absolute_error;
using polatory::rbf::Triharmonic3D;

TEST(rbf_evaluator, trivial) {
  constexpr int kDim = 3;
  using Bbox = Bbox<kDim>;
  using Point = Point<kDim>;
  using Points = Points<kDim>;

  Index n_points = 1024;
  Index n_grad_points = 1024;
  Index n_eval_points = 1024;
  Index n_grad_eval_points = 1024;
  auto accuracy = 1e-4;
  auto grad_accuracy = 1e-4;

  Triharmonic3D<kDim> rbf({1.0});
  rbf.set_anisotropy(random_anisotropy<kDim>());

  auto poly_degree = rbf.cpd_order() - 1;
  Model<kDim> model(std::move(rbf), poly_degree);
  model.set_nugget(0.01);

  Points points = Points::Random(n_points, kDim);
  Points grad_points = Points::Random(n_grad_points, kDim);
  Points eval_points = Points::Random(n_eval_points, kDim);
  Points grad_eval_points = Points::Random(n_grad_eval_points, kDim);

  VecX weights = VecX::Random(n_points + kDim * n_grad_points + model.poly_basis_size());

  Bbox bbox{-Point::Ones(), Point::Ones()};
  Evaluator<kDim> eval(model, bbox, accuracy, grad_accuracy);
  eval.set_source_points(points, grad_points);
  eval.set_weights(weights);
  eval.set_target_points(eval_points, grad_eval_points);

  DirectEvaluator<kDim> direct_eval(model, points, grad_points);
  direct_eval.set_weights(weights);
  direct_eval.set_target_points(eval_points, grad_eval_points);

  auto values = eval.evaluate();
  auto direct_values = direct_eval.evaluate();

  EXPECT_EQ(n_eval_points + kDim * n_grad_eval_points, values.rows());
  EXPECT_EQ(n_eval_points + kDim * n_grad_eval_points, direct_values.rows());

  EXPECT_LT(absolute_error<Eigen::Infinity>(values.head(n_eval_points),
                                            direct_values.head(n_eval_points)),
            accuracy);
  EXPECT_LT(absolute_error<Eigen::Infinity>(values.tail(kDim * n_grad_eval_points),
                                            direct_values.tail(kDim * n_grad_eval_points)),
            grad_accuracy);
}
