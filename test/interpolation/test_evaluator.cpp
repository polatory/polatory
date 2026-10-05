#include <gtest/gtest.h>

#include <Eigen/Core>
#include <jizai/geometry/bbox3d.hpp>
#include <jizai/geometry/point3d.hpp>
#include <jizai/interpolation/direct_evaluator.hpp>
#include <jizai/interpolation/evaluator.hpp>
#include <jizai/model.hpp>
#include <jizai/numeric/error.hpp>
#include <jizai/rbf/polyharmonic_odd.hpp>
#include <jizai/types.hpp>
#include <utility>

#include "../utility.hpp"

using jizai::Index;
using jizai::Model;
using jizai::VecX;
using jizai::geometry::Bbox;
using jizai::geometry::Point;
using jizai::geometry::Points;
using jizai::interpolation::DirectEvaluator;
using jizai::interpolation::Evaluator;
using jizai::numeric::absolute_error;
using jizai::rbf::Triharmonic3D;

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

  Triharmonic3D<kDim> rbf;
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
