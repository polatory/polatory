#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/common/types.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/precision.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::precision;
using polatory::common::valuesd;
using polatory::geometry::bboxNd;
using polatory::geometry::pointNd;
using polatory::geometry::pointsNd;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_evaluator;
using polatory::numeric::relative_error;
using polatory::rbf::triharmonic3d;

TEST(rbf_evaluator, trivial) {
  constexpr int kDim = 3;
  using Bbox = bboxNd<kDim>;
  using Point = pointNd<kDim>;
  using Points = pointsNd<kDim>;

  index_t n_points = 1024;
  index_t n_grad_points = 1024;
  index_t n_eval_points = 1024;
  index_t n_grad_eval_points = 1024;
  auto relative_tolerance = 5e-7;

  triharmonic3d<kDim> rbf({1.0});
  rbf.set_anisotropy(random_anisotropy<kDim>());

  auto poly_degree = rbf.cpd_order() - 1;
  model model(rbf, poly_degree);
  model.set_nugget(0.01);

  Points points = Points::Random(n_points, kDim);
  Points grad_points = Points::Random(n_grad_points, kDim);
  Points eval_points = Points::Random(n_eval_points, kDim);
  Points grad_eval_points = Points::Random(n_grad_eval_points, kDim);

  valuesd weights = valuesd::Random(n_points + kDim * n_grad_points + model.poly_basis_size());

  Bbox bbox{-Point::Ones(), Point::Ones()};
  rbf_evaluator eval(model, bbox, precision::kPrecise);
  eval.set_source_points(points, grad_points);
  eval.set_weights(weights);
  eval.set_target_points(eval_points, grad_eval_points);

  rbf_direct_evaluator direct_eval(model, points, grad_points);
  direct_eval.set_weights(weights);
  direct_eval.set_target_points(eval_points, grad_eval_points);

  auto values = eval.evaluate();
  auto direct_values = direct_eval.evaluate();

  EXPECT_EQ(n_eval_points + kDim * n_grad_eval_points, values.rows());
  EXPECT_EQ(n_eval_points + kDim * n_grad_eval_points, direct_values.rows());

  EXPECT_LT(relative_error(values, direct_values), relative_tolerance);
}
