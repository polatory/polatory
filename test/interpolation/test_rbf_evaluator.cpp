#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/geometry/sphere3d.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/rbf/multiquadric1.hpp>
#include <polatory/rbf/reference/triharmonic3d.hpp>
#include <polatory/types.hpp>

#include "../random_anisotropy.hpp"
#include "utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::precision;
using polatory::common::valuesd;
using polatory::geometry::bbox3d;
using polatory::geometry::point3d;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_evaluator;
using polatory::point_cloud::random_points;
using polatory::rbf::multiquadric1;
using polatory::rbf::reference::triharmonic3d;

namespace {

void test_poly_degree(int poly_degree, index_t n_initial_points, index_t n_initial_grad_points,
                      index_t n_initial_eval_points, index_t n_initial_eval_grad_points) {
  using Rbf = multiquadric1;
  using Model = model<Rbf>;

  const int dim = 3;
  index_t n_points = n_initial_points;
  index_t n_grad_points = n_initial_grad_points;
  index_t n_eval_points = n_initial_eval_points;
  index_t n_eval_grad_points = n_initial_eval_grad_points;

  auto rel_tolerance = 2e-8;
  auto grad_rel_tolerance = 1e-7;

  Rbf rbf({1.0, 0.001});
  rbf.set_anisotropy(random_anisotropy());

  Model model(rbf, dim, poly_degree);
  model.set_nugget(0.01);

  bbox3d bbox{-point3d::Ones(), point3d::Ones()};
  rbf_evaluator<Model> eval(model, bbox, precision::kPrecise);

  for (auto i = 0; i < 4; i++) {
    auto points = random_points(sphere3d(), n_points);
    auto grad_points = random_points(sphere3d(), n_grad_points);
    auto eval_points = random_points(sphere3d(), n_eval_points);
    auto eval_grad_points = random_points(sphere3d(), n_eval_grad_points);

    valuesd weights = valuesd::Random(n_points + dim * n_grad_points + model.poly_basis_size());

    rbf_direct_evaluator<Model> direct_eval(model, points, grad_points);
    direct_eval.set_weights(weights);
    direct_eval.set_field_points(eval_points, eval_grad_points);

    eval.set_source_points(points, grad_points);
    eval.set_weights(weights);
    eval.set_field_points(eval_points, eval_grad_points);

    auto direct_values = direct_eval.evaluate();
    auto values = eval.evaluate();

    EXPECT_EQ(n_eval_points + dim * n_eval_grad_points, direct_values.rows());
    EXPECT_EQ(n_eval_points + dim * n_eval_grad_points, values.rows());

    if (n_eval_points > 0) {
      EXPECT_LT(relative_error(values.head(n_eval_points), direct_values.head(n_eval_points)),
                rel_tolerance);
    }

    if (n_eval_grad_points > 0) {
      EXPECT_LT(relative_error(values.tail(dim * n_eval_grad_points),
                               direct_values.tail(dim * n_eval_grad_points)),
                grad_rel_tolerance);
    }

    n_points *= 2;
    n_grad_points *= 2;
    n_eval_points *= 2;
    n_eval_grad_points *= 2;
  }
}

}  // namespace

TEST(rbf_evaluator, trivial) {
  test_poly_degree(1, 1024, 0, 1024, 0);
  test_poly_degree(1, 0, 256, 0, 256);

  test_poly_degree(1, 1024, 256, 1024, 256);
  test_poly_degree(2, 1024, 256, 1024, 256);
}
