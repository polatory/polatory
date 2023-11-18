#include <gtest/gtest.h>

#include <Eigen/Core>
#include <format>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/rbf/inverse_multiquadric.hpp>
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
using polatory::rbf::inverse_multiquadric1;

namespace {

template <int Dim>
void test(int poly_degree, index_t n_initial_points, index_t n_initial_grad_points,
          index_t n_initial_eval_points, index_t n_initial_eval_grad_points) {
  std::cout << std::format("dim = {}, deg = {}", Dim, poly_degree) << std::endl;

  using Rbf = inverse_multiquadric1<Dim>;
  using Bbox = bboxNd<Dim>;
  using Point = pointNd<Dim>;
  using Points = pointsNd<Dim>;

  index_t n_points = n_initial_points;
  index_t n_grad_points = n_initial_grad_points;
  index_t n_eval_points = n_initial_eval_points;
  index_t n_eval_grad_points = n_initial_eval_grad_points;

  auto rel_tolerance = 1e-10;

  Rbf rbf({1.0, 0.01});
  rbf.set_anisotropy(random_anisotropy<Dim>());

  model model(rbf, poly_degree);
  model.set_nugget(0.01);

  Bbox bbox{-Point::Ones(), Point::Ones()};
  rbf_evaluator eval(model, bbox, precision::kPrecise);

  for (auto i = 0; i < 2; i++) {
    Points points = Points::Random(n_points, Dim);
    Points grad_points = Points::Random(n_grad_points, Dim);
    Points eval_points = Points::Random(n_eval_points, Dim);
    Points eval_grad_points = Points::Random(n_eval_grad_points, Dim);

    valuesd weights = valuesd::Random(n_points + Dim * n_grad_points + model.poly_basis_size());

    rbf_direct_evaluator direct_eval(model, points, grad_points);
    direct_eval.set_weights(weights);
    direct_eval.set_target_points(eval_points, eval_grad_points);

    eval.set_source_points(points, grad_points);
    eval.set_weights(weights);
    eval.set_target_points(eval_points, eval_grad_points);

    auto direct_values = direct_eval.evaluate();
    auto values = eval.evaluate();

    EXPECT_EQ(n_eval_points + Dim * n_eval_grad_points, direct_values.rows());
    EXPECT_EQ(n_eval_points + Dim * n_eval_grad_points, values.rows());

    EXPECT_LT(relative_error(values, direct_values), rel_tolerance);

    n_points *= 8;
    n_grad_points *= 8;
    n_eval_points *= 8;
    n_eval_grad_points *= 8;
  }
}

}  // namespace

TEST(rbf_evaluator, trivial) {
  test<3>(-1, 1024, 0, 1024, 0);
  test<3>(-1, 0, 256, 0, 256);

  for (auto deg = -1; deg <= 2; deg++) {
    test<1>(deg, 1024, 256, 1024, 256);
    test<2>(deg, 1024, 256, 1024, 256);
    test<3>(deg, 1024, 256, 1024, 256);
  }
}
