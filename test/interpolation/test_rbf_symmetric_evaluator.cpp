#include <gtest/gtest.h>

#include <Eigen/Core>
#include <format>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/rbf/inverse_multiquadric.hpp>
#include <polatory/types.hpp>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::precision;
using polatory::common::valuesd;
using polatory::geometry::pointsNd;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_symmetric_evaluator;
using polatory::rbf::inverse_multiquadric1;

namespace {

template <int Dim>
void test(int poly_degree, index_t n_points, index_t n_grad_points, index_t n_eval_points,
          index_t n_eval_grad_points) {
  std::cout << std::format("dim = {}, deg = {}", Dim, poly_degree) << std::endl;

  using Rbf = inverse_multiquadric1<Dim>;
  using Points = pointsNd<Dim>;

  auto rel_tolerance = 1e-10;

  Rbf rbf({1.0, 0.01});
  rbf.set_anisotropy(random_anisotropy<Dim>());

  model model(rbf, poly_degree);

  Points points = Points::Random(n_points, Dim);
  Points grad_points = Points::Random(n_grad_points, Dim);

  rbf_direct_evaluator direct_eval(model, points, grad_points);
  direct_eval.set_target_points(points.topRows(n_eval_points),
                                grad_points.topRows(n_eval_grad_points));

  rbf_symmetric_evaluator eval(model, points, grad_points, precision::kPrecise);

  for (auto i = 0; i < 2; i++) {
    valuesd weights = valuesd::Random(n_points + Dim * n_grad_points + model.poly_basis_size());

    direct_eval.set_weights(weights);
    eval.set_weights(weights);

    auto direct_values = direct_eval.evaluate();
    auto values_full = eval.evaluate();

    EXPECT_EQ(n_eval_points + Dim * n_eval_grad_points, direct_values.rows());
    EXPECT_EQ(n_points + Dim * n_grad_points, values_full.rows());

    valuesd values(direct_values.size());
    values << values_full.head(n_eval_points),
        values_full.segment(n_points, Dim * n_eval_grad_points);

    EXPECT_LT(relative_error(values, direct_values), rel_tolerance);
  }
}

}  // namespace

TEST(rbf_symmetric_evaluator, trivial) {
  for (auto deg = -1; deg <= 2; deg++) {
    test<1>(deg, 32768, 4096, 1024, 1024);
    test<2>(deg, 32768, 4096, 1024, 1024);
    test<3>(deg, 32768, 4096, 1024, 1024);
  }
}
