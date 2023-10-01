#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/rbf/reference/cov_gaussian.hpp>
#include <polatory/types.hpp>

#include "../random_anisotropy.hpp"
#include "utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::precision;
using polatory::common::valuesd;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_symmetric_evaluator;
using polatory::point_cloud::random_points;
using polatory::rbf::reference::cov_gaussian;

namespace {

void test(int poly_degree, index_t n_points, index_t n_grad_points, index_t n_eval_points,
          index_t n_eval_grad_points) {
  constexpr int kDim = 3;
  using Rbf = cov_gaussian<kDim>;
  using Model = model<Rbf>;

  auto rel_tolerance = 1e-10;

  Rbf rbf({1.0, 0.01});
  rbf.set_anisotropy(random_anisotropy());

  Model model(rbf, poly_degree);

  auto points = random_points(sphere3d(), n_points);
  auto grad_points = random_points(sphere3d(), n_grad_points);

  rbf_direct_evaluator<Model> direct_eval(model, points, grad_points);
  direct_eval.set_target_points(points.topRows(n_eval_points),
                               grad_points.topRows(n_eval_grad_points));

  rbf_symmetric_evaluator<Model> eval(model, points, grad_points, precision::kPrecise);

  for (auto i = 0; i < 2; i++) {
    valuesd weights = valuesd::Random(n_points + kDim * n_grad_points + model.poly_basis_size());

    direct_eval.set_weights(weights);
    eval.set_weights(weights);

    auto direct_values = direct_eval.evaluate();
    auto values_full = eval.evaluate();

    EXPECT_EQ(n_eval_points + kDim * n_eval_grad_points, direct_values.rows());
    EXPECT_EQ(n_points + kDim * n_grad_points, values_full.rows());

    valuesd values(direct_values.size());
    values << values_full.head(n_eval_points),
        values_full.segment(n_points, kDim * n_eval_grad_points);

    EXPECT_LT(relative_error(values, direct_values), rel_tolerance);
  }
}

}  // namespace

TEST(rbf_symmetric_evaluator, trivial) {
  for (auto deg = -1; deg <= 2; deg++) {
    test(deg, 32768, 4096, 1024, 1024);
  }
}
