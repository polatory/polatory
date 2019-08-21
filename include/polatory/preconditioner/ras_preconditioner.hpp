// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/preconditioner/fine_grid.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace preconditioner {

class ras_preconditioner : public krylov::linear_operator {
  static constexpr bool kRecomputeAndClear = true;
  static constexpr bool kReportResidual = false;
  static constexpr int Order = 6;
  static constexpr double coarse_ratio = 0.125;
  static constexpr index_t n_coarsest_points = 1024;

public:
  ras_preconditioner(const model& model, const geometry::points3d& in_points);

  common::valuesd operator()(const common::valuesd& v) const override;

  index_t size() const override;

private:
  const model model_without_poly_;
  const geometry::points3d points_;
  const index_t n_points_;
  const index_t n_poly_basis_;
  const std::unique_ptr<interpolation::rbf_symmetric_evaluator<Order>> finest_evaluator_;

  std::unique_ptr<polynomial::lagrange_basis> lagrange_basis_;
  int n_fine_levels_;
  std::vector<std::vector<index_t>> point_idcs_;
  mutable std::vector<std::vector<fine_grid>> fine_grids_;
  std::unique_ptr<coarse_grid> coarse_;
  std::vector<interpolation::rbf_evaluator<Order>> downward_evaluator_;
  std::vector<interpolation::rbf_evaluator<Order>> upward_evaluator_;
  Eigen::MatrixXd p_;
  Eigen::MatrixXd ap_;
};

}  // namespace preconditioner
}  // namespace polatory
