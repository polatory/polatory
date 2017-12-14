// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#define POLATORY_REPORT_RESIDUAL 0
#define POLATORY_RECOMPUTE_AND_CLEAR 1

#include <memory>
#include <vector>

#include <Eigen/Core>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/preconditioner/fine_grid.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace preconditioner {

class ras_preconditioner : public krylov::linear_operator {
  static constexpr int Order = 6;
  static constexpr const double coarse_ratio = 0.125;
  static constexpr const size_t n_coarsest_points = 1024;

public:
  ras_preconditioner(const rbf::rbf& rbf, int poly_dimension, int poly_degree,
                     const geometry::points3d& in_points);

  common::valuesd operator()(const common::valuesd& v) const override;

  size_t size() const override;

private:
  const geometry::points3d points_;
  const size_t n_points_;
  const size_t n_poly_basis_;
  int n_fine_levels_;

#if POLATORY_REPORT_RESIDUAL
  mutable interpolation::rbf_symmetric_evaluator<Order> finest_evaluator_;
#endif

  std::vector<std::vector<size_t>> point_idcs_;
  mutable std::vector<std::vector<fine_grid>> fine_grids_;
  std::shared_ptr<polynomial::lagrange_basis> lagrange_basis_;
  std::unique_ptr<coarse_grid> coarse_;
  std::vector<interpolation::rbf_evaluator<Order>> downward_evaluator_;
  std::vector<interpolation::rbf_evaluator<Order>> upward_evaluator_;
  Eigen::MatrixXd p_;
  Eigen::MatrixXd ap_;
};

} // namespace preconditioner
} // namespace polatory
