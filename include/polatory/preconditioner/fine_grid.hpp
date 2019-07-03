// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>

namespace polatory {
namespace preconditioner {

class fine_grid {
public:
  fine_grid(const model& model,
            std::shared_ptr<polynomial::lagrange_basis> lagrange_basis,
            const std::vector<size_t>& point_indices,
            const std::vector<bool>& inner_point);

  fine_grid(const model& model,
            std::shared_ptr<polynomial::lagrange_basis> lagrange_basis,
            const std::vector<size_t>& point_indices,
            const std::vector<bool>& inner_point,
            const geometry::points3d& points_full);

  void clear();

  void setup(const geometry::points3d& points_full);

  void set_solution_to(Eigen::Ref<common::valuesd> weights_full) const;

  void solve(const Eigen::Ref<const common::valuesd>& values_full);

private:
  const model model_;
  const std::shared_ptr<polynomial::lagrange_basis> lagrange_basis_;
  const std::vector<size_t> point_idcs_;
  const std::vector<bool> inner_point_;

  const size_t l_;
  const size_t m_;

  // Matrix -E.
  Eigen::MatrixXd me_;

  // Cholesky decomposition of matrix Q^T A Q.
  Eigen::LDLT<Eigen::MatrixXd> ldlt_of_qtaq_;

  // Current solution.
  common::valuesd lambda_;
};

}  // namespace preconditioner
}  // namespace polatory
