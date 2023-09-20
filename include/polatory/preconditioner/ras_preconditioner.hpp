#pragma once

#include <Eigen/Core>
#include <map>
#include <memory>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/preconditioner/fine_grid.hpp>
#include <polatory/types.hpp>
#include <tuple>
#include <utility>
#include <vector>

namespace polatory::preconditioner {

class ras_preconditioner : public krylov::linear_operator {
  static constexpr bool kRecomputeAndClear = true;
  static constexpr bool kReportResidual = false;
  static constexpr int Order = 6;
  static constexpr double coarse_ratio = 0.125;
  static constexpr index_t n_coarsest_points = 1024;

  using Evaluator = interpolation::rbf_evaluator<Order>;

 public:
  ras_preconditioner(const model& model, const geometry::points3d& in_points);

  common::valuesd operator()(const common::valuesd& v) const override;

  index_t size() const override;

 private:
  template <class... Args>
  void add_evaluator(int from_level, int to_level, Args&&... args) {
    evaluator_.emplace(std::piecewise_construct, std::forward_as_tuple(from_level, to_level),
                       std::forward_as_tuple(std::forward<Args>(args)...));
  }

  Evaluator& evaluator(int from_level, int to_level) const {
    return evaluator_.at({from_level, to_level});
  }

  const model model_without_poly_;
  const geometry::points3d points_;
  const index_t n_points_;
  const index_t n_poly_basis_;
  const std::unique_ptr<interpolation::rbf_symmetric_evaluator<Order>> finest_evaluator_;

  Eigen::MatrixXd lagrange_pt_;
  int n_levels_;
  std::vector<std::vector<index_t>> point_idcs_;
  mutable std::vector<std::vector<fine_grid>> fine_grids_;
  std::unique_ptr<coarse_grid> coarse_;
  mutable std::map<std::pair<int, int>, Evaluator> evaluator_;
  Eigen::MatrixXd p_;
  Eigen::MatrixXd ap_;
};

}  // namespace polatory::preconditioner
