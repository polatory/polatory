#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory::fmm {

template <int Order>
class fmm_evaluator {
 public:
  fmm_evaluator(const model& model, int tree_height, const geometry::bbox3d& bbox);

  ~fmm_evaluator();

  fmm_evaluator(const fmm_evaluator&) = delete;
  fmm_evaluator(fmm_evaluator&&) = delete;
  fmm_evaluator& operator=(const fmm_evaluator&) = delete;
  fmm_evaluator& operator=(fmm_evaluator&&) = delete;

  common::valuesd evaluate() const;

  void set_field_points(const geometry::points3d& points);

  void set_source_points(const geometry::points3d& points);

  void set_source_points_and_weights(const geometry::points3d& points,
                                     const Eigen::Ref<const common::valuesd>& weights);

  void set_weights(const Eigen::Ref<const common::valuesd>& weights);

 private:
  class impl;

  std::unique_ptr<impl> pimpl_;
};

}  // namespace polatory::fmm
