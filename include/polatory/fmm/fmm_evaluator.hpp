#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/fmm/gradient_kernel.hpp>
#include <polatory/fmm/gradient_transpose_kernel.hpp>
#include <polatory/fmm/hessian_kernel.hpp>
#include <polatory/fmm/kernel.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>

namespace polatory::fmm {

template <class Model, class Kernel>
class fmm_generic_evaluator {
  static constexpr int kDim = Model::kDim;
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;

 public:
  fmm_generic_evaluator(const Model& model, const Bbox& bbox, int order);

  ~fmm_generic_evaluator();

  common::valuesd evaluate() const;

  void set_source_points(const Points& points);

  void set_target_points(const Points& points);

  void set_weights(const Eigen::Ref<const common::valuesd>& weights);

 private:
  class impl;

  std::unique_ptr<impl> impl_;
};

template <class Model>
using fmm_evaluator = fmm_generic_evaluator<Model, kernel<typename Model::rbf_type>>;

template <class Model>
using fmm_gradient_evaluator =
    fmm_generic_evaluator<Model, gradient_kernel<typename Model::rbf_type>>;

template <class Model>
using fmm_gradient_transpose_evaluator =
    fmm_generic_evaluator<Model, gradient_transpose_kernel<typename Model::rbf_type>>;

template <class Model>
using fmm_hessian_evaluator =
    fmm_generic_evaluator<Model, hessian_kernel<typename Model::rbf_type>>;

}  // namespace polatory::fmm
