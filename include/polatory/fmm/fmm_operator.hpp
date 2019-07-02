// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>

#include <polatory/common/types.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>

namespace polatory {
namespace fmm {

template <int Order>
class fmm_operator {
public:
  fmm_operator(const model& model, int tree_height, const geometry::bbox3d& bbox);

  ~fmm_operator();

  fmm_operator(const fmm_operator&) = delete;
  fmm_operator(fmm_operator&&) = delete;
  fmm_operator& operator=(const fmm_operator&) = delete;
  fmm_operator& operator=(fmm_operator&&) = delete;

  common::valuesd evaluate() const;

  void set_points(const geometry::points3d& points);

  void set_weights(const common::valuesd& weights);

private:
  class impl;

  std::unique_ptr<impl> pimpl_;
};

}  // namespace fmm
}  // namespace polatory
