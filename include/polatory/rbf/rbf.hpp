// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>

#include <polatory/common/exception.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/rbf/rbf_kernel.hpp>

namespace polatory {
namespace rbf {

class rbf {
public:
  rbf() {
  }

  rbf(const rbf_kernel& kernel, int poly_dimension, int poly_degree)
    : kern_(kernel.clone())
    , poly_dimension_(poly_dimension)
    , poly_degree_(poly_degree) {
    if (poly_degree < cpd_order() - 1 || poly_degree > 2)
      throw common::invalid_argument("cpd_order() - 1 <= poly_degree <= 2");
  }

  int poly_basis_size() const {
    return polynomial::basis_base::basis_size(poly_dimension_, poly_degree_);
  }

  int cpd_order() const {
    return kern_->cpd_order();
  }

  const rbf_kernel& get() const {
    return *kern_;
  }

  rbf_kernel& get() {
    return *kern_;
  }

  double nugget() const {
    return kern_->nugget();
  }

  int poly_degree() const {
    return poly_degree_;
  }

  int poly_dimension() const {
    return poly_dimension_;
  }

  // This method is for internal use only.
  rbf without_poly() const {
    return rbf(*kern_);
  }

private:
  explicit rbf(const rbf_kernel& kernel)
    : kern_(kernel.clone())
    , poly_dimension_(-1)
    , poly_degree_(-1) {
  }

  std::shared_ptr<rbf_kernel> kern_;
  int poly_dimension_;
  int poly_degree_;
};

}  // namespace rbf
}  // namespace polatory
