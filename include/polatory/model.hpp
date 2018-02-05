// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>

#include <polatory/common/exception.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/rbf/rbf_base.hpp>

namespace polatory {

class model {
public:
  model() {
  }

  model(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree)
    : rbf_(rbf.clone())
    , poly_dimension_(poly_dimension)
    , poly_degree_(poly_degree) {
    if (poly_degree < cpd_order() - 1 || poly_degree > 2)
      throw common::invalid_argument("kernel.cpd_order() - 1 <= poly_degree <= 2");
  }

  int cpd_order() const {
    return rbf_->cpd_order();
  }

  double nugget() const {
    return rbf_->nugget();
  }

  int poly_basis_size() const {
    return polynomial::basis_base::basis_size(poly_dimension_, poly_degree_);
  }

  int poly_degree() const {
    return poly_degree_;
  }

  int poly_dimension() const {
    return poly_dimension_;
  }

  rbf::rbf_base& rbf() {
    return *rbf_;
  }

  const rbf::rbf_base& rbf() const {
    return *rbf_;
  }

  // This method is for internal use only.
  model without_poly() const {
    return model(*rbf_);
  }

private:
  explicit model(const rbf::rbf_base& kernel)
    : rbf_(kernel.clone())
    , poly_dimension_(-1)
    , poly_degree_(-1) {
  }

  std::shared_ptr<rbf::rbf_base> rbf_;
  int poly_dimension_;
  int poly_degree_;
};

}  // namespace polatory
