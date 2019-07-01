// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <limits>
#include <memory>
#include <vector>

#include <polatory/common/exception.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/rbf/rbf_base.hpp>

namespace polatory {

class model {
public:
  model()
    : poly_dimension_(-1)
    , poly_degree_(-1)
    , nugget_(0.0) {
  }

  model(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree)
    : rbf_(rbf.clone())
    , poly_dimension_(poly_dimension)
    , poly_degree_(poly_degree)
    , nugget_(0.0) {
    if (poly_degree < rbf.cpd_order() - 1 || poly_degree > 2)
      throw common::invalid_argument("rbf.cpd_order() - 1 <= poly_degree <= 2");
  }

  ~model() = default;

  model(const model& model)
    : rbf_(model.rbf_->clone())
    , poly_dimension_(model.poly_dimension_)
    , poly_degree_(model.poly_degree_)
    , nugget_(model.nugget_) {
  }

  model(model&& model) = default;

  model& operator=(const model&) = delete;
  model& operator=(model&&) = delete;

  double nugget() const {
    return nugget_;
  }

  // Experimental function.
  size_t num_parameters() const {
    return 1 + rbf_->num_parameters();
  }

  // Experimental function.
  std::vector<double> parameter_lower_bounds() const {
    std::vector<double> lower_bounds{ 0.0 };
    lower_bounds.insert(lower_bounds.end(), rbf_->parameter_lower_bounds().begin(), rbf_->parameter_lower_bounds().end());
    return lower_bounds;
  }

  // Experimental function.
  std::vector<double> parameter_upper_bounds() const {
    std::vector<double> upper_bounds{ std::numeric_limits<double>::infinity() };
    upper_bounds.insert(upper_bounds.end(), rbf_->parameter_upper_bounds().begin(), rbf_->parameter_upper_bounds().end());
    return upper_bounds;
  }

  // Experimental function.
  std::vector<double> parameters() const {
    std::vector<double> params{ nugget() };
    params.insert(params.end(), rbf_->parameters().begin(), rbf_->parameters().end());
    return params;
  }

  int poly_basis_size() const {
    return polynomial::polynomial_basis_base::basis_size(poly_dimension_, poly_degree_);
  }

  int poly_degree() const {
    return poly_degree_;
  }

  int poly_dimension() const {
    return poly_dimension_;
  }

  const rbf::rbf_base& rbf() const {
    return *rbf_;
  }

  void set_nugget(double nugget) {
    nugget_ = nugget;
  }

  // Experimental function.
  void set_parameters(const std::vector<double>& params) {
    set_nugget(params[0]);
    rbf_->set_parameters(std::vector<double>(params.begin() + 1, params.end()));
  }

  // This method is for internal use only.
  model without_poly() const {
    return model(*rbf_);
  }

private:
  explicit model(const rbf::rbf_base& rbf)
    : rbf_(rbf.clone())
    , poly_dimension_(-1)
    , poly_degree_(-1)
    , nugget_(0.0) {
  }

  std::unique_ptr<rbf::rbf_base> rbf_;
  int poly_dimension_;
  int poly_degree_;
  double nugget_;
};

}  // namespace polatory
