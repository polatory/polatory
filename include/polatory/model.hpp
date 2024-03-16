#pragma once

#include <limits>
#include <memory>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <vector>

namespace polatory {

template <int Dim>
class model {
  static constexpr int kDim = Dim;
  using RbfPtr = rbf::RbfPtr<kDim>;

 public:
  model(RbfPtr&& rbf, int poly_degree) : poly_degree_(poly_degree) {
    rbfs_.push_back(std::move(rbf));

    if (poly_degree < cpd_order() - 1 || poly_degree > 2) {
      throw std::invalid_argument("poly_degree must be within " + std::to_string(cpd_order() - 1) +
                                  " to 2.");
    }
  }

  model(std::vector<RbfPtr>&& rbfs, int poly_degree)
      : rbfs_(std::move(rbfs)), poly_degree_(poly_degree) {
    if (poly_degree < cpd_order() - 1 || poly_degree > 2) {
      throw std::invalid_argument("poly_degree must be within " + std::to_string(cpd_order() - 1) +
                                  " to 2.");
    }
  }

  ~model() = default;

  model(const model& model) : poly_degree_(model.poly_degree_), nugget_(model.nugget_) {
    for (const auto& rbf : model.rbfs_) {
      rbfs_.push_back(rbf->clone());
    }
  }

  model(model&& model) = default;

  model& operator=(const model&) = delete;
  model& operator=(model&&) = delete;

  int cpd_order() const {
    auto order = 0;
    for (const auto& rbf : rbfs_) {
      order = std::max(order, rbf->cpd_order());
    }
    return order;
  }

  double nugget() const { return nugget_; }

  int num_parameters() const {
    auto np = 1;
    for (const auto& rbf : rbfs_) {
      np += rbf->num_parameters();
    }
    return np;
  }

  std::vector<double> parameter_lower_bounds() const {
    std::vector<double> lbs{0.0};
    for (const auto& rbf : rbfs_) {
      lbs.insert(lbs.end(), rbf->parameter_lower_bounds().begin(),
                 rbf->parameter_lower_bounds().end());
    }
    return lbs;
  }

  std::vector<double> parameter_upper_bounds() const {
    std::vector<double> ubs{std::numeric_limits<double>::infinity()};
    for (const auto& rbf : rbfs_) {
      ubs.insert(ubs.end(), rbf->parameter_upper_bounds().begin(),
                 rbf->parameter_upper_bounds().end());
    }
    return ubs;
  }

  std::vector<double> parameters() const {
    std::vector<double> params{nugget()};
    for (const auto& rbf : rbfs_) {
      params.insert(params.end(), rbf->parameters().begin(), rbf->parameters().end());
    }
    return params;
  }

  index_t poly_basis_size() const {
    return polynomial::polynomial_basis_base<kDim>::basis_size(poly_degree_);
  }

  int poly_degree() const { return poly_degree_; }

  const RbfPtr& rbf() const { return rbfs_.at(0); }

  const std::vector<RbfPtr>& rbfs() const { return rbfs_; }

  void set_nugget(double nugget) {
    if (nugget < 0.0) {
      throw std::invalid_argument("nugget must be greater than or equal to 0.0.");
    }

    nugget_ = nugget;
  }

  void set_parameters(const std::vector<double>& params) {
    if (static_cast<int>(params.size()) != num_parameters()) {
      throw std::invalid_argument("params.size() must be " + std::to_string(num_parameters()) +
                                  ".");
    }

    set_nugget(params.at(0));

    auto i = 1;
    for (const auto& rbf : rbfs_) {
      rbf->set_parameters(
          std::vector<double>(params.begin() + i, params.begin() + i + rbf->num_parameters()));
      i += rbf->num_parameters();
    }
  }

 private:
  int poly_degree_;
  double nugget_{};
  std::vector<RbfPtr> rbfs_;
};

}  // namespace polatory
