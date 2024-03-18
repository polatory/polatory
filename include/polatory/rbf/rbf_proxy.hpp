#pragma once

#include <memory>
#include <polatory/rbf/rbf_base.hpp>
#include <string>
#include <vector>

namespace polatory::rbf {

template <int Dim>
class rbf_proxy {
  static constexpr int kDim = Dim;
  using Matrix = geometry::matrixNd<Dim>;
  using RbfBase = internal::rbf_base<Dim>;
  using Vector = geometry::vectorNd<Dim>;

 protected:
  explicit rbf_proxy(std::unique_ptr<RbfBase>&& rbf) : rbf_(std::move(rbf)) {}

 public:
  ~rbf_proxy() = default;

  rbf_proxy(const rbf_proxy& other) : rbf_(other.rbf_->clone()) {}

  rbf_proxy(rbf_proxy&& other) = default;

  rbf_proxy& operator=(const rbf_proxy& other) {
    if (this == &other) {
      return *this;
    }

    rbf_ = other.rbf_->clone();
    return *this;
  }

  rbf_proxy& operator=(rbf_proxy&& other) = default;

  const Matrix& anisotropy() const { return rbf_->anisotropy(); }

  int cpd_order() const { return rbf_->cpd_order(); }

  double evaluate(const Vector& diff) const { return rbf_->evaluate(diff); }

  Vector evaluate_gradient(const Vector& diff) const { return rbf_->evaluate_gradient(diff); }

  Matrix evaluate_hessian(const Vector& diff) const { return rbf_->evaluate_hessian(diff); }

  double evaluate_isotropic(const Vector& diff) const { return rbf_->evaluate_isotropic(diff); }

  Vector evaluate_gradient_isotropic(const Vector& diff) const {
    return rbf_->evaluate_gradient_isotropic(diff);
  }

  Matrix evaluate_hessian_isotropic(const Vector& diff) const {
    return rbf_->evaluate_hessian_isotropic(diff);
  }

  RbfBase* get_raw_pointer() const { return rbf_.get(); }

  int num_parameters() const { return rbf_->num_parameters(); }

  const std::vector<double>& parameter_lower_bounds() const {
    return rbf_->parameter_lower_bounds();
  }

  const std::vector<std::string>& parameter_names() const { return rbf_->parameter_names(); }

  const std::vector<double>& parameter_upper_bounds() const {
    return rbf_->parameter_upper_bounds();
  }

  const std::vector<double>& parameters() const { return rbf_->parameters(); }

  void set_anisotropy(const Matrix& aniso) { rbf_->set_anisotropy(aniso); }

  void set_parameters(const std::vector<double>& params) { rbf_->set_parameters(params); }

  double support_radius_isotropic() const { return rbf_->support_radius_isotropic(); }

 private:
  std::unique_ptr<RbfBase> rbf_;
};

#define DEFINE_RBF(RBF_NAME)                                                   \
  template <int Dim>                                                           \
  class RBF_NAME : public rbf_proxy<Dim> {                                     \
   public:                                                                     \
    explicit RBF_NAME(const std::vector<double>& params)                       \
        : rbf_proxy<Dim>(std::make_unique<internal::RBF_NAME<Dim>>(params)) {} \
  };

}  // namespace polatory::rbf
