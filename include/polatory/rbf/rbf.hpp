#pragma once

#include <memory>
#include <polatory/common/io.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <string>
#include <vector>

namespace polatory::rbf {

template <int Dim>
class Rbf {
  static constexpr int kDim = Dim;
  using Mat = Mat<Dim>;
  using RbfBase = internal::RbfBase<Dim>;
  using Vector = geometry::Vector<Dim>;

 protected:
  explicit Rbf(std::unique_ptr<RbfBase>&& rbf) : rbf_(std::move(rbf)) {}

 public:
  ~Rbf() = default;

  // Checking if other.rbf_ has a value is required for copying a default-constructed
  // Rbf on deserialization of a std::vector<Rbf>, etc.
  Rbf(const Rbf& other) : rbf_(other.rbf_ ? other.rbf_->clone() : nullptr) {}

  Rbf(Rbf&& other) = default;

  Rbf& operator=(const Rbf& other) {
    if (this == &other) {
      return *this;
    }

    rbf_ = other.rbf_->clone();
    return *this;
  }

  Rbf& operator=(Rbf&& other) = default;

  const Mat& anisotropy() const { return rbf_->anisotropy(); }

  int cpd_order() const { return rbf_->cpd_order(); }

  double evaluate(const Vector& diff) const { return rbf_->evaluate(diff); }

  Vector evaluate_gradient(const Vector& diff) const { return rbf_->evaluate_gradient(diff); }

  Mat evaluate_hessian(const Vector& diff) const { return rbf_->evaluate_hessian(diff); }

  double evaluate_isotropic(const Vector& diff) const { return rbf_->evaluate_isotropic(diff); }

  Vector evaluate_gradient_isotropic(const Vector& diff) const {
    return rbf_->evaluate_gradient_isotropic(diff);
  }

  Mat evaluate_hessian_isotropic(const Vector& diff) const {
    return rbf_->evaluate_hessian_isotropic(diff);
  }

  RbfBase* get_raw_pointer() const { return rbf_.get(); }

  bool is_covariance_function() const { return rbf_->is_covariance_function(); }

  Index num_parameters() const { return rbf_->num_parameters(); }

  const std::vector<double>& parameter_lower_bounds() const {
    return rbf_->parameter_lower_bounds();
  }

  const std::vector<std::string>& parameter_names() const { return rbf_->parameter_names(); }

  const std::vector<double>& parameter_upper_bounds() const {
    return rbf_->parameter_upper_bounds();
  }

  const std::vector<double>& parameters() const { return rbf_->parameters(); }

  void set_anisotropy(const Mat& aniso) { rbf_->set_anisotropy(aniso); }

  void set_parameters(const std::vector<double>& params) { rbf_->set_parameters(params); }

  std::string short_name() const { return rbf_->short_name(); }

  double support_radius_isotropic() const { return rbf_->support_radius_isotropic(); }

  bool operator==(const Rbf& other) const {
    if (this == &other) {
      return true;
    }

    return short_name() == other.short_name() && parameters() == other.parameters() &&
           anisotropy() == other.anisotropy();
  }

  bool operator!=(const Rbf& other) const { return !(*this == other); }

 private:
  POLATORY_FRIEND_READ_WRITE;

  // For deserialization.
  Rbf() = default;

  std::unique_ptr<RbfBase> rbf_;
};

#define POLATORY_DEFINE_RBF(RBF_NAME)                                     \
  template <int Dim>                                                      \
  class RBF_NAME : public Rbf<Dim> {                                      \
   private:                                                               \
    using RbfInternal = internal::RBF_NAME<Dim>;                          \
                                                                          \
   public:                                                                \
    static inline const std::string kShortName = RbfInternal::kShortName; \
                                                                          \
    explicit RBF_NAME(const std::vector<double>& params)                  \
        : Rbf<Dim>(std::make_unique<RbfInternal>(params)) {}              \
  };

}  // namespace polatory::rbf
