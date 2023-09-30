#pragma once

#include <limits>
#include <memory>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <vector>

namespace polatory {

template <class Rbf>
class model {
 public:
  using rbf_type = Rbf;

  static constexpr int kDim = Rbf::kDim;

  model(const Rbf& rbf, int poly_degree) : rbf_(rbf), poly_degree_(poly_degree) {
    if (poly_degree < rbf.cpd_order() - 1 || poly_degree > 2) {
      throw std::invalid_argument(
          "poly_degree must be within the range of rbf.cpd_order() - 1 to 2.");
    }
  }

  ~model() = default;

  model(const model& model)
      : rbf_(model.rbf_), poly_degree_(model.poly_degree_), nugget_(model.nugget_) {}

  model(model&& model) = default;

  model& operator=(const model&) = delete;
  model& operator=(model&&) = delete;

  double nugget() const { return nugget_; }

  // Experimental function.
  int num_parameters() const { return 1 + rbf_.num_parameters(); }

  // Experimental function.
  std::vector<double> parameter_lower_bounds() const {
    std::vector<double> lower_bounds{0.0};
    lower_bounds.insert(lower_bounds.end(), rbf_.parameter_lower_bounds().begin(),
                        rbf_.parameter_lower_bounds().end());
    return lower_bounds;
  }

  // Experimental function.
  std::vector<double> parameter_upper_bounds() const {
    std::vector<double> upper_bounds{std::numeric_limits<double>::infinity()};
    upper_bounds.insert(upper_bounds.end(), rbf_.parameter_upper_bounds().begin(),
                        rbf_.parameter_upper_bounds().end());
    return upper_bounds;
  }

  // Experimental function.
  std::vector<double> parameters() const {
    std::vector<double> params{nugget()};
    params.insert(params.end(), rbf_.parameters().begin(), rbf_.parameters().end());
    return params;
  }

  index_t poly_basis_size() const {
    return polynomial::polynomial_basis_base<kDim>::basis_size(poly_degree_);
  }

  int poly_degree() const { return poly_degree_; }

  const Rbf& rbf() const { return rbf_; }

  void set_nugget(double nugget) {
    if (nugget < 0.0) {
      throw std::invalid_argument("nugget must be greater than or equal to 0.0.");
    }

    nugget_ = nugget;
  }

  // Experimental function.
  void set_parameters(const std::vector<double>& params) {
    if (static_cast<int>(params.size()) != num_parameters()) {
      throw std::invalid_argument("params.size() must be " + std::to_string(num_parameters()) +
                                  ".");
    }

    set_nugget(params[0]);
    rbf_.set_parameters(std::vector<double>(params.begin() + 1, params.end()));
  }

  // This method is for internal use only.
  model without_poly() const { return model(rbf_); }

 private:
  explicit model(const Rbf& rbf) : rbf_(rbf), poly_degree_(-1) {}

  Rbf rbf_;
  int poly_degree_;
  double nugget_{};
};

}  // namespace polatory
