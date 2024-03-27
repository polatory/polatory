#pragma once

#include <format>
#include <limits>
#include <memory>
#include <numbers>
#include <polatory/geometry/anisotropy.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/rbf/rbf_proxy.hpp>
#include <polatory/types.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace polatory {

template <int Dim>
class model {
  static constexpr int kDim = Dim;
  using RbfProxy = rbf::rbf_proxy<kDim>;

 public:
  model(RbfProxy&& rbf, int poly_degree) : poly_degree_(poly_degree) {
    rbfs_.push_back(std::move(rbf));

    if (poly_degree < cpd_order() - 1 || poly_degree > 2) {
      throw std::invalid_argument("poly_degree must be within " + std::to_string(cpd_order() - 1) +
                                  " to 2.");
    }
  }

  model(std::vector<RbfProxy>&& rbfs, int poly_degree)
      : rbfs_(std::move(rbfs)), poly_degree_(poly_degree) {
    if (poly_degree < cpd_order() - 1 || poly_degree > 2) {
      throw std::invalid_argument("poly_degree must be within " + std::to_string(cpd_order() - 1) +
                                  " to 2.");
    }
  }

  ~model() = default;

  model(const model& model) = default;
  model(model&& model) = default;
  model& operator=(const model&) = default;
  model& operator=(model&&) = default;

  int cpd_order() const {
    auto order = 0;
    for (const auto& rbf : rbfs_) {
      order = std::max(order, rbf.cpd_order());
    }
    return order;
  }

  std::string description() const { throw std::runtime_error("description() is not implemented."); }

  bool is_covariance_model() const {
    for (const auto& rbf : rbfs_) {
      if (!rbf.is_covariance_function()) {
        return false;
      }
    }
    return true;
  }

  double nugget() const { return nugget_; }

  index_t num_parameters() const {
    index_t np = 1;
    for (const auto& rbf : rbfs_) {
      np += rbf.num_parameters();
    }
    return np;
  }

  index_t num_rbfs() const { return static_cast<index_t>(rbfs_.size()); }

  std::vector<double> parameter_lower_bounds() const {
    std::vector<double> lbs{0.0};
    for (const auto& rbf : rbfs_) {
      lbs.insert(lbs.end(), rbf.parameter_lower_bounds().begin(),
                 rbf.parameter_lower_bounds().end());
    }
    return lbs;
  }

  std::vector<std::string> parameter_names() const {
    std::vector<std::string> names{"nugget"};
    for (const auto& rbf : rbfs_) {
      names.insert(names.end(), rbf.parameter_names().begin(), rbf.parameter_names().end());
    }
    return names;
  }

  std::vector<double> parameter_upper_bounds() const {
    std::vector<double> ubs{std::numeric_limits<double>::infinity()};
    for (const auto& rbf : rbfs_) {
      ubs.insert(ubs.end(), rbf.parameter_upper_bounds().begin(),
                 rbf.parameter_upper_bounds().end());
    }
    return ubs;
  }

  std::vector<double> parameters() const {
    std::vector<double> params{nugget()};
    for (const auto& rbf : rbfs_) {
      params.insert(params.end(), rbf.parameters().begin(), rbf.parameters().end());
    }
    return params;
  }

  index_t poly_basis_size() const {
    return polynomial::polynomial_basis_base<kDim>::basis_size(poly_degree_);
  }

  int poly_degree() const { return poly_degree_; }

  const RbfProxy& rbf() const { return rbfs_.at(0); }

  std::vector<RbfProxy>& rbfs() { return rbfs_; }

  const std::vector<RbfProxy>& rbfs() const { return rbfs_; }

  void set_nugget(double nugget) {
    if (nugget < 0.0) {
      throw std::invalid_argument("nugget must be greater than or equal to 0.0.");
    }

    nugget_ = nugget;
  }

  void set_parameters(const std::vector<double>& params) {
    if (static_cast<index_t>(params.size()) != num_parameters()) {
      throw std::invalid_argument("params.size() must be " + std::to_string(num_parameters()) +
                                  ".");
    }

    set_nugget(params.at(0));

    index_t i = 1;
    for (auto& rbf : rbfs_) {
      rbf.set_parameters(
          std::vector<double>(params.begin() + i, params.begin() + i + rbf.num_parameters()));
      i += rbf.num_parameters();
    }
  }

 private:
  std::vector<RbfProxy> rbfs_;
  int poly_degree_;
  double nugget_{};
};

template <>
std::string model<3>::description() const {
  if (!is_covariance_model()) {
    throw std::runtime_error("describe() is only available for covariance models.");
  }

  auto deg = std::numbers::pi / 180.0;

  std::stringstream ss;
  ss << "        Type       Psill       Major  Semi-major       Minor"
        "     Azimuth         Dip    Rotation\n";

  ss << std::format("         nug  {:>10.4f}\n", nugget());

  for (const auto& rbf : rbfs()) {
    auto type = rbf.short_name();
    auto [rotation, scale] = geometry::decompose_inverse_anisotropy<3>(rbf.anisotropy());
    auto psill = rbf.parameters().at(0);
    auto range = rbf.parameters().at(1);
    auto major = scale(0) * range;
    auto semi_major = scale(1) * range;
    auto minor = scale(2) * range;

    auto euler = rotation.eulerAngles(2, 0, 2);
    auto az = -euler(0) / deg;
    auto dip = -euler(1) / deg;
    auto rot = -euler(2) / deg;
    if (dip < -90.0) {
      dip += 180.0;
      rot = 180.0 - rot;
    } else if (dip < 0.0) {
      dip = -dip;
      az += 180.0;
    } else if (dip > 90.0) {
      dip = 180.0 - dip;
      az += 180.0;
      rot = 180.0 - rot;
    }
    if (az < 0.0) {
      az += 360.0;
    } else if (az >= 360.0) {
      az -= 360.0;
    }
    if (rot >= 180.0) {
      rot -= 180.0;
    }

    ss << std::format(
        "  {:>10}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}\n",
        type, psill, major, semi_major, minor, az, dip, rot);
  }

  return ss.str();
}

}  // namespace polatory
