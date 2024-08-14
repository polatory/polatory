#pragma once

#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <numbers>
#include <polatory/common/io.hpp>
#include <polatory/geometry/anisotropy.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/rbf/rbf.hpp>
#include <polatory/types.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace polatory {

template <int Dim>
class Interpolant;

template <int Dim>
class Model {
  static constexpr int kDim = Dim;
  using Rbf = rbf::Rbf<kDim>;

 public:
  // Non-constexpr for the sake of Python bindings.
  static inline const int kMinRequiredPolyDegree = -2;

  explicit Model(Rbf&& rbf, int poly_degree = kMinRequiredPolyDegree)
      : Model(std::vector<Rbf>{std::move(rbf)}, poly_degree) {}

  explicit Model(std::vector<Rbf>&& rbfs, int poly_degree = kMinRequiredPolyDegree)
      : rbfs_(std::move(rbfs)) {
    if (rbfs_.empty()) {
      throw std::invalid_argument("rbfs must not be empty");
    }

    auto min_poly_degree = cpd_order() - 1;
    auto max_poly_degree = 2;
    if (poly_degree == kMinRequiredPolyDegree) {
      poly_degree_ = min_poly_degree;
    } else if (poly_degree >= min_poly_degree && poly_degree <= max_poly_degree) {
      poly_degree_ = poly_degree;
    } else {
      throw std::invalid_argument(
          std::format("poly_degree must be within {} to {}", min_poly_degree, max_poly_degree));
    }
  }

  ~Model() = default;

  Model(const Model& model) = default;
  Model(Model&& model) = default;
  Model& operator=(const Model&) = default;
  Model& operator=(Model&&) = default;

  int cpd_order() const {
    auto order = 0;
    for (const auto& rbf : rbfs_) {
      order = std::max(order, rbf.cpd_order());
    }
    return order;
  }

  std::string description() const;

  bool is_covariance_model() const {
    for (const auto& rbf : rbfs_) {
      if (!rbf.is_covariance_function()) {
        return false;
      }
    }
    return true;
  }

  double nugget() const { return nugget_; }

  Index num_parameters() const {
    Index np = 1;
    for (const auto& rbf : rbfs_) {
      np += rbf.num_parameters();
    }
    return np;
  }

  Index num_rbfs() const { return static_cast<Index>(rbfs_.size()); }

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

  Index poly_basis_size() const {
    return polynomial::PolynomialBasisBase<kDim>::basis_size(poly_degree_);
  }

  int poly_degree() const { return poly_degree_; }

  std::vector<Rbf>& rbfs() { return rbfs_; }

  const std::vector<Rbf>& rbfs() const { return rbfs_; }

  void set_nugget(double nugget) {
    if (!(nugget >= 0.0)) {
      throw std::invalid_argument("nugget must be non-negative");
    }

    nugget_ = nugget;
  }

  void set_parameters(const std::vector<double>& params) {
    if (static_cast<Index>(params.size()) != num_parameters()) {
      throw std::invalid_argument(std::format("params.size() must be {}", num_parameters()));
    }

    set_nugget(params.at(0));

    Index i = 1;
    for (auto& rbf : rbfs_) {
      rbf.set_parameters(
          std::vector<double>(params.begin() + i, params.begin() + i + rbf.num_parameters()));
      i += rbf.num_parameters();
    }
  }

  bool operator==(const Model& other) const {
    if (this == &other) {
      return true;
    }

    return rbfs() == other.rbfs() && poly_degree() == other.poly_degree() &&
           nugget() == other.nugget();
  }

  bool operator!=(const Model& other) const { return !(*this == other); }

  POLATORY_IMPLEMENT_LOAD_SAVE(Model);

 private:
  POLATORY_FRIEND_READ_WRITE;

  // For deserialization of an Interpolant<Dim>.
  friend class Interpolant<Dim>;

  // For deserialization.
  Model() = default;

  std::vector<Rbf> rbfs_;
  int poly_degree_{};
  double nugget_{};
};

template <>
inline std::string Model<1>::description() const {
  if (!is_covariance_model()) {
    throw std::runtime_error("description is only available for covariance models");
  }

  std::stringstream ss;
  ss << "        Type       Psill       Range\n";

  ss << std::format("         nug  {:>10.4f}\n", nugget());

  for (const auto& rbf : rbfs()) {
    auto type = rbf.short_name();
    auto [rotation, scale] = geometry::decompose_inverse_anisotropy(rbf.anisotropy());
    auto psill = rbf.parameters().at(0);
    auto range = scale(0) * rbf.parameters().at(1);

    ss << std::format("  {:>10}  {:>10.4f}  {:>10.4f}\n", type, psill, range);
  }

  return ss.str();
}

template <>
inline std::string Model<2>::description() const {
  if (!is_covariance_model()) {
    throw std::runtime_error("description is only available for covariance models");
  }

  auto deg = std::numbers::pi / 180.0;

  std::stringstream ss;
  ss << "        Type       Psill       Major       Minor    Rotation\n";

  ss << std::format("         nug  {:>10.4f}\n", nugget());

  for (const auto& rbf : rbfs()) {
    auto type = rbf.short_name();
    auto [rotation, scale] = geometry::decompose_inverse_anisotropy(rbf.anisotropy());
    auto psill = rbf.parameters().at(0);
    auto range = rbf.parameters().at(1);
    auto major = scale(0) * range;
    auto minor = scale(1) * range;

    auto rot = -std::atan2(rotation(1, 0), rotation(0, 0)) / deg;
    if (rot < 0.0) {
      rot += 180.0;
    }

    ss << std::format("  {:>10}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}\n", type, psill, major,
                      minor, rot);
  }

  return ss.str();
}

template <>
inline std::string Model<3>::description() const {
  if (!is_covariance_model()) {
    throw std::runtime_error("description is only available for covariance models");
  }

  auto deg = std::numbers::pi / 180.0;

  std::stringstream ss;
  ss << "        Type       Psill       Major  Semi-major       Minor"
        "     Dip Az.         Dip    Rotation\n";

  ss << std::format("         nug  {:>10.4f}\n", nugget());

  for (const auto& rbf : rbfs()) {
    auto type = rbf.short_name();
    auto [rotation, scale] = geometry::decompose_inverse_anisotropy(rbf.anisotropy());
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
      rot = -rot;
    } else if (dip < 0.0) {
      dip = -dip;
      az += 180.0;
    } else if (dip > 90.0) {
      dip = 180.0 - dip;
      az += 180.0;
      rot = -rot;
    }
    if (az < 0.0) {
      az += 360.0;
    }
    if (rot < 0.0) {
      rot += 180.0;
    }

    ss << std::format(
        "  {:>10}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}\n",
        type, psill, major, semi_major, minor, az, dip, rot);
  }

  return ss.str();
}

}  // namespace polatory

namespace polatory::common {

template <int Dim>
struct Read<Model<Dim>> {
  void operator()(std::istream& is, Model<Dim>& t) const {
    read(is, t.rbfs_);
    read(is, t.poly_degree_);
    read(is, t.nugget_);
  }
};

template <int Dim>
struct Write<Model<Dim>> {
  void operator()(std::ostream& os, const Model<Dim>& t) const {
    write(os, t.rbfs_);
    write(os, t.poly_degree_);
    write(os, t.nugget_);
  }
};

}  // namespace polatory::common
