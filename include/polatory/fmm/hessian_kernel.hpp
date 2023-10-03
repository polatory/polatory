#pragma once

#include <Eigen/Core>
#include <array>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <scalfmm/container/point.hpp>
#include <scalfmm/matrix_kernels/mk_common.hpp>
#include <string>
#include <type_traits>

namespace polatory {
namespace fmm {

template <class Rbf>
struct hessian_kernel {
  static constexpr int kDim = Rbf::kDim;
  using Vector = geometry::vectorNd<kDim>;
  using Matrix = geometry::matrixNd<kDim>;

  static constexpr auto homogeneity_tag{scalfmm::matrix_kernels::homogeneity::non_homogenous};
  static constexpr auto symmetry_tag{scalfmm::matrix_kernels::symmetry::symmetric};
  static constexpr std::size_t km{kDim};
  static constexpr std::size_t kn{kDim};
  template <typename ValueType>
  using matrix_type = std::array<ValueType, kn * km>;
  template <typename ValueType>
  using vector_type = std::array<ValueType, kn>;

  explicit hessian_kernel(const Rbf& rbf) : rbf_(rbf) {}

  hessian_kernel(const hessian_kernel&) = default;

  const std::string name() const { return std::string(""); }

  template <typename ValueType>
  [[nodiscard]] inline constexpr auto mutual_coefficient() const {
    vector_type<ValueType> mc;
    std::fill(std::begin(mc), std::end(mc), ValueType(1.0));
    return mc;
  }

  [[nodiscard]] inline auto evaluate(
      scalfmm::container::point<double, kDim> const& x,
      scalfmm::container::point<double, kDim> const& y) const noexcept {
    Vector diff;
    for (auto i = 0; i < kDim; i++) {
      diff(i) = x.at(i) - y.at(i);
    }

    auto a = rbf_.anisotropy();
    Matrix h = a.transpose() * rbf_.evaluate_hessian_isotropic(diff) * a;

    matrix_type<double> result;
    for (auto i = 0; i < kDim; i++) {
      for (auto j = 0; j < kDim; j++) {
        result.at(kDim * i + j) = -h(i, j);
      }
    }

    return result;
  }

  [[nodiscard]] inline auto evaluate(
      scalfmm::container::point<xsimd::batch<double>, kDim> const& x,
      scalfmm::container::point<xsimd::batch<double>, kDim> const& y) const noexcept {
    using decayed_type = typename std::decay_t<xsimd::batch<double>>;

    std::array<std::array<double, 4>, kDim * kDim> v;

    auto a = rbf_.anisotropy();
    for (std::size_t i = 0; i < x.at(0).size; i++) {
      Vector diff;
      for (auto j = 0; j < kDim; j++) {
        diff(j) = x.at(j).get(i) - y.at(j).get(i);
      }

      Matrix h = a.transpose() * rbf_.evaluate_hessian_isotropic(diff) * a;

      for (auto j = 0; j < kDim; j++) {
        for (auto k = 0; k < kDim; k++) {
          v.at(kDim * j + k).at(i) = -h(j, k);
        }
      }
    }

    matrix_type<decayed_type> result;
    for (auto i = 0; i < kDim * kDim; i++) {
      result.at(i) = decayed_type::load(v.at(i).data(), xsimd::unaligned_mode{});
    }

    return result;
  }

  static constexpr int separation_criterion{1};

 private:
  const Rbf rbf_;
};

}  // namespace fmm
}  // namespace polatory
