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
        result.at(kDim * i + j) = h(i, j);
      }
    }

    return result;
  }

  [[nodiscard]] inline auto evaluate(
      scalfmm::container::point<xsimd::batch<double>, kDim> const& x,
      scalfmm::container::point<xsimd::batch<double>, kDim> const& y) const noexcept {
    using decayed_type = typename std::decay_t<xsimd::batch<double>>;

    std::array<double, 4> v00;
    std::array<double, 4> v01;
    std::array<double, 4> v02;
    std::array<double, 4> v11;
    std::array<double, 4> v12;
    std::array<double, 4> v22;

    auto a = rbf_.anisotropy();
    for (std::size_t i = 0; i < x.at(0).size; i++) {
      Vector diff;
      for (auto j = 0; j < kDim; j++) {
        diff(j) = x.at(j).get(i) - y.at(j).get(i);
      }

      Matrix h = a.transpose() * rbf_.evaluate_hessian_isotropic(diff) * a;

      v00.at(i) = h(0, 0);
      v01.at(i) = h(0, 1);
      v02.at(i) = h(0, 2);
      v11.at(i) = h(1, 1);
      v12.at(i) = h(1, 2);
      v22.at(i) = h(2, 2);
    }

    matrix_type<decayed_type> result;
    result.at(kDim * 0 + 0) = decayed_type::load(v00.data(), xsimd::unaligned_mode{});
    if constexpr (kDim > 1) {
      result.at(kDim * 0 + 1) = decayed_type::load(v01.data(), xsimd::unaligned_mode{});
      result.at(kDim * 1 + 0) = decayed_type::load(v01.data(), xsimd::unaligned_mode{});
      result.at(kDim * 1 + 1) = decayed_type::load(v11.data(), xsimd::unaligned_mode{});
    }
    if constexpr (kDim > 2) {
      result.at(kDim * 0 + 2) = decayed_type::load(v02.data(), xsimd::unaligned_mode{});
      result.at(kDim * 1 + 2) = decayed_type::load(v12.data(), xsimd::unaligned_mode{});
      result.at(kDim * 2 + 0) = decayed_type::load(v02.data(), xsimd::unaligned_mode{});
      result.at(kDim * 2 + 1) = decayed_type::load(v12.data(), xsimd::unaligned_mode{});
      result.at(kDim * 2 + 2) = decayed_type::load(v22.data(), xsimd::unaligned_mode{});
    }

    return result;
  }

  static constexpr int separation_criterion{1};

 private:
  const Rbf rbf_;
};

}  // namespace fmm
}  // namespace polatory
