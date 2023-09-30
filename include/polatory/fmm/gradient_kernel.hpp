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
struct gradient_kernel {
  static constexpr int kDim = Rbf::kDim;
  using Vector = geometry::vectorNd<kDim>;
  using Matrix = geometry::matrixNd<kDim>;

  static constexpr auto homogeneity_tag{scalfmm::matrix_kernels::homogeneity::non_homogenous};
  static constexpr auto symmetry_tag{scalfmm::matrix_kernels::symmetry::non_symmetric};
  static constexpr std::size_t km{kDim};
  static constexpr std::size_t kn{1};
  template <typename ValueType>
  using matrix_type = std::array<ValueType, kn * km>;
  template <typename ValueType>
  using vector_type = std::array<ValueType, kn>;

  explicit gradient_kernel(const Rbf& rbf) : rbf_(rbf) {}

  gradient_kernel(const gradient_kernel&) = default;

  const std::string name() const { return std::string(""); }

  template <typename ValueType>
  [[nodiscard]] inline constexpr auto mutual_coefficient() const {
    using decayed_type = typename std::decay_t<ValueType>;
    return vector_type<decayed_type>{decayed_type(-1.0)};
  }

  [[nodiscard]] inline auto evaluate(
      scalfmm::container::point<double, kDim> const& x,
      scalfmm::container::point<double, kDim> const& y) const noexcept {
    Vector diff;
    for (auto i = 0; i < kDim; i++) {
      diff(i) = x.at(i) - y.at(i);
    }

    Vector g = rbf_.evaluate_gradient_isotropic(diff) * rbf_.anisotropy();

    matrix_type<double> result;
    for (auto i = 0; i < kDim; i++) {
      result.at(i) = -g(i);
    }

    return result;
  }

  [[nodiscard]] inline auto evaluate(
      scalfmm::container::point<xsimd::batch<double>, kDim> const& x,
      scalfmm::container::point<xsimd::batch<double>, kDim> const& y) const noexcept {
    using decayed_type = typename std::decay_t<xsimd::batch<double>>;

    std::array<double, 4> v0;
    std::array<double, 4> v1;
    std::array<double, 4> v2;

    for (std::size_t i = 0; i < x.at(0).size; i++) {
      Vector diff;
      for (auto j = 0; j < kDim; j++) {
        diff(j) = x.at(j).get(i) - y.at(j).get(i);
      }

      Vector g = rbf_.evaluate_gradient_isotropic(diff) * rbf_.anisotropy();

      v0.at(i) = -g(0);
      v1.at(i) = -g(1);
      v2.at(i) = -g(2);
    }

    matrix_type<decayed_type> result;
    result.at(0) = decayed_type::load(v0.data(), xsimd::unaligned_mode{});
    if constexpr (kDim > 1) {
      result.at(1) = decayed_type::load(v1.data(), xsimd::unaligned_mode{});
    }
    if constexpr (kDim > 2) {
      result.at(2) = decayed_type::load(v2.data(), xsimd::unaligned_mode{});
    }

    return result;
  }

  static constexpr int separation_criterion{1};

 private:
  const Rbf rbf_;
};

}  // namespace fmm
}  // namespace polatory