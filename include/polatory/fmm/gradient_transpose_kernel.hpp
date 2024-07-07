#pragma once

#include <Eigen/Core>
#include <array>
#include <memory>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <scalfmm/container/point.hpp>
#include <scalfmm/matrix_kernels/mk_common.hpp>
#include <string>
#include <type_traits>

namespace polatory::fmm {

template <class Rbf>
struct gradient_transpose_kernel {
  static constexpr int kDim = Rbf::kDim;
  using Vector = geometry::vectorNd<kDim>;
  using Matrix = geometry::matrixNd<kDim>;

  static constexpr auto homogeneity_tag{scalfmm::matrix_kernels::homogeneity::non_homogenous};
  static constexpr auto symmetry_tag{scalfmm::matrix_kernels::symmetry::non_symmetric};
  static constexpr std::size_t km{1};
  static constexpr std::size_t kn{kDim};
  template <typename ValueType>
  using matrix_type = std::array<ValueType, kn * km>;
  template <typename ValueType>
  using vector_type = std::array<ValueType, kn>;

  explicit gradient_transpose_kernel(const Rbf& rbf) : rbf_(rbf) {}

  std::string name() const { return ""; }

  template <typename ValueType>
  constexpr auto mutual_coefficient() const {
    vector_type<ValueType> mc;
    std::fill(std::begin(mc), std::end(mc), ValueType(-1.0));
    return mc;
  }

  template <class T, class U>
  auto evaluate(scalfmm::container::point<T, kDim> const& x,
                scalfmm::container::point<U, kDim> const& y) const {
    Vector diff;
    for (auto i = 0; i < kDim; i++) {
      diff(i) = x.at(i) - y.at(i);
    }

    Vector g = rbf_.evaluate_gradient_isotropic(diff) * rbf_.anisotropy();

    matrix_type<double> result;
    for (auto i = 0; i < kDim; i++) {
      result.at(i) = g(i);
    }

    return result;
  }

  auto evaluate(scalfmm::container::point<xsimd::batch<double>, kDim> const& x,
                scalfmm::container::point<xsimd::batch<double>, kDim> const& y) const {
    using decayed_type = typename std::decay_t<xsimd::batch<double>>;

    std::array<std::array<double, 4>, kDim> v{};

    for (std::size_t i = 0; i < x.at(0).size; i++) {
      Vector diff;
      for (auto j = 0; j < kDim; j++) {
        diff(j) = x.at(j).get(i) - y.at(j).get(i);
      }

      Vector g = rbf_.evaluate_gradient_isotropic(diff) * rbf_.anisotropy();

      for (auto j = 0; j < kDim; j++) {
        v.at(j).at(i) = g(j);
      }
    }

    matrix_type<decayed_type> result;
    for (auto i = 0; i < kDim; i++) {
      result.at(i) = decayed_type::load(v.at(i).data(), xsimd::unaligned_mode{});
    }

    return result;
  }

  static constexpr int separation_criterion{1};

 private:
  const Rbf rbf_;
};

}  // namespace polatory::fmm
