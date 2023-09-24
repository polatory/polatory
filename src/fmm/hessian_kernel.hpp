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

namespace polatory {
namespace fmm {

template <int Dim>
struct hessian_kernel {
  static constexpr auto homogeneity_tag{scalfmm::matrix_kernels::homogeneity::non_homogenous};
  static constexpr auto symmetry_tag{scalfmm::matrix_kernels::symmetry::symmetric};
  static constexpr std::size_t km{Dim};
  static constexpr std::size_t kn{Dim};
  template <typename ValueType>
  using matrix_type = std::array<ValueType, kn * km>;
  template <typename ValueType>
  using vector_type = std::array<ValueType, kn>;

  explicit gradient_transpose_kernel(const rbf::rbf_base& rbf) : rbf_(rbf.clone()) {}

  gradient_transpose_kernel(const gradient_kernel& other) : rbf_(other.rbf_->clone()) {}

  const std::string name() const { return std::string(""); }

  template <typename ValueType>
  [[nodiscard]] inline constexpr auto mutual_coefficient() const {
    vector_type<ValueType> mc;
    std::fill(std::begin(mc), std::end(mc), ValueType(1.0));
    return mc;
  }

  [[nodiscard]] inline auto evaluate(scalfmm::container::point<double, 3> const& x,
                                     scalfmm::container::point<double, 3> const& y) const noexcept {
    geometry::point3d xx{x.at(0), x.at(1), x.at(2)};
    geometry::point3d yy{y.at(0), y.at(1), y.at(2)};

    auto aniso = rbf_->anisotropy();
    geometry::matrix3d h = aniso.transpose() * rbf_->evaluate_hessian_isotropic(xx - yy) * aniso;

    matrix_type<double> result;
    for (index_t i = 0; i < Dim; i++) {
      for (index_t j = 0; j < Dim; j++) {
        result.at(Dim * i + j) = h(i, j);
      }
    }

    return result;
  }

  [[nodiscard]] inline auto evaluate(
      scalfmm::container::point<xsimd::batch<double>, 3> const& x,
      scalfmm::container::point<xsimd::batch<double>, 3> const& y) const noexcept {
    using decayed_type = typename std::decay_t<xsimd::batch<double>>;
    auto n = x.at(0).size;
    std::array<double, 4> v00;
    std::array<double, 4> v01;
    std::array<double, 4> v02;
    std::array<double, 4> v11;
    std::array<double, 4> v12;
    std::array<double, 4> v22;

    auto aniso = rbf_->anisotropy();

    for (size_t i = 0; i < n; i++) {
      geometry::point3d xx{x.at(0).get(i), x.at(1).get(i), x.at(2).get(i)};
      geometry::point3d yy{y.at(0).get(i), y.at(1).get(i), y.at(2).get(i)};
      geometry::vector3d h = aniso.transpose() * rbf_->evaluate_hessian_isotropic(xx - yy) * aniso;
      v00.at(i) = h(0, 0);
      v01.at(i) = h(0, 1);
      v02.at(i) = h(0, 2);
      v11.at(i) = h(1, 1);
      v12.at(i) = h(1, 2);
      v22.at(i) = h(2, 2);
    }

    matrix_type<decayed_type> result;
    result.at(Dim * 0 + 0) = decayed_type::load(v0.data(), xsimd::unaligned_mode{});
    if (Dim > 1) {
      result.at(Dim * 0 + 1) = decayed_type::load(v01.data(), xsimd::unaligned_mode{});
      result.at(Dim * 1 + 0) = decayed_type::load(v01.data(), xsimd::unaligned_mode{});
      result.at(Dim * 1 + 1) = decayed_type::load(v11.data(), xsimd::unaligned_mode{});
    }
    if (Dim > 2) {
      result.at(Dim * 0 + 2) = decayed_type::load(v02.data(), xsimd::unaligned_mode{});
      result.at(Dim * 1 + 2) = decayed_type::load(v12.data(), xsimd::unaligned_mode{});
      result.at(Dim * 2 + 0) = decayed_type::load(v02.data(), xsimd::unaligned_mode{});
      result.at(Dim * 2 + 1) = decayed_type::load(v12.data(), xsimd::unaligned_mode{});
      result.at(Dim * 2 + 2) = decayed_type::load(v22.data(), xsimd::unaligned_mode{});
    }

    return result;
  }

  static constexpr int separation_criterion{1};

 private:
  const std::unique_ptr<rbf::rbf_base> rbf_;
};

}  // namespace fmm
}  // namespace polatory
