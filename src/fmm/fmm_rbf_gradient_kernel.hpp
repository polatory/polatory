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
struct fmm_rbf_gradient_kernel {
  static constexpr auto homogeneity_tag{scalfmm::matrix_kernels::homogeneity::non_homogenous};
  static constexpr auto symmetry_tag{scalfmm::matrix_kernels::symmetry::non_symmetric};
  static constexpr std::size_t km{Dim};
  static constexpr std::size_t kn{1};
  template <typename ValueType>
  using matrix_type = std::array<ValueType, kn * km>;
  template <typename ValueType>
  using vector_type = std::array<ValueType, kn>;

  explicit fmm_rbf_gradient_kernel(const rbf::rbf_base& rbf) : rbf_(rbf.clone()) {}

  fmm_rbf_gradient_kernel(const fmm_rbf_gradient_kernel& other) : rbf_(other.rbf_->clone()) {}

  const std::string name() const { return std::string(""); }

  template <typename ValueType>
  [[nodiscard]] inline constexpr auto mutual_coefficient() const {
    using decayed_type = typename std::decay_t<ValueType>;
    return vector_type<decayed_type>{decayed_type(-1.0)};
  }

  [[nodiscard]] inline auto evaluate(scalfmm::container::point<double, 3> const& x,
                                     scalfmm::container::point<double, 3> const& y) const noexcept {
    geometry::point3d xx{x.at(0), x.at(1), x.at(2)};
    geometry::point3d yy{y.at(0), y.at(1), y.at(2)};

    geometry::vector3d g = rbf_->evaluate_gradient_isotropic(xx - yy) * rbf_->anisotropy();

    matrix_type<double> result;
    for (index_t i = 0; i < Dim; i++) {
      result.at(i) = -g(i);
    }

    return result;
  }

  [[nodiscard]] inline auto evaluate(
      scalfmm::container::point<xsimd::batch<double>, 3> const& x,
      scalfmm::container::point<xsimd::batch<double>, 3> const& y) const noexcept {
    using decayed_type = typename std::decay_t<xsimd::batch<double>>;
    auto n = x.at(0).size;
    std::array<double, 4> v0;
    std::array<double, 4> v1;
    std::array<double, 4> v2;

    for (size_t i = 0; i < n; i++) {
      geometry::point3d xx{x.at(0).get(i), x.at(1).get(i), x.at(2).get(i)};
      geometry::point3d yy{y.at(0).get(i), y.at(1).get(i), y.at(2).get(i)};
      geometry::vector3d g = rbf_->evaluate_gradient_isotropic(xx - yy) * rbf_->anisotropy();
      v0.at(i) = -g(0);
      v1.at(i) = -g(1);
      v2.at(i) = -g(2);
    }

    matrix_type<decayed_type> result;
    result.at(0) = decayed_type::load(v0.data(), xsimd::unaligned_mode{});
    if (Dim > 1) {
      result.at(1) = decayed_type::load(v1.data(), xsimd::unaligned_mode{});
    }
    if (Dim > 2) {
      result.at(2) = decayed_type::load(v2.data(), xsimd::unaligned_mode{});
    }

    return result;
  }

  static constexpr int separation_criterion{1};

 private:
  const std::unique_ptr<rbf::rbf_base> rbf_;
};

}  // namespace fmm
}  // namespace polatory
