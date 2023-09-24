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

struct fmm_rbf_hessian_kernel {
  static constexpr auto homogeneity_tag{scalfmm::matrix_kernels::homogeneity::non_homogenous};
  static constexpr auto symmetry_tag{scalfmm::matrix_kernels::symmetry::non_symmetric};
  static constexpr std::size_t km{1};
  static constexpr std::size_t kn{6};
  template <typename ValueType>
  using matrix_type = std::array<ValueType, kn * km>;
  template <typename ValueType>
  using vector_type = std::array<ValueType, kn>;

  explicit fmm_rbf_hessian_kernel(const rbf::rbf_base& rbf) : rbf_(rbf.clone()) {}

  fmm_rbf_hessian_kernel(const fmm_rbf_hessian_kernel& other) : rbf_(other.rbf_->clone()) {}

  const std::string name() const { return std::string(""); }

  template <typename ValueType>
  [[nodiscard]] inline constexpr auto mutual_coefficient() const {
    using decayed_type = typename std::decay_t<ValueType>;
    return vector_type<decayed_type>{decayed_type(1.0)};
  }

  [[nodiscard]] inline auto evaluate(scalfmm::container::point<double, 3> const& x,
                                     scalfmm::container::point<double, 3> const& y) const noexcept {
    geometry::point3d xx{x.at(0), x.at(1), x.at(2)};
    geometry::point3d yy{y.at(0), y.at(1), y.at(2)};

    auto h = rbf_->evaluate_hessian_isotropic(xx - yy);

    return matrix_type<double>{h(0, 0), h(0, 1), h(0, 2), h(1, 1), h(1, 2), h(2, 2)};
  }

  [[nodiscard]] inline auto evaluate(
      scalfmm::container::point<xsimd::batch<double>, 3> const& x,
      scalfmm::container::point<xsimd::batch<double>, 3> const& y) const noexcept {
    using decayed_type = typename std::decay_t<xsimd::batch<double>>;
    auto n = x.at(0).size;
    std::array<double, 4> v0;
    std::array<double, 4> v1;
    std::array<double, 4> v2;
    std::array<double, 4> v3;
    std::array<double, 4> v4;
    std::array<double, 4> v5;

    for (size_t i = 0; i < n; i++) {
      geometry::point3d xx{x.at(0).get(i), x.at(1).get(i), x.at(2).get(i)};
      geometry::point3d yy{y.at(0).get(i), y.at(1).get(i), y.at(2).get(i)};
      auto h = rbf_->evaluate_hessian_isotropic(xx - yy);
      v0.at(i) = h(0, 0);
      v1.at(i) = h(0, 1);
      v2.at(i) = h(0, 2);
      v3.at(i) = h(1, 1);
      v4.at(i) = h(1, 2);
      v5.at(i) = h(2, 2);
    }

    return matrix_type<decayed_type>{decayed_type::load(v0.data(), xsimd::unaligned_mode{}),
                                     decayed_type::load(v1.data(), xsimd::unaligned_mode{}),
                                     decayed_type::load(v2.data(), xsimd::unaligned_mode{}),
                                     decayed_type::load(v3.data(), xsimd::unaligned_mode{}),
                                     decayed_type::load(v4.data(), xsimd::unaligned_mode{}),
                                     decayed_type::load(v5.data(), xsimd::unaligned_mode{})};
  }

  static constexpr int separation_criterion{1};

 private:
  const std::unique_ptr<rbf::rbf_base> rbf_;
};

}  // namespace fmm
}  // namespace polatory
