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

struct fmm_rbf_kernel {
  static constexpr auto homogeneity_tag{scalfmm::matrix_kernels::homogeneity::non_homogenous};
  static constexpr auto symmetry_tag{scalfmm::matrix_kernels::symmetry::symmetric};
  static constexpr std::size_t km{1};
  static constexpr std::size_t kn{1};
  template <typename ValueType>
  using matrix_type = std::array<ValueType, kn * km>;
  template <typename ValueType>
  using vector_type = std::array<ValueType, kn>;

  explicit fmm_rbf_kernel(const rbf::rbf_base& rbf) : rbf_(rbf.clone()) {}

  fmm_rbf_kernel(const fmm_rbf_kernel& other) : rbf_(other.rbf_->clone()) {}

  const std::string name() const { return std::string(""); }

  template <typename ValueType>
  [[nodiscard]] inline constexpr auto mutual_coefficient() const {
    using decayed_type = typename std::decay_t<ValueType>;
    return vector_type<decayed_type>{decayed_type(1.)};
  }

  [[nodiscard]] inline auto evaluate(scalfmm::container::point<double, 3> const& x,
                                     scalfmm::container::point<double, 3> const& y) const noexcept {
    geometry::point3d xx{x.at(0), x.at(1), x.at(2)};
    geometry::point3d yy{y.at(0), y.at(1), y.at(2)};

    return matrix_type<double>{rbf_->evaluate_isotropic(xx - yy)};
  }

  [[nodiscard]] inline auto evaluate(
      scalfmm::container::point<xsimd::batch<double>, 3> const& x,
      scalfmm::container::point<xsimd::batch<double>, 3> const& y) const noexcept {
    using decayed_type = typename std::decay_t<xsimd::batch<double>>;
    auto n = x.at(0).size;
    std::vector<double> v(n);

    for (size_t i = 0; i < n; i++) {
      geometry::point3d xx{x.at(0).get(i), x.at(1).get(i), x.at(2).get(i)};
      geometry::point3d yy{y.at(0).get(i), y.at(1).get(i), y.at(2).get(i)};
      v.at(i) = rbf_->evaluate_isotropic(xx - yy);
    }

    return matrix_type<decayed_type>{decayed_type::load(v.data(), xsimd::unaligned_mode{})};
  }

  static constexpr int separation_criterion{1};

 private:
  const std::unique_ptr<rbf::rbf_base> rbf_;
};

struct one_over_r {
  static constexpr auto homogeneity_tag{scalfmm::matrix_kernels::homogeneity::non_homogenous};
  static constexpr auto symmetry_tag{scalfmm::matrix_kernels::symmetry::symmetric};
  static constexpr std::size_t km{1};
  static constexpr std::size_t kn{1};
  template <typename ValueType>
  using matrix_type = std::array<ValueType, kn * km>;
  template <typename ValueType>
  using vector_type = std::array<ValueType, kn>;

  const std::string name() const { return std::string("one_over_r"); }

  template <typename ValueType>
  [[nodiscard]] inline constexpr auto mutual_coefficient() const {
    using decayed_type = typename std::decay_t<ValueType>;
    return vector_type<decayed_type>{decayed_type(1.)};
  }

  template <typename ValueType>
  [[nodiscard]] inline auto evaluate(
      scalfmm::container::point<ValueType, 3> const& x,
      scalfmm::container::point<ValueType, 3> const& y) const noexcept {
    return variadic_evaluate(x, y, std::make_index_sequence<3>{});
  }

  template <typename ValueType, std::size_t Dim, std::size_t... Is>
  [[nodiscard]] inline auto variadic_evaluate(scalfmm::container::point<ValueType, Dim> const& xs,
                                              scalfmm::container::point<ValueType, Dim> const& ys,
                                              std::index_sequence<Is...>) const noexcept {
    using decayed_type = typename std::decay_t<ValueType>;
    return matrix_type<decayed_type>{
        decayed_type(1.0) /
        xsimd::sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...))};
  }

  static constexpr int separation_criterion{1};
};

}  // namespace fmm
}  // namespace polatory
