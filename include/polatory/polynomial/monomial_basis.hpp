#pragma once

#include <Eigen/Core>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/types.hpp>

namespace polatory::polynomial {

template <int Dim>
class monomial_basis : public polynomial_basis_base<Dim> {
  using Base = polynomial_basis_base<Dim>;

 public:
  using Base::basis_size;
  using Base::degree;

  static constexpr int kDim = Dim;

  explicit monomial_basis(int degree) : Base(degree) {
    POLATORY_ASSERT(degree >= 0 && degree <= 2);
  }

  template <class DerivedPoints>
  Eigen::MatrixXd evaluate(const Eigen::MatrixBase<DerivedPoints>& points) const {
    return evaluate(points, geometry::points3d(0, 3));
  }

  template <class DerivedPoints, class DerivedGradPoints>
  Eigen::MatrixXd evaluate(const Eigen::MatrixBase<DerivedPoints>& points,
                           const Eigen::MatrixBase<DerivedGradPoints>& grad_points) const {
    auto mu = points.rows();
    auto sigma = grad_points.rows();

    Eigen::MatrixXd result(basis_size(), mu + Dim * sigma);

    switch (kDim) {
      case 1:
        switch (degree()) {
          case 0:
            for (index_t i = 0; i < mu; i++) {
              result(0, i) = 1.0;  // 1
            }
            for (index_t i = 0; i < sigma; i++) {
              auto i_x = mu + i;
              result(0, i_x) = 0.0;  // 1_x
            }
            break;

          case 1:
            for (index_t i = 0; i < mu; i++) {
              auto p = points.row(i);
              result(0, i) = 1.0;   // 1
              result(1, i) = p(0);  // x
            }
            for (index_t i = 0; i < sigma; i++) {
              auto i_x = mu + i;
              result(0, i_x) = 0.0;  // 1_x
              result(1, i_x) = 1.0;  // x_x
            }
            break;

          case 2:
            for (index_t i = 0; i < mu; i++) {
              auto p = points.row(i);
              result(0, i) = 1.0;          // 1
              result(1, i) = p(0);         // x
              result(2, i) = p(0) * p(0);  // x^2
            }
            for (index_t i = 0; i < sigma; i++) {
              auto p = grad_points.row(i);
              auto i_x = mu + i;
              result(0, i_x) = 0.0;         // 1_x
              result(1, i_x) = 1.0;         // x_x
              result(2, i_x) = 2.0 * p(0);  // x^2_x
            }
            break;

          default:
            POLATORY_UNREACHABLE();
            break;
        }
        break;

      case 2:
        switch (degree()) {
          case 0:
            for (index_t i = 0; i < mu; i++) {
              result(0, i) = 1.0;  // 1
            }
            for (index_t i = 0; i < sigma; i++) {
              auto i_x = mu + 2 * i;
              auto i_y = mu + 2 * i + 1;
              result(0, i_x) = 0.0;  // 1_x
              result(0, i_y) = 0.0;  // 1_y
            }
            break;

          case 1:
            for (index_t i = 0; i < mu; i++) {
              auto p = points.row(i);
              result(0, i) = 1.0;   // 1
              result(1, i) = p(0);  // x
              result(2, i) = p(1);  // y
            }
            for (index_t i = 0; i < sigma; i++) {
              auto i_x = mu + 2 * i;
              auto i_y = mu + 2 * i + 1;
              result(0, i_x) = 0.0;  // 1_x
              result(0, i_y) = 0.0;  // 1_y
              result(1, i_x) = 1.0;  // x_x
              result(1, i_y) = 0.0;  // x_y
              result(2, i_x) = 0.0;  // y_x
              result(2, i_y) = 1.0;  // y_y
            }
            break;

          case 2:
            for (index_t i = 0; i < mu; i++) {
              auto p = points.row(i);
              result(0, i) = 1.0;          // 1
              result(1, i) = p(0);         // x
              result(2, i) = p(1);         // y
              result(3, i) = p(0) * p(0);  // x^2
              result(4, i) = p(0) * p(1);  // xy
              result(5, i) = p(1) * p(1);  // y^2
            }
            for (index_t i = 0; i < sigma; i++) {
              auto p = grad_points.row(i);
              auto i_x = mu + 2 * i;
              auto i_y = mu + 2 * i + 1;
              result(0, i_x) = 0.0;         // 1_x
              result(0, i_y) = 0.0;         // 1_y
              result(1, i_x) = 1.0;         // x_x
              result(1, i_y) = 0.0;         // x_y
              result(2, i_x) = 0.0;         // y_x
              result(2, i_y) = 1.0;         // y_y
              result(3, i_x) = 2.0 * p(0);  // x^2_x
              result(3, i_y) = 0.0;         // x^2_y
              result(4, i_x) = p(1);        // xy_x
              result(4, i_y) = p(0);        // xy_y
              result(5, i_x) = 0.0;         // y^2_x
              result(5, i_y) = 2.0 * p(1);  // y^2_y
            }
            break;

          default:
            POLATORY_UNREACHABLE();
            break;
        }
        break;

      case 3:
        switch (degree()) {
          case 0:
            for (index_t i = 0; i < mu; i++) {
              result(0, i) = 1.0;  // 1
            }
            for (index_t i = 0; i < sigma; i++) {
              auto i_x = mu + 3 * i;
              auto i_y = mu + 3 * i + 1;
              auto i_z = mu + 3 * i + 2;
              result(0, i_x) = 0.0;  // 1_x
              result(0, i_y) = 0.0;  // 1_y
              result(0, i_z) = 0.0;  // 1_z
            }
            break;

          case 1:
            for (index_t i = 0; i < mu; i++) {
              auto p = points.row(i);
              result(0, i) = 1.0;   // 1
              result(1, i) = p(0);  // x
              result(2, i) = p(1);  // y
              result(3, i) = p(2);  // z
            }
            for (index_t i = 0; i < sigma; i++) {
              auto i_x = mu + 3 * i;
              auto i_y = mu + 3 * i + 1;
              auto i_z = mu + 3 * i + 2;
              result(0, i_x) = 0.0;  // 1_x
              result(0, i_y) = 0.0;  // 1_y
              result(0, i_z) = 0.0;  // 1_z
              result(1, i_x) = 1.0;  // x_x
              result(1, i_y) = 0.0;  // x_y
              result(1, i_z) = 0.0;  // x_z
              result(2, i_x) = 0.0;  // y_x
              result(2, i_y) = 1.0;  // y_y
              result(2, i_z) = 0.0;  // y_z
              result(3, i_x) = 0.0;  // z_x
              result(3, i_y) = 0.0;  // z_y
              result(3, i_z) = 1.0;  // z_z
            }
            break;

          case 2:
            for (index_t i = 0; i < mu; i++) {
              auto p = points.row(i);
              result(0, i) = 1.0;          // 1
              result(1, i) = p(0);         // x
              result(2, i) = p(1);         // y
              result(3, i) = p(2);         // z
              result(4, i) = p(0) * p(0);  // x^2
              result(5, i) = p(0) * p(1);  // xy
              result(6, i) = p(0) * p(2);  // xz
              result(7, i) = p(1) * p(1);  // y^2
              result(8, i) = p(1) * p(2);  // yz
              result(9, i) = p(2) * p(2);  // z^2
            }
            for (index_t i = 0; i < sigma; i++) {
              auto p = grad_points.row(i);
              auto i_x = mu + 3 * i;
              auto i_y = mu + 3 * i + 1;
              auto i_z = mu + 3 * i + 2;
              result(0, i_x) = 0.0;         // 1_x
              result(0, i_y) = 0.0;         // 1_y
              result(0, i_z) = 0.0;         // 1_z
              result(1, i_x) = 1.0;         // x_x
              result(1, i_y) = 0.0;         // x_y
              result(1, i_z) = 0.0;         // x_z
              result(2, i_x) = 0.0;         // y_x
              result(2, i_y) = 1.0;         // y_y
              result(2, i_z) = 0.0;         // y_z
              result(3, i_x) = 0.0;         // z_x
              result(3, i_y) = 0.0;         // z_y
              result(3, i_z) = 1.0;         // z_z
              result(4, i_x) = 2.0 * p(0);  // x^2_x
              result(4, i_y) = 0.0;         // x^2_y
              result(4, i_z) = 0.0;         // x^2_z
              result(5, i_x) = p(1);        // xy_x
              result(5, i_y) = p(0);        // xy_y
              result(5, i_z) = 0.0;         // xy_z
              result(6, i_x) = p(2);        // xz_x
              result(6, i_y) = 0.0;         // xz_y
              result(6, i_z) = p(0);        // xz_z
              result(7, i_x) = 0.0;         // y^2_x
              result(7, i_y) = 2.0 * p(1);  // y^2_y
              result(7, i_z) = 0.0;         // y^2_z
              result(8, i_x) = 0.0;         // yz_x
              result(8, i_y) = p(2);        // yz_y
              result(8, i_z) = p(1);        // yz_z
              result(9, i_x) = 0.0;         // z^2_x
              result(9, i_y) = 0.0;         // z^2_y
              result(9, i_z) = 2.0 * p(2);  // z^2_z
            }
            break;

          default:
            POLATORY_UNREACHABLE();
            break;
        }
        break;

      default:
        POLATORY_UNREACHABLE();
        break;
    }

    return result;
  }
};

}  // namespace polatory::polynomial
