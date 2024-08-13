#pragma once

#include <Eigen/Core>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/types.hpp>

namespace polatory::polynomial {

template <int Dim>
class MonomialBasis : public PolynomialBasisBase<Dim> {
 public:
  static constexpr int kDim = Dim;

 private:
  using Base = PolynomialBasisBase<kDim>;
  using Points = geometry::Points<kDim>;

 public:
  using Base::basis_size;
  using Base::degree;

  explicit MonomialBasis(int degree) : Base(degree) { POLATORY_ASSERT(degree >= 0 && degree <= 2); }

  template <class DerivedPoints>
  MatX evaluate(const Eigen::MatrixBase<DerivedPoints>& points) const {
    return evaluate(points, Points(0, Dim));
  }

  template <class DerivedPoints, class DerivedGradPoints>
  MatX evaluate(const Eigen::MatrixBase<DerivedPoints>& points,
                const Eigen::MatrixBase<DerivedGradPoints>& grad_points) const {
    auto mu = points.rows();
    auto sigma = grad_points.rows();

    MatX result(mu + Dim * sigma, basis_size());

    switch (kDim) {
      case 1:
        switch (degree()) {
          case 0:
            for (Index i = 0; i < mu; i++) {
              result(i, 0) = 1.0;  // 1
            }
            for (Index i = 0; i < sigma; i++) {
              auto i_x = mu + i;
              result(i_x, 0) = 0.0;  // 1_x
            }
            break;

          case 1:
            for (Index i = 0; i < mu; i++) {
              auto p = points.row(i);
              result(i, 0) = 1.0;   // 1
              result(i, 1) = p(0);  // x
            }
            for (Index i = 0; i < sigma; i++) {
              auto i_x = mu + i;
              result(i_x, 0) = 0.0;  // 1_x
              result(i_x, 1) = 1.0;  // x_x
            }
            break;

          case 2:
            for (Index i = 0; i < mu; i++) {
              auto p = points.row(i);
              result(i, 0) = 1.0;          // 1
              result(i, 1) = p(0);         // x
              result(i, 2) = p(0) * p(0);  // x^2
            }
            for (Index i = 0; i < sigma; i++) {
              auto p = grad_points.row(i);
              auto i_x = mu + i;
              result(i_x, 0) = 0.0;         // 1_x
              result(i_x, 1) = 1.0;         // x_x
              result(i_x, 2) = 2.0 * p(0);  // x^2_x
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
            for (Index i = 0; i < mu; i++) {
              result(i, 0) = 1.0;  // 1
            }
            for (Index i = 0; i < sigma; i++) {
              auto i_x = mu + 2 * i;
              auto i_y = mu + 2 * i + 1;
              result(i_x, 0) = 0.0;  // 1_x
              result(i_y, 0) = 0.0;  // 1_y
            }
            break;

          case 1:
            for (Index i = 0; i < mu; i++) {
              auto p = points.row(i);
              result(i, 0) = 1.0;   // 1
              result(i, 1) = p(0);  // x
              result(i, 2) = p(1);  // y
            }
            for (Index i = 0; i < sigma; i++) {
              auto i_x = mu + 2 * i;
              auto i_y = mu + 2 * i + 1;
              result(i_x, 0) = 0.0;  // 1_x
              result(i_y, 0) = 0.0;  // 1_y
              result(i_x, 1) = 1.0;  // x_x
              result(i_y, 1) = 0.0;  // x_y
              result(i_x, 2) = 0.0;  // y_x
              result(i_y, 2) = 1.0;  // y_y
            }
            break;

          case 2:
            for (Index i = 0; i < mu; i++) {
              auto p = points.row(i);
              result(i, 0) = 1.0;          // 1
              result(i, 1) = p(0);         // x
              result(i, 2) = p(1);         // y
              result(i, 3) = p(0) * p(0);  // x^2
              result(i, 4) = p(0) * p(1);  // xy
              result(i, 5) = p(1) * p(1);  // y^2
            }
            for (Index i = 0; i < sigma; i++) {
              auto p = grad_points.row(i);
              auto i_x = mu + 2 * i;
              auto i_y = mu + 2 * i + 1;
              result(i_x, 0) = 0.0;         // 1_x
              result(i_y, 0) = 0.0;         // 1_y
              result(i_x, 1) = 1.0;         // x_x
              result(i_y, 1) = 0.0;         // x_y
              result(i_x, 2) = 0.0;         // y_x
              result(i_y, 2) = 1.0;         // y_y
              result(i_x, 3) = 2.0 * p(0);  // x^2_x
              result(i_y, 3) = 0.0;         // x^2_y
              result(i_x, 4) = p(1);        // xy_x
              result(i_y, 4) = p(0);        // xy_y
              result(i_x, 5) = 0.0;         // y^2_x
              result(i_y, 5) = 2.0 * p(1);  // y^2_y
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
            for (Index i = 0; i < mu; i++) {
              result(i, 0) = 1.0;  // 1
            }
            for (Index i = 0; i < sigma; i++) {
              auto i_x = mu + 3 * i;
              auto i_y = mu + 3 * i + 1;
              auto i_z = mu + 3 * i + 2;
              result(i_x, 0) = 0.0;  // 1_x
              result(i_y, 0) = 0.0;  // 1_y
              result(i_z, 0) = 0.0;  // 1_z
            }
            break;

          case 1:
            for (Index i = 0; i < mu; i++) {
              auto p = points.row(i);
              result(i, 0) = 1.0;   // 1
              result(i, 1) = p(0);  // x
              result(i, 2) = p(1);  // y
              result(i, 3) = p(2);  // z
            }
            for (Index i = 0; i < sigma; i++) {
              auto i_x = mu + 3 * i;
              auto i_y = mu + 3 * i + 1;
              auto i_z = mu + 3 * i + 2;
              result(i_x, 0) = 0.0;  // 1_x
              result(i_y, 0) = 0.0;  // 1_y
              result(i_z, 0) = 0.0;  // 1_z
              result(i_x, 1) = 1.0;  // x_x
              result(i_y, 1) = 0.0;  // x_y
              result(i_z, 1) = 0.0;  // x_z
              result(i_x, 2) = 0.0;  // y_x
              result(i_y, 2) = 1.0;  // y_y
              result(i_z, 2) = 0.0;  // y_z
              result(i_x, 3) = 0.0;  // z_x
              result(i_y, 3) = 0.0;  // z_y
              result(i_z, 3) = 1.0;  // z_z
            }
            break;

          case 2:
            for (Index i = 0; i < mu; i++) {
              auto p = points.row(i);
              result(i, 0) = 1.0;          // 1
              result(i, 1) = p(0);         // x
              result(i, 2) = p(1);         // y
              result(i, 3) = p(2);         // z
              result(i, 4) = p(0) * p(0);  // x^2
              result(i, 5) = p(0) * p(1);  // xy
              result(i, 6) = p(0) * p(2);  // xz
              result(i, 7) = p(1) * p(1);  // y^2
              result(i, 8) = p(1) * p(2);  // yz
              result(i, 9) = p(2) * p(2);  // z^2
            }
            for (Index i = 0; i < sigma; i++) {
              auto p = grad_points.row(i);
              auto i_x = mu + 3 * i;
              auto i_y = mu + 3 * i + 1;
              auto i_z = mu + 3 * i + 2;
              result(i_x, 0) = 0.0;         // 1_x
              result(i_y, 0) = 0.0;         // 1_y
              result(i_z, 0) = 0.0;         // 1_z
              result(i_x, 1) = 1.0;         // x_x
              result(i_y, 1) = 0.0;         // x_y
              result(i_z, 1) = 0.0;         // x_z
              result(i_x, 2) = 0.0;         // y_x
              result(i_y, 2) = 1.0;         // y_y
              result(i_z, 2) = 0.0;         // y_z
              result(i_x, 3) = 0.0;         // z_x
              result(i_y, 3) = 0.0;         // z_y
              result(i_z, 3) = 1.0;         // z_z
              result(i_x, 4) = 2.0 * p(0);  // x^2_x
              result(i_y, 4) = 0.0;         // x^2_y
              result(i_z, 4) = 0.0;         // x^2_z
              result(i_x, 5) = p(1);        // xy_x
              result(i_y, 5) = p(0);        // xy_y
              result(i_z, 5) = 0.0;         // xy_z
              result(i_x, 6) = p(2);        // xz_x
              result(i_y, 6) = 0.0;         // xz_y
              result(i_z, 6) = p(0);        // xz_z
              result(i_x, 7) = 0.0;         // y^2_x
              result(i_y, 7) = 2.0 * p(1);  // y^2_y
              result(i_z, 7) = 0.0;         // y^2_z
              result(i_x, 8) = 0.0;         // yz_x
              result(i_y, 8) = p(2);        // yz_y
              result(i_z, 8) = p(1);        // yz_z
              result(i_x, 9) = 0.0;         // z^2_x
              result(i_y, 9) = 0.0;         // z^2_y
              result(i_z, 9) = 2.0 * p(2);  // z^2_z
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
