#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>
#include <numeric>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::preconditioner {

template <int Dim>
class MatQ {
  static constexpr int kDim = Dim;
  using Bbox = geometry::Bbox<kDim>;
  using MonomialBasis = polynomial::MonomialBasis<kDim>;
  using Point = geometry::Point<kDim>;
  using Points = geometry::Points<kDim>;
  using Vector = geometry::Vector<kDim>;

  static constexpr double kRankThreshold = 1e-10;

 public:
  MatQ(int poly_degree, const Points& points, const Points& grad_points) {
    auto mu = points.rows();
    auto sigma = grad_points.rows();
    auto m = mu + kDim * sigma;

    indices_.resize(m);
    std::iota(indices_.begin(), indices_.end(), Index{0});

    top_ = MatX(0, m);

    if (poly_degree < 0) {
      return;
    }

    auto bbox = Bbox::from_points(points).convex_hull(Bbox::from_points(grad_points));
    Point center = bbox.center();
    Vector radius = bbox.width() / 2.0;
    radius = (radius.array() > 0.0).select(radius, Vector::Ones());

    Points scaled_points = (points.rowwise() - center).array().rowwise() / radius.array();
    Points scaled_grad_points = (grad_points.rowwise() - center).array().rowwise() / radius.array();
    MatX p = MonomialBasis(poly_degree).evaluate(scaled_points, scaled_grad_points);

    Eigen::ColPivHouseholderQR<MatX> col_qr(p);
    const auto& col_perm = col_qr.colsPermutation();

    col_qr.setThreshold(kRankThreshold);
    rank_ = col_qr.rank();
    if (rank_ == 0) {
      return;
    }

    Eigen::ColPivHouseholderQR<MatX> row_qr(p.transpose());
    const auto& row_perm = row_qr.colsPermutation();

    VecX grad_row_scale = radius.transpose().replicate(sigma, 1);
    p.bottomRows(kDim * sigma).array().colwise() /= grad_row_scale.array();

    p = p(row_perm.indices(), col_perm.indices()).eval();
    top_ = -p.topLeftCorner(rank_, rank_)
                .transpose()
                .partialPivLu()
                .solve(p.bottomLeftCorner(m - rank_, rank_).transpose());

    indices_.assign(row_perm.indices().data(), row_perm.indices().data() + m);
  }

  const std::vector<Index>& indices() const { return indices_; }

  Index rank() const { return rank_; }

  const MatX& top() const { return top_; }

 private:
  std::vector<Index> indices_;
  Index rank_{};
  MatX top_;
};

}  // namespace polatory::preconditioner
