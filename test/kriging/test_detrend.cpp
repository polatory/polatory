#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/detrend.hpp>
#include <polatory/types.hpp>

using polatory::Index;
using polatory::VecX;
using polatory::geometry::Points2;
using polatory::kriging::detrend;

TEST(detrend, trivial) {
  const Index n_points{10000};

  Points2 points = Points2::Random(n_points, 2);
  VecX values = VecX::Zero(n_points);

  for (Index i = 0; i < n_points; ++i) {
    auto x = points(i, 0);
    auto y = points(i, 1);
    values(i) = 1.0 + x + y + x * x + x * y + y * y;
  }

  // When n_points is large enough, the coefficients of the detrending polynomial can be found by:
  //
  //           /
  //   arg min |   (P(x, y) - Q(x, y))^2 dx dy,
  //   a, b, c / A
  //
  // where A = [-1, 1] x [-1, 1], P(x, y) = 1 + x + y + x^2 + x y + y^2,
  // and Q(x, y) = a + b x + c y, thus a = 5/3, b = 1, and c = 1.

  auto detrended = detrend(points, values, 1);

  // When n_points is large enough, the normalized moments are:
  //
  //           1  /
  //   m_ij = --- |   x^i y^j (P(x, y) - Q(x, y)) dx dy.
  //          |A| / A

  VecX moments = VecX::Zero(6);
  VecX m = VecX(6);
  for (Index i = 0; i < n_points; ++i) {
    auto x = points(i, 0);
    auto y = points(i, 1);
    m << 1.0, x, y, x * x, x * y, y * y;
    moments += m * detrended(i);
  }
  moments /= static_cast<double>(n_points);

  ASSERT_NEAR(moments(0), 0.0, 1e-12);
  ASSERT_NEAR(moments(1), 0.0, 1e-12);
  ASSERT_NEAR(moments(2), 0.0, 1e-12);
  ASSERT_NEAR(moments(3), 4.0 / 45.0, 1e-2);
  ASSERT_NEAR(moments(4), 1.0 / 9.0, 1e-2);
  ASSERT_NEAR(moments(5), 4.0 / 45.0, 1e-2);
}
