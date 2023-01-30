#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/common/eigen_utility.hpp>

using polatory::common::concatenate_cols;
using polatory::common::concatenate_rows;
using polatory::common::take_cols;
using polatory::common::take_rows;

template <bool RowMajor>
void test_take_cols() {
  using Matrix = Eigen::Matrix<double, 2, 6, RowMajor ? Eigen::RowMajor : Eigen::ColMajor>;

  Matrix m = Matrix::Random(2, 6);

  auto m2_fixed = take_cols(m, 1, 3, 4);
  auto m2_dynamic = take_cols(m, {1, 3, 4});

  Eigen::MatrixXd m2_expected(2, 3);
  m2_expected << m.col(1), m.col(3), m.col(4);

  EXPECT_EQ(2u, decltype(m2_fixed)::RowsAtCompileTime);
  EXPECT_EQ(3u, decltype(m2_fixed)::ColsAtCompileTime);
  EXPECT_EQ(RowMajor, decltype(m2_fixed)::IsRowMajor);
  EXPECT_EQ(m2_expected, m2_fixed);

  EXPECT_EQ(2u, decltype(m2_dynamic)::RowsAtCompileTime);
  EXPECT_EQ(Eigen::Dynamic, decltype(m2_dynamic)::ColsAtCompileTime);
  EXPECT_EQ(RowMajor, decltype(m2_dynamic)::IsRowMajor);
  EXPECT_EQ(m2_expected, m2_dynamic);
}

template <bool RowMajor>
void test_take_rows() {
  using Matrix = Eigen::Matrix<double, 6, 2, RowMajor ? Eigen::RowMajor : Eigen::ColMajor>;

  Matrix m = Matrix::Random(6, 2);

  auto m2_fixed = take_rows(m, 1, 3, 4);
  auto m2_dynamic = take_rows(m, {1, 3, 4});

  Eigen::MatrixXd m2_expected(3, 2);
  m2_expected << m.row(1), m.row(3), m.row(4);

  EXPECT_EQ(3u, decltype(m2_fixed)::RowsAtCompileTime);
  EXPECT_EQ(2u, decltype(m2_fixed)::ColsAtCompileTime);
  EXPECT_EQ(RowMajor, decltype(m2_fixed)::IsRowMajor);
  EXPECT_EQ(m2_expected, m2_fixed);

  EXPECT_EQ(Eigen::Dynamic, decltype(m2_dynamic)::RowsAtCompileTime);
  EXPECT_EQ(2u, decltype(m2_dynamic)::ColsAtCompileTime);
  EXPECT_EQ(RowMajor, decltype(m2_dynamic)::IsRowMajor);
  EXPECT_EQ(m2_expected, m2_dynamic);
}

TEST(concatenate_cols, trivial) {
  Eigen::MatrixXd a = Eigen::MatrixXd::Random(3, 1);
  Eigen::MatrixXd b = Eigen::MatrixXd::Random(3, 2);
  Eigen::MatrixXd c = Eigen::MatrixXd::Random(3, 3);

  auto d = concatenate_cols(a, b, c);

  Eigen::MatrixXd d_expected(3, 6);
  d_expected << a, b, c;

  EXPECT_EQ(d_expected, d);
}

TEST(concatenate_rows, trivial) {
  Eigen::MatrixXd a = Eigen::MatrixXd::Random(1, 3);
  Eigen::MatrixXd b = Eigen::MatrixXd::Random(2, 3);
  Eigen::MatrixXd c = Eigen::MatrixXd::Random(3, 3);

  auto d = concatenate_rows(a, b, c);

  Eigen::MatrixXd d_expected(6, 3);
  d_expected << a, b, c;

  EXPECT_EQ(d_expected, d);
}

TEST(take_cols, trivial) {
  test_take_cols<false>();
  test_take_cols<true>();
}

TEST(take_rows, trivial) {
  test_take_rows<false>();
  test_take_rows<true>();
}
