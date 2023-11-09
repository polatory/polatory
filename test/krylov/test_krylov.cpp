#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <memory>
#include <polatory/krylov/fgmres.hpp>
#include <polatory/krylov/gmres.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/krylov/minres.hpp>
#include <polatory/types.hpp>

using polatory::index_t;
using polatory::common::valuesd;
using polatory::krylov::fgmres;
using polatory::krylov::gmres;
using polatory::krylov::linear_operator;
using polatory::krylov::minres;

namespace {

class random_symmetric : public linear_operator {
 public:
  explicit random_symmetric(index_t n) : n_(n) {
    m_ = (Eigen::MatrixXd::Random(n, n) + Eigen::MatrixXd::Ones(n, n)) / 2.0;
    for (index_t i = 1; i < n; i++) {
      for (index_t j = 0; j < i; j++) {
        m_(i, j) = m_(j, i);
      }
    }
  }

  const Eigen::MatrixXd& matrix() const { return m_; }

  valuesd operator()(const valuesd& v) const override { return m_ * v; }

  index_t size() const override { return n_; }

 private:
  const index_t n_;
  Eigen::MatrixXd m_;
};

class preconditioner : public linear_operator {
 public:
  explicit preconditioner(const random_symmetric& op) : n_(op.size()) {
    Eigen::MatrixXd perturbation = 0.1 * valuesd::Random(n_).asDiagonal();
    m_ = op.matrix().inverse() + perturbation;
  }

  valuesd operator()(const valuesd& v) const override { return m_ * v; }

  index_t size() const override { return n_; }

 private:
  const index_t n_;
  Eigen::MatrixXd m_;
};

class krylov_test : public ::testing::Test {
 protected:
  static constexpr index_t n = 100;

  std::unique_ptr<random_symmetric> op;
  std::unique_ptr<preconditioner> pc;

  valuesd solution;
  valuesd rhs;

  valuesd x0;

  void SetUp() override {
    op = std::make_unique<random_symmetric>(n);
    pc = std::make_unique<preconditioner>(*op);

    solution = (valuesd::Random(n) + valuesd::Ones(n)) / 2.0;
    rhs = (*op)(solution);

    x0 = valuesd::Random(n);
  }

  void TearDown() override {}

  template <class Solver>
  void test_solver(bool with_initial_solution, bool with_right_pc, bool with_left_pc) {
    Solver solver(*op, rhs, n);
    if (with_initial_solution) solver.set_initial_solution(x0);
    if (with_right_pc) solver.set_right_preconditioner(*pc);
    if (with_left_pc) solver.set_left_preconditioner(*pc);
    solver.setup();

    auto last_residual = 0.0;
    for (index_t i = 0; i < solver.max_iterations(); i++) {
      solver.iterate_process();
      auto approx_solution = solver.solution_vector();
      auto current_residual = solver.relative_residual();

      if (!with_left_pc) {
        EXPECT_NEAR((rhs - (*op)(approx_solution)).norm() / rhs.norm(), current_residual, 1e-12);
      }

      if (i > 0) {
        EXPECT_LT(current_residual, last_residual);
      }

      last_residual = current_residual;
    }
  }
};

constexpr index_t krylov_test::n;

}  // namespace

TEST_F(krylov_test, fgmres) {
  test_solver<fgmres>(false, false, false);
  test_solver<fgmres>(false, true, false);
  test_solver<fgmres>(true, false, false);
  test_solver<fgmres>(true, true, false);
}

TEST_F(krylov_test, gmres) {
  test_solver<gmres>(false, false, false);
  test_solver<gmres>(false, true, false);
  test_solver<gmres>(false, false, true);
  test_solver<gmres>(true, false, false);
  test_solver<gmres>(true, true, false);
  test_solver<gmres>(true, false, true);
}

TEST_F(krylov_test, minres) {
  test_solver<minres>(false, false, false);
  test_solver<minres>(true, false, false);
}
