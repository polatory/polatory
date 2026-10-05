#include <gtest/gtest.h>

#include <Eigen/LU>
#include <jizai/krylov/fgmres.hpp>
#include <jizai/krylov/gmres.hpp>
#include <jizai/krylov/linear_operator.hpp>
#include <jizai/krylov/minres.hpp>
#include <jizai/numeric/error.hpp>
#include <jizai/types.hpp>
#include <memory>

using jizai::Index;
using jizai::MatX;
using jizai::VecX;
using jizai::krylov::Fgmres;
using jizai::krylov::Gmres;
using jizai::krylov::LinearOperator;
using jizai::krylov::Minres;
using jizai::numeric::relative_error;

namespace {

class RandomSymmetric : public LinearOperator {
 public:
  explicit RandomSymmetric(Index n) : n_(n) {
    m_ = (MatX::Random(n, n) + MatX::Ones(n, n)) / 2.0;
    for (Index i = 1; i < n; i++) {
      for (Index j = 0; j < i; j++) {
        m_(i, j) = m_(j, i);
      }
    }
  }

  const MatX& matrix() const { return m_; }

  VecX operator()(const VecX& v) const override { return m_ * v; }

  Index size() const override { return n_; }

 private:
  const Index n_;
  MatX m_;
};

class Identity : public LinearOperator {
 public:
  explicit Identity(Index n) : n_(n) {}

  VecX operator()(const VecX& v) const override { return v; }

  Index size() const override { return n_; }

 private:
  const Index n_;
};

class Preconditioner : public LinearOperator {
 public:
  explicit Preconditioner(const RandomSymmetric& op) : n_(op.size()) {
    MatX perturbation = 0.1 * VecX::Random(n_).asDiagonal();
    m_ = op.matrix().inverse() + perturbation;
  }

  VecX operator()(const VecX& v) const override { return m_ * v; }

  Index size() const override { return n_; }

 private:
  const Index n_;
  MatX m_;
};

class KrylovTest : public ::testing::Test {
 protected:
  static constexpr Index n = 100;

  std::unique_ptr<RandomSymmetric> op;
  std::unique_ptr<Preconditioner> pc;

  VecX solution;
  VecX rhs;

  VecX x0;

  void SetUp() override {
    op = std::make_unique<RandomSymmetric>(n);
    pc = std::make_unique<Preconditioner>(*op);

    solution = (VecX::Random(n) + VecX::Ones(n)) / 2.0;
    rhs = (*op)(solution);

    x0 = VecX::Random(n);
  }

  void TearDown() override {}

  template <class Solver>
  void test_solver(bool with_initial_solution, bool with_right_pc, bool with_left_pc) {
    Solver solver(*op, rhs, n);
    if (with_initial_solution) {
      solver.set_initial_solution(x0);
    }
    if (with_right_pc) {
      solver.set_right_preconditioner(*pc);
    }
    if (with_left_pc) {
      solver.set_left_preconditioner(*pc);
    }
    solver.setup();

    auto last_residual = 0.0;
    for (Index i = 0; i < solver.max_iterations(); i++) {
      solver.iterate_process();
      auto approx_solution = solver.solution_vector();
      auto current_residual = solver.relative_residual();

      if (!with_left_pc) {
        EXPECT_NEAR(relative_error((*op)(approx_solution), rhs), current_residual, 1e-12);
        EXPECT_LT(relative_error(rhs - solver.residual_vector(), (*op)(approx_solution)), 1e-12);
      }

      if (i > 0) {
        EXPECT_LT(current_residual, last_residual);
      }

      last_residual = current_residual;
    }
  }
};

}  // namespace

TEST_F(KrylovTest, breakdown) {
  Identity op(n);
  // Normalizing e0 is exact, so the Arnoldi process breaks down exactly.
  VecX e0 = VecX::Unit(n, 0);
  Fgmres solver(op, e0, n);
  solver.setup();
  solver.iterate_process();

  EXPECT_EQ(solver.absolute_residual(), 0.0);
  EXPECT_TRUE(solver.residual_vector().isZero());
  EXPECT_LT(relative_error(solver.solution_vector(), e0), 1e-15);
}

TEST_F(KrylovTest, fgmres) {
  test_solver<Fgmres>(false, false, false);
  test_solver<Fgmres>(false, true, false);
  test_solver<Fgmres>(true, false, false);
  test_solver<Fgmres>(true, true, false);
}

TEST_F(KrylovTest, gmres) {
  test_solver<Gmres>(false, false, false);
  test_solver<Gmres>(false, true, false);
  test_solver<Gmres>(false, false, true);
  test_solver<Gmres>(true, false, false);
  test_solver<Gmres>(true, true, false);
  test_solver<Gmres>(true, false, true);
}

TEST_F(KrylovTest, minres) {
  test_solver<Minres>(false, false, false);
  test_solver<Minres>(true, false, false);
}
