#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <jizai/types.hpp>

namespace jizai::numeric {

inline double condition_number(const MatX& m) {
  Eigen::SelfAdjointEigenSolver<MatX> es(m, Eigen::EigenvaluesOnly);
  VecX abs_ev = es.eigenvalues().cwiseAbs();
  return abs_ev.maxCoeff() / abs_ev.minCoeff();
}

}  // namespace jizai::numeric
