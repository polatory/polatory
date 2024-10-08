#pragma once

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <utility>

namespace polatory::geometry {

// Decomposes the *inverse* of the given anisotropy matrix into rotation and scaling parts.
template <int Dim>
std::pair<Mat<Dim>, Vector<Dim>> decompose_inverse_anisotropy(const Mat<Dim> aniso) {
  // https://math.stackexchange.com/a/3895150
  Mat<Dim> m = aniso.transpose() * aniso;
  Eigen::SelfAdjointEigenSolver<Mat<Dim>> eigensolver(m);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("failed to compute the eigensystem");
  }
  Mat<Dim> rot = eigensolver.eigenvectors();
  if (rot.determinant() < 0) {
    rot.col(0) *= -1.0;
  }
  Vector<Dim> scale = eigensolver.eigenvalues().array().rsqrt();
  return {rot, scale};
}

}  // namespace polatory::geometry
