#pragma once

#include <Eigen/Eigenvalues>
#include <polatory/geometry/point3d.hpp>
#include <stdexcept>
#include <utility>

namespace polatory::geometry {

// Decomposes the *inverse* of the given anisotropy matrix into rotation and scaling parts.
template <int Dim>
std::pair<matrixNd<Dim>, vectorNd<Dim>> decompose_inverse_anisotropy(const matrixNd<Dim> aniso) {
  // https://math.stackexchange.com/a/3895150
  matrixNd<Dim> m = aniso.transpose() * aniso;
  Eigen::SelfAdjointEigenSolver<matrixNd<Dim>> eigensolver(m);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("Failed to compute eigensystem.");
  }
  matrixNd<Dim> rot = eigensolver.eigenvectors();
  if (rot.determinant() < 0) {
    rot.col(0) *= -1.0;
  }
  vectorNd<Dim> scale = eigensolver.eigenvalues().array().rsqrt();
  return {rot, scale};
}

}  // namespace polatory::geometry
