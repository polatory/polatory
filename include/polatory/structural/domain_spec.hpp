#pragma once

#include <Eigen/LU>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace polatory::structural {

class DomainSpec3 {
 public:
  DomainSpec3(const Mat3& anisotropy, const geometry::Point3& bbox_min,
              const geometry::Point3& bbox_max, std::vector<Index> support_indices,
              std::vector<double> model_parameters = {})
      : anisotropy_(anisotropy),
        bbox_(bbox_min, bbox_max),
        support_indices_(std::move(support_indices)),
        model_parameters_(std::move(model_parameters)) {
    if (!(anisotropy_.determinant() > 0.0)) {
      throw std::invalid_argument("anisotropy must have a positive determinant");
    }
    if (bbox_.is_empty()) {
      throw std::invalid_argument("domain bbox must not be empty");
    }
    if (support_indices_.empty()) {
      throw std::invalid_argument("support_indices must not be empty");
    }
  }

  const Mat3& anisotropy() const { return anisotropy_; }

  const geometry::Bbox3& bbox() const { return bbox_; }

  const std::vector<double>& model_parameters() const { return model_parameters_; }

  const std::vector<Index>& support_indices() const { return support_indices_; }

 private:
  Mat3 anisotropy_;
  geometry::Bbox3 bbox_;
  std::vector<Index> support_indices_;
  std::vector<double> model_parameters_;
};

}  // namespace polatory::structural
