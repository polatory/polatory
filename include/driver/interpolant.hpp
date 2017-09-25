#pragma once

#include <functional>
#include <tuple>

#include <Eigen/Core>

#include "geometry/bbox3.hpp"
#include "interpolation/rbf_fitter.hpp"
#include "interpolation/rbf_incremental_fitter.hpp"
#include "rbf/rbf_base.hpp"

namespace polatory {
namespace driver {

class interpolant {
public:
   using points_type = std::vector<Eigen::Vector3d>;
   using values_type = Eigen::VectorXd;
   using point_transform_type = std::function<Eigen::Vector3d(const Eigen::Vector3d&)>;

   interpolant(const rbf::rbf_base& rbf, int poly_degree)
      : rbf_(rbf)
      , poly_degree_(poly_degree)
   {
   }

   // TODO: Create evaluator interface.
   void get_evaluator(const geometry::bbox3d& bbox)
   {
      // TODO
   }

   void fit(const points_type& points, const values_type& values, double absolute_tolerance)
   {
      auto transformed = transform_points(points);

      interpolation::rbf_fitter fitter(rbf_, poly_degree_, transformed);

      centers_ = transformed;
      weights_ = fitter.fit(values, absolute_tolerance);
   }

   void fit_incrementally(const points_type& points, const values_type& values, double absolute_tolerance)
   {
      auto transformed = transform_points(points);

      interpolation::rbf_incremental_fitter fitter(rbf_, poly_degree_, transformed);

      std::tie(centers_, weights_) = fitter.fit(values, absolute_tolerance);
   }

   void set_point_transform(const point_transform_type& forward_transform)
   {
      point_transform_ = forward_transform;
   }

private:
   points_type transform_points(const points_type& points) const
   {
      if (!point_transform_)
         return points;

      points_type transformed;
      transformed.reserve(points.size());

      for (const auto& p : points) {
         transformed.push_back(point_transform_(p));
      }

      return transformed;
   }

   const rbf::rbf_base& rbf_;
   const int poly_degree_;

   point_transform_type point_transform_;

   std::vector<Eigen::Vector3d> centers_;
   Eigen::VectorXd weights_;
};


} // namespace driver
} // namespace polatory
