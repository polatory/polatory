#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/model.hpp>
#include <polatory/rbf/cov_spheroidal3.hpp>
#include <polatory/structural/domain_spec.hpp>
#include <polatory/structural/interpolant.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace {

using polatory::Index;
using polatory::Mat3;
using polatory::Model;
using polatory::VecX;
using polatory::geometry::Point3;
using polatory::geometry::Points3;
using polatory::rbf::CovSpheroidal3;
using polatory::structural::DomainSpec3;
using polatory::structural::StructuralInterpolant3;

TEST(structural_interpolant, one_domain_matches_standard_interpolant) {
  Points3 points(8, 3);
  points << 0.0, 0.0, 0.0,  //
      1.0, 0.0, 0.0,       //
      0.0, 1.0, 0.0,       //
      1.0, 1.0, 0.0,       //
      0.0, 0.0, 1.0,       //
      1.0, 0.0, 1.0,       //
      0.0, 1.0, 1.0,       //
      1.0, 1.0, 1.0;

  VecX values(8);
  values << -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0;

  Mat3 anisotropy;
  anisotropy << 0.8, 0.0, 0.0,  //
      0.0, 1.2, 0.0,            //
      0.0, 0.0, 1.0;

  CovSpheroidal3<3> standard_rbf({1.0, 4.0});
  standard_rbf.set_anisotropy(anisotropy);
  Model<3> standard_model(std::move(standard_rbf), 0);
  standard_model.set_nugget(0.01);

  polatory::Interpolant<3> standard(standard_model);
  standard.fit(points, values, 1e-8, 100);

  CovSpheroidal3<3> structural_rbf({1.0, 4.0});
  Model<3> structural_model(std::move(structural_rbf), 0);
  structural_model.set_nugget(0.01);

  Point3 bbox_min;
  bbox_min << -1.0, -1.0, -1.0;
  Point3 bbox_max;
  bbox_max << 2.0, 2.0, 2.0;

  std::vector<Index> support_indices{0, 1, 2, 3, 4, 5, 6, 7};
  DomainSpec3 domain(anisotropy, bbox_min, bbox_max, support_indices);

  StructuralInterpolant3 structural(structural_model);
  structural.fit(points, values, {domain}, 1e-8, 100);

  Points3 queries(4, 3);
  queries << 0.2, 0.2, 0.2,  //
      0.8, 0.3, 0.4,         //
      0.4, 0.7, 0.9,         //
      0.5, 0.5, 0.5;

  VecX expected = standard.evaluate(queries);
  VecX actual = structural.evaluate(queries);

  EXPECT_LT((expected - actual).cwiseAbs().maxCoeff(), 1e-9);
  EXPECT_EQ(structural.num_domains(), 1);
}

TEST(structural_interpolant, returns_outside_value_outside_all_domain_boxes) {
  Points3 points(4, 3);
  points << 0.0, 0.0, 0.0,  //
      1.0, 0.0, 0.0,       //
      0.0, 1.0, 0.0,       //
      0.0, 0.0, 1.0;

  VecX values(4);
  values << -1.0, 0.0, 0.5, 1.0;

  CovSpheroidal3<3> rbf({1.0, 4.0});
  Model<3> model(std::move(rbf), 0);

  Point3 bbox_min;
  bbox_min << -0.5, -0.5, -0.5;
  Point3 bbox_max;
  bbox_max << 1.5, 1.5, 1.5;

  DomainSpec3 domain(Mat3::Identity(), bbox_min, bbox_max, {0, 1, 2, 3});

  StructuralInterpolant3 structural(model, -7.0);
  structural.fit(points, values, {domain}, 1e-6, 100);

  Points3 query(1, 3);
  query << 10.0, 10.0, 10.0;

  EXPECT_DOUBLE_EQ(structural.evaluate(query)(0), -7.0);
}

}  // namespace
