#include <Python.h>
#undef _GNU_SOURCE
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <limits>
#include <polatory/polatory.hpp>
#include <polatory/structural/adaptive_domain_builder.hpp>
#include <polatory/structural/adaptive_domain_orientation_average.hpp>
#include <polatory/structural/domain_orientation_average.hpp>
#include <vector>

namespace py = pybind11;
using namespace py::literals;
using namespace polatory;

static constexpr double kInfinity = std::numeric_limits<double>::infinity();

PYBIND11_MODULE(_structural, m) {
  // Register the standard Polatory types (Model<3>, FieldFunction, etc.) first.
  py::module_::import("polatory._core");

  using AdaptiveDomainBuilder = structural::AdaptiveStructuralDomainBuilder3;
  using DomainBuilder = structural::StructuralDomainBuilder3;
  using DomainSpec = structural::DomainSpec3;
  using StructuralInterpolant = structural::StructuralInterpolant3;
  using TrendInput = structural::StructuralTrendInput3;
  using TrendSamples = structural::StructuralTrendSamples3;
  using TrendType = structural::StructuralTrendType;

  py::enum_<TrendType>(m, "StructuralTrendType")
      .value("STRONGEST_ALONG_INPUTS", TrendType::kStrongestAlongInputs)
      .value("BLENDING", TrendType::kBlending)
      .value("NON_DECAYING", TrendType::kNonDecaying)
      .export_values();

  py::class_<TrendInput>(m, "StructuralTrendInput3")
      .def(py::init<geometry::Points3, structural::TriangleFaces3, double, double>(),
           "vertices"_a, "faces"_a, "strength"_a, "range"_a)
      .def_property_readonly("vertices", &TrendInput::vertices)
      .def_property_readonly("faces", &TrendInput::faces)
      .def_property_readonly("strength", &TrendInput::strength)
      .def_property_readonly("range", &TrendInput::range);

  py::class_<TrendSamples>(m, "StructuralTrendSamples3")
      .def_property_readonly("normals", [](const TrendSamples& samples) {
        return samples.normals;
      })
      .def_property_readonly("ratios", [](const TrendSamples& samples) {
        return samples.ratios;
      })
      .def_property_readonly("distances", [](const TrendSamples& samples) {
        return samples.distances;
      })
      .def_property_readonly("dominant_inputs", [](const TrendSamples& samples) {
        return samples.dominant_inputs;
      })
      .def_property_readonly("anisotropies", [](const TrendSamples& samples) {
        return samples.anisotropies;
      });

  py::class_<DomainSpec>(m, "StructuralDomain3")
      .def(py::init([](const Mat3& anisotropy, const Eigen::Vector3d& bbox_min,
                       const Eigen::Vector3d& bbox_max,
                       std::vector<Index> support_indices,
                       std::vector<double> model_parameters) {
             geometry::Point3 min_row = bbox_min.transpose();
             geometry::Point3 max_row = bbox_max.transpose();
             return DomainSpec(anisotropy, min_row, max_row,
                               std::move(support_indices),
                               std::move(model_parameters));
           }),
           "anisotropy"_a, "bbox_min"_a, "bbox_max"_a,
           "support_indices"_a,
           "model_parameters"_a = std::vector<double>{})
      .def_property_readonly("anisotropy", &DomainSpec::anisotropy)
      .def_property_readonly("bbox_min", [](const DomainSpec& spec) {
        return Eigen::Vector3d(spec.bbox().min().transpose());
      })
      .def_property_readonly("bbox_max", [](const DomainSpec& spec) {
        return Eigen::Vector3d(spec.bbox().max().transpose());
      })
      .def_property_readonly("support_indices", &DomainSpec::support_indices)
      .def_property_readonly("model_parameters", &DomainSpec::model_parameters);

  py::class_<DomainBuilder>(m, "StructuralDomainBuilder3")
      .def(py::init<double, double, Index>(),
           "domain_size"_a = 0.0,
           "overlap"_a = 0.0,
           "min_support_points"_a = 4)
      .def_property_readonly("domain_size", &DomainBuilder::domain_size)
      .def_property_readonly("overlap", &DomainBuilder::overlap)
      .def_property_readonly("min_support_points",
                             &DomainBuilder::min_support_points)
      .def("build", &DomainBuilder::build,
           "points"_a,
           "inputs"_a,
           "trend_type"_a = TrendType::kStrongestAlongInputs,
           "model_parameters"_a = std::vector<double>{})
      .def(
          "build_orientation_averaged",
          [](const DomainBuilder& builder,
             const geometry::Points3& points,
             const std::vector<TrendInput>& inputs,
             TrendType trend_type,
             const std::vector<double>& model_parameters) {
            return structural::build_orientation_averaged_domains(
                points, inputs, trend_type, model_parameters,
                builder.domain_size(), builder.overlap(),
                builder.min_support_points());
          },
          "points"_a,
          "inputs"_a,
          "trend_type"_a = TrendType::kStrongestAlongInputs,
          "model_parameters"_a = std::vector<double>{})
      .def("sample", &DomainBuilder::sample,
           "query_points"_a,
           "inputs"_a,
           "trend_type"_a = TrendType::kStrongestAlongInputs);

  py::class_<AdaptiveDomainBuilder>(m, "AdaptiveStructuralDomainBuilder3")
      .def(py::init<double, double, double, double, Index, Index, int>(),
           "overlap"_a = 0.0,
           "orientation_consistency"_a = 0.97,
           "minimum_core_size"_a = 0.0,
           "maximum_core_size"_a = 0.0,
           "minimum_core_points"_a = 24,
           "minimum_support_points"_a = 4,
           "maximum_depth"_a = 20)
      .def_property_readonly("overlap",
                             &AdaptiveDomainBuilder::overlap)
      .def_property_readonly(
          "orientation_consistency",
          &AdaptiveDomainBuilder::orientation_consistency)
      .def_property_readonly("minimum_core_size",
                             &AdaptiveDomainBuilder::minimum_core_size)
      .def_property_readonly("maximum_core_size",
                             &AdaptiveDomainBuilder::maximum_core_size)
      .def_property_readonly("minimum_core_points",
                             &AdaptiveDomainBuilder::minimum_core_points)
      .def_property_readonly("minimum_support_points",
                             &AdaptiveDomainBuilder::minimum_support_points)
      .def_property_readonly("maximum_depth",
                             &AdaptiveDomainBuilder::maximum_depth)
      .def("build", &AdaptiveDomainBuilder::build,
           "points"_a,
           "inputs"_a,
           "trend_type"_a = TrendType::kStrongestAlongInputs,
           "model_parameters"_a = std::vector<double>{})
      .def(
          "build_orientation_averaged",
          [](const AdaptiveDomainBuilder& builder,
             const geometry::Points3& points,
             const std::vector<TrendInput>& inputs,
             TrendType trend_type,
             const std::vector<double>& model_parameters) {
            return structural::build_adaptive_orientation_averaged_domains(
                points, inputs, trend_type, model_parameters,
                builder.overlap(), builder.orientation_consistency(),
                builder.minimum_core_size(), builder.maximum_core_size(),
                builder.minimum_core_points(), builder.minimum_support_points(),
                builder.maximum_depth());
          },
          "points"_a,
          "inputs"_a,
          "trend_type"_a = TrendType::kStrongestAlongInputs,
          "model_parameters"_a = std::vector<double>{});

  py::class_<StructuralInterpolant>(m, "StructuralInterpolant3")
      .def(py::init<const Model<3>&, double, double, double>(),
           "base_model"_a,
           "outside_value"_a = -1.0,
           "blend_power"_a = 1.0,
           "alignment_strength"_a = 0.0)
      .def_property_readonly("bbox_min", [](const StructuralInterpolant& interpolant) {
        return Eigen::Vector3d(interpolant.bbox().min().transpose());
      })
      .def_property_readonly("bbox_max", [](const StructuralInterpolant& interpolant) {
        return Eigen::Vector3d(interpolant.bbox().max().transpose());
      })
      .def_property_readonly("blend_power", &StructuralInterpolant::blend_power)
      .def_property_readonly("alignment_strength",
                             &StructuralInterpolant::alignment_strength)
      .def_property_readonly("domain_offsets",
                             &StructuralInterpolant::domain_offsets)
      .def_property_readonly("num_domains", &StructuralInterpolant::num_domains)
      .def_property_readonly("outside_value", &StructuralInterpolant::outside_value)
      .def("fit", &StructuralInterpolant::fit, "points"_a, "values"_a,
           "domains"_a, "tolerance"_a, "max_iter"_a = 100,
           "accuracy"_a = kInfinity)
      .def(
          "fit_from_meshes",
          [](StructuralInterpolant& interpolant,
             const geometry::Points3& points,
             const VecX& values,
             const std::vector<TrendInput>& inputs,
             double tolerance,
             TrendType trend_type,
             int max_iter,
             double accuracy,
             double domain_size,
             double overlap,
             Index min_support_points) {
            DomainBuilder builder(domain_size, overlap, min_support_points);
            auto domains = builder.build(points, inputs, trend_type);
            interpolant.fit(points, values, domains, tolerance, max_iter,
                             accuracy);
            return domains;
          },
          "points"_a,
          "values"_a,
          "inputs"_a,
          "tolerance"_a,
          "trend_type"_a = TrendType::kStrongestAlongInputs,
          "max_iter"_a = 100,
          "accuracy"_a = kInfinity,
          "domain_size"_a = 0.0,
          "overlap"_a = 0.0,
          "min_support_points"_a = 4)
      .def(
          "fit_from_meshes_adaptive",
          [](StructuralInterpolant& interpolant,
             const geometry::Points3& points,
             const VecX& values,
             const std::vector<TrendInput>& inputs,
             double tolerance,
             TrendType trend_type,
             int max_iter,
             double accuracy,
             double overlap,
             double orientation_consistency,
             double minimum_core_size,
             double maximum_core_size,
             Index minimum_core_points,
             Index minimum_support_points,
             int maximum_depth) {
            AdaptiveDomainBuilder builder(
                overlap, orientation_consistency, minimum_core_size,
                maximum_core_size, minimum_core_points,
                minimum_support_points, maximum_depth);
            auto domains = builder.build(points, inputs, trend_type);
            interpolant.fit(points, values, domains, tolerance, max_iter,
                             accuracy);
            return domains;
          },
          "points"_a,
          "values"_a,
          "inputs"_a,
          "tolerance"_a,
          "trend_type"_a = TrendType::kStrongestAlongInputs,
          "max_iter"_a = 100,
          "accuracy"_a = kInfinity,
          "overlap"_a = 0.0,
          "orientation_consistency"_a = 0.97,
          "minimum_core_size"_a = 0.0,
          "maximum_core_size"_a = 0.0,
          "minimum_core_points"_a = 24,
          "minimum_support_points"_a = 4,
          "maximum_depth"_a = 20)
      .def("evaluate", &StructuralInterpolant::evaluate, "points"_a,
           "accuracy"_a = kInfinity);

  py::class_<isosurface::StructuralRbfFieldFunction, isosurface::FieldFunction>(
      m, "StructuralRbfFieldFunction")
      .def(py::init<StructuralInterpolant&, double>(), "interpolant"_a,
           "accuracy"_a = kInfinity);
}
