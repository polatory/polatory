#include <Python.h>
#undef _GNU_SOURCE
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <limits>
#include <polatory/polatory.hpp>
#include <vector>

namespace py = pybind11;
using namespace py::literals;
using namespace polatory;

static constexpr double kInfinity = std::numeric_limits<double>::infinity();

PYBIND11_MODULE(_structural, m) {
  // Register the standard Polatory types (Model<3>, FieldFunction, etc.) first.
  py::module_::import("polatory._core");

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
                       const Eigen::Vector3d& bbox_max, std::vector<Index> support_indices,
                       std::vector<double> model_parameters) {
             geometry::Point3 min_row = bbox_min.transpose();
             geometry::Point3 max_row = bbox_max.transpose();
             return DomainSpec(anisotropy, min_row, max_row, std::move(support_indices),
                               std::move(model_parameters));
           }),
           "anisotropy"_a, "bbox_min"_a, "bbox_max"_a, "support_indices"_a,
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
      .def("sample", &DomainBuilder::sample,
           "query_points"_a,
           "inputs"_a,
           "trend_type"_a = TrendType::kStrongestAlongInputs);

  py::class_<StructuralInterpolant>(m, "StructuralInterpolant3")
      .def(py::init<const Model<3>&, double, double>(), "base_model"_a,
           "outside_value"_a = -1.0, "blend_power"_a = 1.0)
      .def_property_readonly("bbox_min", [](const StructuralInterpolant& interpolant) {
        return Eigen::Vector3d(interpolant.bbox().min().transpose());
      })
      .def_property_readonly("bbox_max", [](const StructuralInterpolant& interpolant) {
        return Eigen::Vector3d(interpolant.bbox().max().transpose());
      })
      .def_property_readonly("blend_power", &StructuralInterpolant::blend_power)
      .def_property_readonly("num_domains", &StructuralInterpolant::num_domains)
      .def_property_readonly("outside_value", &StructuralInterpolant::outside_value)
      .def("fit", &StructuralInterpolant::fit, "points"_a, "values"_a, "domains"_a,
           "tolerance"_a, "max_iter"_a = 100, "accuracy"_a = kInfinity)
      .def(
          "fit_from_meshes",
          [](StructuralInterpolant& interpolant,
             const geometry::Points3& points,
             const VecX& values,
             const std::vector<TrendInput>& inputs,
             TrendType trend_type,
             double tolerance,
             int max_iter,
             double accuracy,
             double domain_size,
             double overlap,
             Index min_support_points) {
            DomainBuilder builder(domain_size, overlap, min_support_points);
            auto domains = builder.build(points, inputs, trend_type);
            interpolant.fit(points, values, domains, tolerance, max_iter, accuracy);
            return domains;
          },
          "points"_a,
          "values"_a,
          "inputs"_a,
          "trend_type"_a = TrendType::kStrongestAlongInputs,
          "tolerance"_a,
          "max_iter"_a = 100,
          "accuracy"_a = kInfinity,
          "domain_size"_a = 0.0,
          "overlap"_a = 0.0,
          "min_support_points"_a = 4)
      .def("evaluate", &StructuralInterpolant::evaluate, "points"_a,
           "accuracy"_a = kInfinity);

  py::class_<isosurface::StructuralRbfFieldFunction, isosurface::FieldFunction>(
      m, "StructuralRbfFieldFunction")
      .def(py::init<StructuralInterpolant&, double>(), "interpolant"_a,
           "accuracy"_a = kInfinity);
}
