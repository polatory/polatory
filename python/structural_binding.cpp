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

  using DomainSpec = structural::DomainSpec3;
  using StructuralInterpolant = structural::StructuralInterpolant3;

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
      .def("evaluate", &StructuralInterpolant::evaluate, "points"_a,
           "accuracy"_a = kInfinity);

  py::class_<isosurface::StructuralRbfFieldFunction, isosurface::FieldFunction>(
      m, "StructuralRbfFieldFunction")
      .def(py::init<StructuralInterpolant&, double>(), "interpolant"_a,
           "accuracy"_a = kInfinity);
}
