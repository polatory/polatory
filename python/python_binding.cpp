#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <polatory/polatory.hpp>

#define xstr(s) str(s)
#define str(s) #s

using namespace polatory;
namespace py = pybind11;
using namespace py::literals;

geometry::bbox3d bbox3d_from_points(const geometry::points3d& points) {
  return geometry::bbox3d::from_points(points);
}

PYBIND11_MODULE(_core, m) {
  py::class_<rbf::rbf_base>(m, "_RbfBase")
      .def_property("anisotropy", &rbf::rbf_base::anisotropy, &rbf::rbf_base::set_anisotropy);

  py::class_<rbf::biharmonic2d, rbf::rbf_base>(m, "Biharmonic2D")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::biharmonic3d, rbf::rbf_base>(m, "Biharmonic3D")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_exponential, rbf::rbf_base>(m, "CovExponential")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_spheroidal3, rbf::rbf_base>(m, "CovSpheroidal3")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_spheroidal5, rbf::rbf_base>(m, "CovSpheroidal5")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_spheroidal7, rbf::rbf_base>(m, "CovSpheroidal7")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_spheroidal9, rbf::rbf_base>(m, "CovSpheroidal9")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::multiquadric1, rbf::rbf_base>(m, "Multiquadric1")
      .def(py::init<const std::vector<double>&>(), "params"_a);

  py::class_<model>(m, "Model")
      .def(py::init<const rbf::rbf_base&, int, int>(), "rbf"_a, "poly_dimension"_a, "poly_degree"_a)
      .def_property("nugget", &model::nugget, &model::set_nugget);

  py::class_<interpolant>(m, "Interpolant")
      .def(py::init<const model&>(), "model"_a)
      .def("evaluate", &interpolant::evaluate, "points"_a)
      .def("fit", &interpolant::fit, "points"_a, "values"_a, "absolute_tolerance"_a,
           "max_iter"_a = 32)
      .def("fit_incrementally", &interpolant::fit_incrementally, "points"_a, "values"_a,
           "absolute_tolerance"_a, "max_iter"_a = 32)
      .def("fit_inequality", &interpolant::fit_inequality, "points"_a, "values"_a, "values_lb"_a,
           "values_ub"_a, "absolute_tolerance"_a, "max_iter"_a = 32)
      .def_property_readonly("centers", &interpolant::centers)
      .def_property_readonly("weights", &interpolant::weights);

  py::class_<point_cloud::distance_filter>(m, "DistanceFilter")
      .def(py::init<const geometry::points3d&, double>(), "points"_a, "distance"_a)
      .def_property_readonly("filtered_indices", &point_cloud::distance_filter::filtered_indices);

  py::class_<point_cloud::normal_estimator>(m, "NormalEstimator")
      .def(py::init<const geometry::points3d&>(), "points"_a)
      .def("estimate_with_knn",
           py::overload_cast<index_t, double>(&point_cloud::normal_estimator::estimate_with_knn),
           "k"_a, "plane_factor_threshold"_a = 1.8)
      .def("estimate_with_knn",
           py::overload_cast<const std::vector<index_t>&, double>(
               &point_cloud::normal_estimator::estimate_with_knn),
           "ks"_a, "plane_factor_threshold"_a = 1.8)
      .def("estimate_with_radius", &point_cloud::normal_estimator::estimate_with_radius, "radius"_a,
           "plane_factor_threshold"_a = 1.8)
      .def("orient_by_outward_vector", &point_cloud::normal_estimator::orient_by_outward_vector,
           "v"_a)
      .def("orient_closed_surface", &point_cloud::normal_estimator::orient_closed_surface, "k"_a);

  py::class_<point_cloud::offset_points_generator>(m, "OffsetPointsGenerator")
      .def(py::init<const geometry::points3d&, const geometry::vectors3d&, double>(), "points"_a,
           "normals"_a, "offset"_a)
      .def(
          py::init<const geometry::points3d&, const geometry::vectors3d&, const common::valuesd&>(),
          "points"_a, "normals"_a, "offsets"_a)
      .def_property_readonly("new_points", &point_cloud::offset_points_generator::new_points)
      .def_property_readonly("new_normals", &point_cloud::offset_points_generator::new_normals);

  py::class_<point_cloud::sdf_data_generator>(m, "SdfDataGenerator")
      .def(
          py::init<const geometry::points3d&, const geometry::vectors3d&, double, double, double>(),
          "points"_a, "normals"_a, "min_distance"_a, "max_distance"_a, "multiplication"_a = 2.0)
      .def_property_readonly("sdf_points", &point_cloud::sdf_data_generator::sdf_points)
      .def_property_readonly("sdf_values", &point_cloud::sdf_data_generator::sdf_values);

  py::class_<geometry::bbox3d>(m, "Bbox3d")
      .def(py::init<>())
      .def(py::init<const geometry::point3d&, const geometry::point3d&>(), "min"_a, "max"_a)
      .def_static("from_points", &bbox3d_from_points, "points"_a)
      .def_property_readonly("min", &geometry::bbox3d::min)
      .def_property_readonly("max", &geometry::bbox3d::max);

  py::class_<isosurface::field_function>(m, "_FieldFunction");

  py::class_<isosurface::rbf_field_function, isosurface::field_function>(m, "RbfFieldFunction")
      .def(py::init<interpolant&>(), "interpolant"_a);

  py::class_<isosurface::rbf_field_function_25d, isosurface::field_function>(m,
                                                                             "RbfFieldFunction25d")
      .def(py::init<interpolant&>(), "interpolant"_a);

  py::class_<isosurface::surface>(m, "Surface")
      .def("export_obj", &isosurface::surface::export_obj, "filename"_a)
      .def_property_readonly("vertices", &isosurface::surface::vertices)
      .def_property_readonly("faces", &isosurface::surface::faces);

  py::class_<isosurface::isosurface>(m, "Isosurface")
      .def(py::init<const geometry::bbox3d&, double>(), "bbox"_a, "resolution"_a)
      .def("generate", &isosurface::isosurface::generate, "field_fn"_a, "isovalue"_a = 0.0)
      .def("generate_from_seed_points", &isosurface::isosurface::generate_from_seed_points,
           "seed_points"_a, "field_fn"_a, "isovalue"_a = 0.0);

  m.attr("__version__") = xstr(POLATORY_VERSION);
}
