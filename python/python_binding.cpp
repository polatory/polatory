#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#undef _GNU_SOURCE
#include <polatory/polatory.hpp>
#include <string>
#include <vector>

#define xstr(s) str(s)
#define str(s) #s

using namespace polatory;
namespace py = pybind11;
using namespace py::literals;

template <int Dim>
geometry::bboxNd<Dim> bbox_from_points(const geometry::pointsNd<Dim>& points) {
  return geometry::bboxNd<Dim>::from_points(points);
}

template <int Dim, class Rbf>
void define_rbf(py::module& m, const std::string& name) {
  py::class_<Rbf, rbf::rbf_proxy<Dim>>(m, name.c_str())
      .def(py::init<const std::vector<double>&>(), "params"_a);
}

template <int Dim>
void define_module(py::module& m) {
  py::class_<geometry::bboxNd<Dim>>(m, "Bbox")
      .def(py::init<>())
      .def(py::init<const geometry::pointNd<Dim>&, const geometry::pointNd<Dim>&>(), "min"_a,
           "max"_a)
      .def_static("from_points", &bbox_from_points<Dim>, "points"_a)
      .def_property_readonly("min", &geometry::bboxNd<Dim>::min)
      .def_property_readonly("max", &geometry::bboxNd<Dim>::max);

  py::class_<rbf::rbf_proxy<Dim>>(m, "Rbf")
      .def_property("anisotropy", &rbf::rbf_proxy<Dim>::anisotropy,
                    &rbf::rbf_proxy<Dim>::set_anisotropy)
      .def_property_readonly("cpd_order", &rbf::rbf_proxy<Dim>::cpd_order)
      .def_property_readonly("num_parameters", &rbf::rbf_proxy<Dim>::num_parameters)
      .def_property("parameters", &rbf::rbf_proxy<Dim>::parameters,
                    &rbf::rbf_proxy<Dim>::set_parameters)
      .def("evaluate", &rbf::rbf_proxy<Dim>::evaluate, "diff"_a)
      .def("evaluate_gradient", &rbf::rbf_proxy<Dim>::evaluate_gradient, "diff"_a)
      .def("evaluate_hessian", &rbf::rbf_proxy<Dim>::evaluate_hessian, "diff"_a);

  define_rbf<Dim, rbf::biharmonic2d<Dim>>(m, "Biharmonic2D");
  define_rbf<Dim, rbf::biharmonic3d<Dim>>(m, "Biharmonic3D");
  define_rbf<Dim, rbf::cov_cauchy3<Dim>>(m, "CovCauchy3");
  define_rbf<Dim, rbf::cov_cauchy5<Dim>>(m, "CovCauchy5");
  define_rbf<Dim, rbf::cov_cauchy7<Dim>>(m, "CovCauchy7");
  define_rbf<Dim, rbf::cov_cauchy9<Dim>>(m, "CovCauchy9");
  define_rbf<Dim, rbf::cov_cubic<Dim>>(m, "CovCubic");
  define_rbf<Dim, rbf::cov_exponential<Dim>>(m, "CovExponential");
  define_rbf<Dim, rbf::cov_gaussian<Dim>>(m, "CovGaussian");
  define_rbf<Dim, rbf::cov_spherical<Dim>>(m, "CovSpherical");
  define_rbf<Dim, rbf::cov_spheroidal3<Dim>>(m, "CovSpheroidal3");
  define_rbf<Dim, rbf::cov_spheroidal5<Dim>>(m, "CovSpheroidal5");
  define_rbf<Dim, rbf::cov_spheroidal7<Dim>>(m, "CovSpheroidal7");
  define_rbf<Dim, rbf::cov_spheroidal9<Dim>>(m, "CovSpheroidal9");
  define_rbf<Dim, rbf::inverse_multiquadric1<Dim>>(m, "InverseMultiquadric1");
  define_rbf<Dim, rbf::multiquadric1<Dim>>(m, "Multiquadric1");
  define_rbf<Dim, rbf::multiquadric3<Dim>>(m, "Multiquadric3");
  define_rbf<Dim, rbf::triharmonic2d<Dim>>(m, "Triharmonic2D");
  define_rbf<Dim, rbf::triharmonic3d<Dim>>(m, "Triharmonic3D");

  py::class_<model<Dim>>(m, "Model")
      .def(py::init<rbf::rbf_proxy<Dim>, int>(), "rbf"_a, "poly_degree"_a)
      .def(py::init<std::vector<rbf::rbf_proxy<Dim>>, int>(), "rbfs"_a, "poly_degree"_a)
      .def_property_readonly("cpd_order", &model<Dim>::cpd_order)
      .def_property("nugget", &model<Dim>::nugget, &model<Dim>::set_nugget)
      .def_property_readonly("num_parameters", &model<Dim>::num_parameters)
      .def_property("parameters", &model<Dim>::parameters, &model<Dim>::set_parameters)
      .def_property_readonly("poly_basis_size", &model<Dim>::poly_basis_size)
      .def_property_readonly("poly_degree", &model<Dim>::poly_degree)
      .def_property_readonly("rbfs", &model<Dim>::rbfs);

  py::class_<interpolant<Dim>>(m, "Interpolant")
      .def(py::init<const model<Dim>&>(), "model"_a)
      .def_property_readonly("centers", &interpolant<Dim>::centers)
      .def_property_readonly("weights", &interpolant<Dim>::weights)
      .def("evaluate", &interpolant<Dim>::evaluate, "points"_a)
      .def("fit",
           py::overload_cast<const geometry::pointsNd<Dim>&, const common::valuesd&, double, int>(
               &interpolant<Dim>::fit),
           "points"_a, "values"_a, "absolute_tolerance"_a, "max_iter"_a = 100)
      .def("fit_with_grad",
           py::overload_cast<const geometry::pointsNd<Dim>&, const geometry::pointsNd<Dim>&,
                             const common::valuesd&, double, double, int>(&interpolant<Dim>::fit),
           "points"_a, "grad_points"_a, "values"_a, "absolute_tolerance"_a,
           "grad_absolute_tolerance"_a, "max_iter"_a = 100)
      .def("fit_incrementally",
           py::overload_cast<const geometry::pointsNd<Dim>&, const common::valuesd&, double, int>(
               &interpolant<Dim>::fit_incrementally),
           "points"_a, "values"_a, "absolute_tolerance"_a, "max_iter"_a = 100)
      .def("fit_incrementally_with_grad",
           py::overload_cast<const geometry::pointsNd<Dim>&, const geometry::pointsNd<Dim>&,
                             const common::valuesd&, double, double, int>(
               &interpolant<Dim>::fit_incrementally),
           "points"_a, "grad_points"_a, "values"_a, "absolute_tolerance"_a,
           "grad_absolute_tolerance"_a, "max_iter"_a = 100)
      .def("fit_inequality",
           py::overload_cast<const geometry::pointsNd<Dim>&, const common::valuesd&,
                             const common::valuesd&, const common::valuesd&, double, int>(
               &interpolant<Dim>::fit_inequality),
           "points"_a, "values"_a, "values_lb"_a, "values_ub"_a, "absolute_tolerance"_a,
           "max_iter"_a = 100);

  py::class_<point_cloud::distance_filter<Dim>>(m, "DistanceFilter")
      .def(py::init<const geometry::pointsNd<Dim>&, double>(), "points"_a, "distance"_a)
      .def_property_readonly("filtered_indices",
                             &point_cloud::distance_filter<Dim>::filtered_indices);
}

PYBIND11_MODULE(_core, m) {
  auto p1 = m.def_submodule("p1");
  auto p2 = m.def_submodule("p2");
  auto p3 = m.def_submodule("p3");

  define_module<1>(p1);
  define_module<2>(p2);
  define_module<3>(p3);

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

  py::class_<point_cloud::sdf_data_generator>(m, "SdfDataGenerator")
      .def(
          py::init<const geometry::points3d&, const geometry::vectors3d&, double, double, double>(),
          "points"_a, "normals"_a, "min_distance"_a, "max_distance"_a, "multiplication"_a = 2.0)
      .def_property_readonly("sdf_points", &point_cloud::sdf_data_generator::sdf_points)
      .def_property_readonly("sdf_values", &point_cloud::sdf_data_generator::sdf_values);

  py::class_<isosurface::field_function>(m, "_FieldFunction");

  py::class_<isosurface::rbf_field_function, isosurface::field_function>(m, "RbfFieldFunction")
      .def(py::init<interpolant<3>&>(), "interpolant"_a);

  py::class_<isosurface::rbf_field_function_25d, isosurface::field_function>(m,
                                                                             "RbfFieldFunction25d")
      .def(py::init<interpolant<2>&>(), "interpolant"_a);

  py::class_<isosurface::isosurface>(m, "Isosurface")
      .def(py::init<const geometry::bbox3d&, double>(), "bbox"_a, "resolution"_a)
      .def("generate", &isosurface::isosurface::generate, "field_fn"_a, "isovalue"_a = 0.0,
           "refine"_a = true)
      .def("generate_from_seed_points", &isosurface::isosurface::generate_from_seed_points,
           "seed_points"_a, "field_fn"_a, "isovalue"_a = 0.0, "refine"_a = true);

  py::class_<isosurface::surface>(m, "Surface")
      .def("export_obj", &isosurface::surface::export_obj, "filename"_a)
      .def_property_readonly("faces", &isosurface::surface::faces)
      .def_property_readonly("vertices", &isosurface::surface::vertices);

  m.attr("__version__") = xstr(POLATORY_VERSION);
}
