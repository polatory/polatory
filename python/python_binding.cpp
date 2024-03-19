#include <Python.h>
#undef _GNU_SOURCE
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
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

template <int Dim>
std::unique_ptr<kriging::variogram_fitting<Dim>> make_variogram_fitting(
    const kriging::empirical_variogram<Dim>& emp_variog, const model<Dim>& model,
    const std::string& weight) {
  const kriging::weight_function* weight_fn{nullptr};

  if (weight == "num_pairs") {
    weight_fn = &kriging::weight_functions::num_pairs;
  } else if (weight == "num_pairs_over_distance_squared") {
    weight_fn = &kriging::weight_functions::num_pairs_over_distance_squared;
  } else if (weight == "num_pairs_over_model_gamma_squared") {
    weight_fn = &kriging::weight_functions::num_pairs_over_model_gamma_squared;
  } else if (weight == "one") {
    weight_fn = &kriging::weight_functions::one;
  } else if (weight == "one_over_distance_squared") {
    weight_fn = &kriging::weight_functions::one_over_distance_squared;
  } else if (weight == "one_over_model_gamma_squared") {
    weight_fn = &kriging::weight_functions::one_over_model_gamma_squared;
  } else {
    throw std::invalid_argument("Unknown weight function: " + weight);
  }

  return std::make_unique<kriging::variogram_fitting<Dim>>(emp_variog, model, *weight_fn);
}

template <int Dim, class Rbf>
void define_rbf(py::module& m, const std::string& name) {
  py::class_<Rbf, rbf::rbf_proxy<Dim>>(m, name.c_str())
      .def(py::init<const std::vector<double>&>(), "params"_a);
}

template <int Dim>
void define_module(py::module& m) {
  using Bbox = geometry::bboxNd<Dim>;
  using DistanceFilter = point_cloud::distance_filter<Dim>;
  using EmpiricalVariogram = kriging::empirical_variogram<Dim>;
  using Interpolant = interpolant<Dim>;
  using Model = model<Dim>;
  using Points = geometry::pointsNd<Dim>;
  using RbfProxy = rbf::rbf_proxy<Dim>;
  using VariogramFitting = kriging::variogram_fitting<Dim>;

  py::class_<Bbox>(m, "Bbox")
      .def(py::init<>())
      .def(py::init<const Points&, const Points>(), "min"_a, "max"_a)
      .def_static("from_points", &bbox_from_points<Dim>, "points"_a)
      .def_property_readonly("min", &Bbox::min)
      .def_property_readonly("max", &Bbox::max);

  py::class_<RbfProxy>(m, "Rbf")
      .def_property("anisotropy", &RbfProxy::anisotropy, &RbfProxy::set_anisotropy)
      .def_property_readonly("cpd_order", &RbfProxy::cpd_order)
      .def_property_readonly("num_parameters", &RbfProxy::num_parameters)
      .def_property("parameters", &RbfProxy::parameters, &RbfProxy::set_parameters)
      .def("evaluate", &RbfProxy::evaluate, "diff"_a)
      .def("evaluate_gradient", &RbfProxy::evaluate_gradient, "diff"_a)
      .def("evaluate_hessian", &RbfProxy::evaluate_hessian, "diff"_a);

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

  py::class_<Model>(m, "Model")
      .def(py::init<RbfProxy, int>(), "rbf"_a, "poly_degree"_a)
      .def(py::init<std::vector<RbfProxy>, int>(), "rbfs"_a, "poly_degree"_a)
      .def_property_readonly("cpd_order", &Model::cpd_order)
      .def_property("nugget", &Model::nugget, &Model::set_nugget)
      .def_property_readonly("num_parameters", &Model::num_parameters)
      .def_property("parameters", &Model::parameters, &Model::set_parameters)
      .def_property_readonly("poly_basis_size", &Model::poly_basis_size)
      .def_property_readonly("poly_degree", &Model::poly_degree)
      .def_property_readonly("rbfs", &Model::rbfs);

  py::class_<Interpolant>(m, "Interpolant")
      .def(py::init<const Model&>(), "model"_a)
      .def_property_readonly("centers", &Interpolant::centers)
      .def_property_readonly("weights", &Interpolant::weights)
      .def("evaluate", &Interpolant::evaluate, "points"_a)
      .def("fit",
           py::overload_cast<const Points&, const common::valuesd&, double, int>(&Interpolant::fit),
           "points"_a, "values"_a, "absolute_tolerance"_a, "max_iter"_a = 100)
      .def("fit_with_grad",
           py::overload_cast<const Points&, const Points&, const common::valuesd&, double, double,
                             int>(&Interpolant::fit),
           "points"_a, "grad_points"_a, "values"_a, "absolute_tolerance"_a,
           "grad_absolute_tolerance"_a, "max_iter"_a = 100)
      .def("fit_incrementally",
           py::overload_cast<const Points&, const common::valuesd&, double, int>(
               &Interpolant::fit_incrementally),
           "points"_a, "values"_a, "absolute_tolerance"_a, "max_iter"_a = 100)
      .def("fit_incrementally_with_grad",
           py::overload_cast<const Points&, const Points&, const common::valuesd&, double, double,
                             int>(&Interpolant::fit_incrementally),
           "points"_a, "grad_points"_a, "values"_a, "absolute_tolerance"_a,
           "grad_absolute_tolerance"_a, "max_iter"_a = 100)
      .def("fit_inequality",
           py::overload_cast<const Points&, const common::valuesd&, const common::valuesd&,
                             const common::valuesd&, double, int>(&Interpolant::fit_inequality),
           "points"_a, "values"_a, "values_lb"_a, "values_ub"_a, "absolute_tolerance"_a,
           "max_iter"_a = 100);

  py::class_<DistanceFilter>(m, "DistanceFilter")
      .def(py::init<const Points&, double>(), "points"_a, "distance"_a)
      .def_property_readonly("filtered_indices", &DistanceFilter::filtered_indices);

  py::class_<EmpiricalVariogram>(m, "EmpiricalVariogram")
      .def(py::init<const Points&, const common::valuesd&, double, index_t>(), "points"_a,
           "values"_a, "bin_width"_a, "num_bins"_a)
      .def_property_readonly("bin_distance", &EmpiricalVariogram::bin_distance)
      .def_property_readonly("bin_gamma", &EmpiricalVariogram::bin_gamma)
      .def_property_readonly("bin_num_pairs", &EmpiricalVariogram::bin_num_pairs);

  py::class_<VariogramFitting>(m, "VariogramFitting")
      .def(py::init(&make_variogram_fitting<Dim>), "emp_variog"_a, "model"_a,
           "weight"_a = "num_pairs_over_distance_squared")
      .def_property_readonly("parameters", &VariogramFitting::parameters);
}

PYBIND11_MODULE(_core, m) {
  auto one = m.def_submodule("one");
  auto two = m.def_submodule("two");
  auto three = m.def_submodule("three");

  define_module<1>(one);
  define_module<2>(two);
  define_module<3>(three);

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
