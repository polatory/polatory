#include <Python.h>
#undef _GNU_SOURCE
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <polatory/kriging.hpp>
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
  using Bbox = geometry::bboxNd<Dim>;
  using DistanceFilter = point_cloud::distance_filter<Dim>;
  using Interpolant = interpolant<Dim>;
  using Model = model<Dim>;
  using Points = geometry::pointsNd<Dim>;
  using RbfProxy = rbf::rbf_proxy<Dim>;
  using Variogram = kriging::variogram<Dim>;
  using VariogramCalculator = kriging::variogram_calculator<Dim>;
  using VariogramFitting = kriging::variogram_fitting<Dim>;
  using VariogramSet = kriging::variogram_set<Dim>;

  py::class_<Bbox>(m, "Bbox")
      .def(py::init<>())
      .def(py::init<const Points&, const Points>(), "min"_a, "max"_a)
      .def_static("from_points", &bbox_from_points<Dim>, "points"_a)
      .def_property_readonly("is_empty", &Bbox::is_empty)
      .def_property_readonly("min", &Bbox::min)
      .def_property_readonly("max", &Bbox::max);

  py::class_<RbfProxy>(m, "Rbf")
      .def_property("anisotropy", &RbfProxy::anisotropy, &RbfProxy::set_anisotropy)
      .def_property_readonly("cpd_order", &RbfProxy::cpd_order)
      .def_property_readonly("is_covariance_function", &RbfProxy::is_covariance_function)
      .def_property_readonly("num_parameters", &RbfProxy::num_parameters)
      .def_property("parameters", &RbfProxy::parameters, &RbfProxy::set_parameters)
      .def_property_readonly("short_name", &RbfProxy::short_name)
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
      .def(py::init<RbfProxy, int>(), "rbf"_a, "poly_degree"_a = Model::kMinRequiredPolyDegree)
      .def(py::init<std::vector<RbfProxy>, int>(), "rbfs"_a,
           "poly_degree"_a = Model::kMinRequiredPolyDegree)
      .def_readonly_static("MIN_REQUIRED_POLY_DEGREE", &Model::kMinRequiredPolyDegree)
      .def_property_readonly("cpd_order", &Model::cpd_order)
      .def_property_readonly("description", &Model::description)
      .def_property("nugget", &Model::nugget, &Model::set_nugget)
      .def_property_readonly("num_parameters", &Model::num_parameters)
      .def_property("parameters", &Model::parameters, &Model::set_parameters)
      .def_property_readonly("poly_basis_size", &Model::poly_basis_size)
      .def_property_readonly("poly_degree", &Model::poly_degree)
      .def_property_readonly(
          "rbfs", static_cast<const std::vector<RbfProxy>& (Model::*)() const>(&Model::rbfs))
      .def_static("load", &Model::load, "filename"_a)
      .def("save", &Model::save, "filename"_a);

  py::class_<Interpolant>(m, "Interpolant")
      .def(py::init<const Model&>(), "model"_a)
      .def_property_readonly("bbox", &Interpolant::bbox)
      .def_property_readonly("centers", &Interpolant::centers)
      .def_property_readonly("grad_centers", &Interpolant::grad_centers)
      .def_property_readonly("model", &Interpolant::model)
      .def_property_readonly("weights", &Interpolant::weights)
      .def("evaluate", py::overload_cast<const Points&>(&Interpolant::evaluate), "points"_a)
      .def("evaluate", py::overload_cast<const Points&, const Points&>(&Interpolant::evaluate),
           "points"_a, "grad_points"_a)
      .def("fit",
           py::overload_cast<const Points&, const vectord&, double, int, const Interpolant*>(
               &Interpolant::fit),
           "points"_a, "values"_a, "absolute_tolerance"_a, "max_iter"_a = 100,
           "initial"_a = nullptr)
      .def("fit",
           py::overload_cast<const Points&, const Points&, const vectord&, double, double, int,
                             const Interpolant*>(&Interpolant::fit),
           "points"_a, "grad_points"_a, "values"_a, "absolute_tolerance"_a,
           "grad_absolute_tolerance"_a, "max_iter"_a = 100, "initial"_a = nullptr)
      .def("fit_incrementally",
           py::overload_cast<const Points&, const vectord&, double, int>(
               &Interpolant::fit_incrementally),
           "points"_a, "values"_a, "absolute_tolerance"_a, "max_iter"_a = 100)
      .def("fit_incrementally",
           py::overload_cast<const Points&, const Points&, const vectord&, double, double, int>(
               &Interpolant::fit_incrementally),
           "points"_a, "grad_points"_a, "values"_a, "absolute_tolerance"_a,
           "grad_absolute_tolerance"_a, "max_iter"_a = 100)
      .def("fit_inequality",
           py::overload_cast<const Points&, const vectord&, const vectord&, const vectord&, double,
                             int>(&Interpolant::fit_inequality),
           "points"_a, "values"_a, "values_lb"_a, "values_ub"_a, "absolute_tolerance"_a,
           "max_iter"_a = 100)
      .def_static("load", &Interpolant::load, "filename"_a)
      .def("save", &Interpolant::save, "filename"_a);

  py::class_<DistanceFilter>(m, "DistanceFilter")
      .def(py::init<const Points&, double>(), "points"_a, "distance"_a)
      .def_property_readonly("filtered_indices", &DistanceFilter::filtered_indices);

  py::class_<Variogram>(m, "Variogram")
      .def_property_readonly("bin_distance", &Variogram::bin_distance)
      .def_property_readonly("bin_gamma", &Variogram::bin_gamma)
      .def_property_readonly("bin_num_pairs", &Variogram::bin_num_pairs)
      .def_property_readonly("direction", &Variogram::direction)
      .def_property_readonly("num_bins", &Variogram::num_bins)
      .def_property_readonly("num_pairs", &Variogram::num_pairs);

  py::class_<VariogramCalculator>(m, "VariogramCalculator")
      .def(py::init<double, index_t>(), "lag_distance"_a, "num_lags"_a)
      .def_readonly_static("AUTOMATIC_ANGLE_TOLERANCE",
                           &VariogramCalculator::kAutomaticAngleTolerance)
      .def_readonly_static("AUTOMATIC_LAG_TOLERANCE", &VariogramCalculator::kAutomaticLagTolerance)
      .def_readonly_static("ISOTROPIC_DIRECTIONS", &VariogramCalculator::kIsotropicDirections)
      .def_readonly_static("ANISOTROPIC_DIRECTIONS", &VariogramCalculator::kAnisotropicDirections)
      .def_property("angle_tolerance", &VariogramCalculator::angle_tolerance,
                    &VariogramCalculator::set_angle_tolerance)
      .def_property("directions", &VariogramCalculator::directions,
                    &VariogramCalculator::set_directions)
      .def_property("lag_tolerance", &VariogramCalculator::lag_tolerance,
                    &VariogramCalculator::set_lag_tolerance)
      .def("calculate", &VariogramCalculator::calculate, "points"_a, "values"_a);

  py::class_<VariogramFitting>(m, "VariogramFitting")
      .def(py::init<const VariogramSet&, const Model&, const kriging::weight_function&, bool>(),
           "variog_set"_a, "model"_a,
           "weight_fn"_a = kriging::weight_function::kNumPairsOverDistanceSquared,
           "fit_anisotropy"_a = true)
      .def_property_readonly("brief_report", &VariogramFitting::brief_report)
      .def_property_readonly("full_report", &VariogramFitting::full_report)
      .def_property_readonly("final_cost", &VariogramFitting::final_cost)
      .def_property_readonly("model", &VariogramFitting::model);

  py::class_<VariogramSet>(m, "VariogramSet")
      .def_property_readonly("num_pairs", &VariogramSet::num_pairs)
      .def_property_readonly("num_variograms", &VariogramSet::num_variograms)
      .def_property_readonly("variograms", &VariogramSet::variograms)
      .def_static("load", &VariogramSet::load, "filename"_a)
      .def("save", &VariogramSet::save, "filename"_a);

  m.def("cross_validate", &kriging::cross_validate<Dim>, "model"_a, "points"_a, "values"_a,
        "set_ids"_a, "absolute_tolerance"_a, "max_iter"_a = 100);

  m.def("detrend", &kriging::detrend<Dim>, "points"_a, "values"_a, "degree"_a);
}

PYBIND11_MODULE(_core, m) {
  using NormalEstimator = point_cloud::normal_estimator;

  py::class_<NormalEstimator>(m, "NormalEstimator")
      .def(py::init<const geometry::points3d&>(), "points"_a)
      .def_property_readonly("normals", &NormalEstimator::normals)
      .def_property_readonly("plane_factors", &NormalEstimator::plane_factors)
      .def("estimate_with_knn",
           static_cast<NormalEstimator& (NormalEstimator::*)(index_t)&>(
               &NormalEstimator::estimate_with_knn),
           "k"_a)
      .def("estimate_with_knn",
           static_cast<NormalEstimator& (NormalEstimator::*)(const std::vector<index_t>&)&>(
               &NormalEstimator::estimate_with_knn),
           "ks"_a)
      .def("estimate_with_radius",
           static_cast<NormalEstimator& (NormalEstimator::*)(double)&>(
               &NormalEstimator::estimate_with_radius),
           "radius"_a)
      .def("estimate_with_radius",
           static_cast<NormalEstimator& (NormalEstimator::*)(const std::vector<double>&)&>(
               &NormalEstimator::estimate_with_radius),
           "radii"_a)
      .def("filter_by_plane_factor",
           static_cast<NormalEstimator& (NormalEstimator::*)(double)&>(
               &NormalEstimator::filter_by_plane_factor),
           "threshold"_a = 1.8)
      .def("orient_toward_direction",
           static_cast<NormalEstimator& (NormalEstimator::*)(const geometry::vector3d&)&>(
               &NormalEstimator::orient_toward_direction),
           "direction"_a)
      .def("orient_toward_point",
           static_cast<NormalEstimator& (NormalEstimator::*)(const geometry::point3d&)&>(
               &NormalEstimator::orient_toward_point),
           "point"_a)
      .def("orient_closed_surface",
           static_cast<NormalEstimator& (NormalEstimator::*)(index_t)&>(
               &NormalEstimator::orient_closed_surface),
           "k"_a = 100);

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

  py::class_<kriging::weight_function>(m, "WeightFunction")
      .def(py::init<double, double, double>(), "exp_distance"_a = 0.0, "exp_model_gamma"_a = 0.0,
           "exp_num_pairs"_a = 0.0)
      .def_readonly_static("NUM_PAIRS", &kriging::weight_function::kNumPairs)
      .def_readonly_static("NUM_PAIRS_OVER_DISTANCE_SQUARED",
                           &kriging::weight_function::kNumPairsOverDistanceSquared)
      .def_readonly_static("NUM_PAIRS_OVER_MODEL_GAMMA_SQUARED",
                           &kriging::weight_function::kNumPairsOverModelGammaSquared)
      .def_readonly_static("ONE", &kriging::weight_function::kOne)
      .def_readonly_static("ONE_OVER_DISTANCE_SQUARED",
                           &kriging::weight_function::kOneOverDistanceSquared)
      .def_readonly_static("ONE_OVER_MODEL_GAMMA_SQUARED",
                           &kriging::weight_function::kOneOverModelGammaSquared);

  m.attr("__version__") = xstr(POLATORY_VERSION);

  auto one = m.def_submodule("one");
  auto two = m.def_submodule("two");
  auto three = m.def_submodule("three");

  define_module<1>(one);
  define_module<2>(two);
  define_module<3>(three);
}
