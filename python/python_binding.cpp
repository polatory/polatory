#include <Python.h>
#undef _GNU_SOURCE
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <polatory/kriging.hpp>
#include <polatory/polatory.hpp>
#include <string>
#include <vector>

#define xstr(s) str(s)
#define str(s) #s

using namespace polatory;
namespace py = pybind11;
using namespace py::literals;

static constexpr double kInfinity = std::numeric_limits<double>::infinity();

template <int Dim>
geometry::Bbox<Dim> bbox_from_points(const geometry::Points<Dim>& points) {
  return geometry::Bbox<Dim>::from_points(points);
}

template <int Dim, class Rbf>
void define_rbf(py::module& m, const std::string& name) {
  py::class_<Rbf, rbf::Rbf<Dim>>(m, name.c_str())
      .def(py::init<const std::vector<double>&>(), "params"_a);
}

template <int Dim>
void define_module(py::module& m) {
  using Bbox = geometry::Bbox<Dim>;
  using DistanceFilter = point_cloud::DistanceFilter<Dim>;
  using Interpolant = Interpolant<Dim>;
  using Model = Model<Dim>;
  using Points = geometry::Points<Dim>;
  using Rbf = rbf::Rbf<Dim>;
  using Variogram = kriging::Variogram<Dim>;
  using VariogramCalculator = kriging::VariogramCalculator<Dim>;
  using VariogramFitting = kriging::VariogramFitting<Dim>;
  using VariogramSet = kriging::VariogramSet<Dim>;

  py::class_<Bbox>(m, "Bbox")
      .def(py::init<>())
      .def(py::init<const Points&, const Points>(), "min"_a, "max"_a)
      .def_static("from_points", &bbox_from_points<Dim>, "points"_a)
      .def_property_readonly("is_empty", &Bbox::is_empty)
      .def_property_readonly("min", &Bbox::min)
      .def_property_readonly("max", &Bbox::max);

  py::class_<Rbf>(m, "Rbf")
      .def_property("anisotropy", &Rbf::anisotropy, &Rbf::set_anisotropy)
      .def_property_readonly("cpd_order", &Rbf::cpd_order)
      .def_property_readonly("is_covariance_function", &Rbf::is_covariance_function)
      .def_property_readonly("num_parameters", &Rbf::num_parameters)
      .def_property("parameters", &Rbf::parameters, &Rbf::set_parameters)
      .def_property_readonly("short_name", &Rbf::short_name)
      .def("evaluate", &Rbf::evaluate, "diff"_a)
      .def("evaluate_gradient", &Rbf::evaluate_gradient, "diff"_a)
      .def("evaluate_hessian", &Rbf::evaluate_hessian, "diff"_a);

  define_rbf<Dim, rbf::Biharmonic2D<Dim>>(m, "Biharmonic2D");
  define_rbf<Dim, rbf::Biharmonic3D<Dim>>(m, "Biharmonic3D");
  define_rbf<Dim, rbf::CovCubic<Dim>>(m, "CovCubic");
  define_rbf<Dim, rbf::CovExponential<Dim>>(m, "CovExponential");
  define_rbf<Dim, rbf::CovGaussian<Dim>>(m, "CovGaussian");
  define_rbf<Dim, rbf::CovGeneralizedCauchy3<Dim>>(m, "CovGeneralizedCauchy3");
  define_rbf<Dim, rbf::CovGeneralizedCauchy5<Dim>>(m, "CovGeneralizedCauchy5");
  define_rbf<Dim, rbf::CovGeneralizedCauchy7<Dim>>(m, "CovGeneralizedCauchy7");
  define_rbf<Dim, rbf::CovGeneralizedCauchy9<Dim>>(m, "CovGeneralizedCauchy9");
  define_rbf<Dim, rbf::CovSpherical<Dim>>(m, "CovSpherical");
  define_rbf<Dim, rbf::CovSpheroidal3<Dim>>(m, "CovSpheroidal3");
  define_rbf<Dim, rbf::CovSpheroidal5<Dim>>(m, "CovSpheroidal5");
  define_rbf<Dim, rbf::CovSpheroidal7<Dim>>(m, "CovSpheroidal7");
  define_rbf<Dim, rbf::CovSpheroidal9<Dim>>(m, "CovSpheroidal9");
  define_rbf<Dim, rbf::Triharmonic2D<Dim>>(m, "Triharmonic2D");
  define_rbf<Dim, rbf::Triharmonic3D<Dim>>(m, "Triharmonic3D");

  py::class_<Model>(m, "Model")
      .def(py::init<Rbf, int>(), "rbf"_a, "poly_degree"_a = Model::kMinRequiredPolyDegree)
      .def(py::init<std::vector<Rbf>, int>(), "rbfs"_a,
           "poly_degree"_a = Model::kMinRequiredPolyDegree)
      .def_readonly_static("MIN_REQUIRED_POLY_DEGREE", &Model::kMinRequiredPolyDegree)
      .def_property_readonly("cpd_order", &Model::cpd_order)
      .def_property_readonly("description", &Model::description)
      .def_property("nugget", &Model::nugget, &Model::set_nugget)
      .def_property_readonly("num_parameters", &Model::num_parameters)
      .def_property("parameters", &Model::parameters, &Model::set_parameters)
      .def_property_readonly("poly_basis_size", &Model::poly_basis_size)
      .def_property_readonly("poly_degree", &Model::poly_degree)
      .def_property_readonly("rbfs",
                             static_cast<const std::vector<Rbf>& (Model::*)() const>(&Model::rbfs))
      .def_static("load", &Model::load, "filename"_a)
      .def("save", &Model::save, "filename"_a);

  py::class_<Interpolant>(m, "Interpolant")
      .def(py::init<const Model&>(), "model"_a)
      .def_property_readonly("bbox", &Interpolant::bbox)
      .def_property_readonly("centers", &Interpolant::centers)
      .def_property_readonly("grad_centers", &Interpolant::grad_centers)
      .def_property_readonly("model", &Interpolant::model)
      .def_property_readonly("weights", &Interpolant::weights)
      .def("evaluate", py::overload_cast<const Points&, double>(&Interpolant::evaluate), "points"_a,
           "accuracy"_a = kInfinity)
      .def("evaluate",
           py::overload_cast<const Points&, const Points&, double, double>(&Interpolant::evaluate),
           "points"_a, "grad_points"_a, "accuracy"_a = kInfinity, "grad_accuracy"_a = kInfinity)
      .def("fit",
           py::overload_cast<const Points&, const VecX&, double, int, double, const Interpolant*>(
               &Interpolant::fit),
           "points"_a, "values"_a, "tolerance"_a, "max_iter"_a = 100, "accuracy"_a = kInfinity,
           "initial"_a = nullptr)
      .def("fit",
           py::overload_cast<const Points&, const Points&, const VecX&, double, double, int, double,
                             double, const Interpolant*>(&Interpolant::fit),
           "points"_a, "grad_points"_a, "values"_a, "tolerance"_a, "grad_tolerance"_a,
           "max_iter"_a = 100, "accuracy"_a = kInfinity, "grad_accuracy"_a = kInfinity,
           "initial"_a = nullptr)
      .def("fit_incrementally",
           py::overload_cast<const Points&, const VecX&, double, int, double>(
               &Interpolant::fit_incrementally),
           "points"_a, "values"_a, "tolerance"_a, "max_iter"_a = 100, "accuracy"_a = kInfinity)
      .def("fit_incrementally",
           py::overload_cast<const Points&, const Points&, const VecX&, double, double, int, double,
                             double>(&Interpolant::fit_incrementally),
           "points"_a, "grad_points"_a, "values"_a, "tolerance"_a, "grad_tolerance"_a,
           "max_iter"_a = 100, "accuracy"_a = kInfinity, "grad_accuracy"_a = kInfinity)
      .def("fit_inequality",
           py::overload_cast<const Points&, const VecX&, const VecX&, const VecX&, double, int,
                             double, const Interpolant*>(&Interpolant::fit_inequality),
           "points"_a, "values"_a, "values_lb"_a, "values_ub"_a, "tolerance"_a, "max_iter"_a = 100,
           "accuracy"_a = kInfinity, "initial"_a = nullptr)
      .def_static("load", &Interpolant::load, "filename"_a)
      .def("save", &Interpolant::save, "filename"_a);

  py::class_<DistanceFilter>(m, "DistanceFilter")
      .def(py::init<const Points&>(), "points"_a)
      .def_property_readonly("filtered_indices", &DistanceFilter::filtered_indices)
      .def("filter", py::overload_cast<double>(&DistanceFilter::filter), "distance"_a);

  py::class_<Variogram>(m, "Variogram")
      .def_property_readonly("bin_distance", &Variogram::bin_distance)
      .def_property_readonly("bin_gamma", &Variogram::bin_gamma)
      .def_property_readonly("bin_num_pairs", &Variogram::bin_num_pairs)
      .def_property_readonly("direction", &Variogram::direction)
      .def_property_readonly("num_bins", &Variogram::num_bins)
      .def_property_readonly("num_pairs", &Variogram::num_pairs)
      .def("back_transform", &Variogram::back_transform, "t"_a);

  py::class_<VariogramCalculator>(m, "VariogramCalculator")
      .def(py::init<double, Index>(), "lag_distance"_a, "num_lags"_a)
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
      .def(py::init<const VariogramSet&, const Model&, const kriging::WeightFunction&, bool>(),
           "variog_set"_a, "model"_a,
           "weight_fn"_a = kriging::WeightFunction::kNumPairsOverDistanceSquared,
           "fit_anisotropy"_a = true)
      .def_property_readonly("brief_report", &VariogramFitting::brief_report)
      .def_property_readonly("full_report", &VariogramFitting::full_report)
      .def_property_readonly("final_cost", &VariogramFitting::final_cost)
      .def_property_readonly("model", &VariogramFitting::model);

  py::class_<VariogramSet>(m, "VariogramSet")
      .def_property_readonly("num_pairs", &VariogramSet::num_pairs)
      .def_property_readonly("num_variograms", &VariogramSet::num_variograms)
      .def_property_readonly("variograms", &VariogramSet::variograms)
      .def("back_transform", &VariogramSet::back_transform, "t"_a)
      .def_static("load", &VariogramSet::load, "filename"_a)
      .def("save", &VariogramSet::save, "filename"_a);

  m.def("cross_validate", &kriging::cross_validate<Dim>, "model"_a, "points"_a, "values"_a,
        "set_ids"_a, "tolerance"_a, "max_iter"_a = 100, "accuracy"_a = kInfinity);

  m.def("detrend", &kriging::detrend<Dim>, "points"_a, "values"_a, "degree"_a);
}

PYBIND11_MODULE(_core, m) {
  using Bbox = geometry::Bbox3;
  using Mat = Mat3;
  using NormalEstimator = point_cloud::NormalEstimator;

  py::class_<NormalEstimator>(m, "NormalEstimator")
      .def(py::init<const geometry::Points3&>(), "points"_a)
      .def_property_readonly("normals", &NormalEstimator::normals)
      .def_property_readonly("plane_factors", &NormalEstimator::plane_factors)
      .def("estimate_with_knn",
           static_cast<NormalEstimator& (NormalEstimator::*)(Index)&>(
               &NormalEstimator::estimate_with_knn),
           "k"_a)
      .def("estimate_with_knn",
           static_cast<NormalEstimator& (NormalEstimator::*)(const std::vector<Index>&)&>(
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
           static_cast<NormalEstimator& (NormalEstimator::*)(const geometry::Vector3&)&>(
               &NormalEstimator::orient_toward_direction),
           "direction"_a)
      .def("orient_toward_point",
           static_cast<NormalEstimator& (NormalEstimator::*)(const geometry::Point3&)&>(
               &NormalEstimator::orient_toward_point),
           "point"_a)
      .def("orient_closed_surface",
           static_cast<NormalEstimator& (NormalEstimator::*)(Index)&>(
               &NormalEstimator::orient_closed_surface),
           "k"_a = 100);

  py::class_<point_cloud::SdfDataGenerator>(m, "SdfDataGenerator")
      .def(py::init<const geometry::Points3&, const geometry::Vectors3&, double, double,
                    const Mat&>(),
           "points"_a, "normals"_a, "min_distance"_a, "max_distance"_a, "aniso"_a = Mat::Identity())
      .def_property_readonly("sdf_points", &point_cloud::SdfDataGenerator::sdf_points)
      .def_property_readonly("sdf_values", &point_cloud::SdfDataGenerator::sdf_values);

  py::class_<isosurface::FieldFunction>(m, "_FieldFunction");

  py::class_<isosurface::RbfFieldFunction, isosurface::FieldFunction>(m, "RbfFieldFunction")
      .def(py::init<Interpolant<3>&, double, double>(), "interpolant"_a, "accuracy"_a = kInfinity,
           "grad_accuracy"_a = kInfinity);

  py::class_<isosurface::RbfFieldFunction25D, isosurface::FieldFunction>(m, "RbfFieldFunction25D")
      .def(py::init<Interpolant<2>&, double, double>(), "interpolant"_a, "accuracy"_a = kInfinity,
           "grad_accuracy"_a = kInfinity);

  py::class_<isosurface::Isosurface>(m, "Isosurface")
      .def(py::init<const Bbox&, double, const Mat&>(), "bbox"_a, "resolution"_a,
           "aniso"_a = Mat::Identity())
      .def("generate", &isosurface::Isosurface::generate, "field_fn"_a, "isovalue"_a = 0.0,
           "refine"_a = 1)
      .def("generate_from_seed_points", &isosurface::Isosurface::generate_from_seed_points,
           "seed_points"_a, "field_fn"_a, "isovalue"_a = 0.0, "refine"_a = 1);

  py::class_<isosurface::Mesh>(m, "Mesh")
      .def("export_obj", &isosurface::Mesh::export_obj, "filename"_a)
      .def_property_readonly("faces", &isosurface::Mesh::faces)
      .def_property_readonly("vertices", &isosurface::Mesh::vertices);

  py::class_<kriging::NormalScoreTransformation>(m, "NormalScoreTransformation")
      .def(py::init<int>(), "order"_a = 30)
      .def("transform", &kriging::NormalScoreTransformation::transform, "z"_a)
      .def("back_transform", &kriging::NormalScoreTransformation::back_transform, "y"_a);

  py::class_<kriging::WeightFunction>(m, "WeightFunction")
      .def(py::init<double, double, double>(), "exp_distance"_a = 0.0, "exp_model_gamma"_a = 0.0,
           "exp_num_pairs"_a = 0.0)
      .def_readonly_static("NUM_PAIRS", &kriging::WeightFunction::kNumPairs)
      .def_readonly_static("NUM_PAIRS_OVER_DISTANCE_SQUARED",
                           &kriging::WeightFunction::kNumPairsOverDistanceSquared)
      .def_readonly_static("NUM_PAIRS_OVER_MODEL_GAMMA_SQUARED",
                           &kriging::WeightFunction::kNumPairsOverModelGammaSquared)
      .def_readonly_static("ONE", &kriging::WeightFunction::kOne)
      .def_readonly_static("ONE_OVER_DISTANCE_SQUARED",
                           &kriging::WeightFunction::kOneOverDistanceSquared)
      .def_readonly_static("ONE_OVER_MODEL_GAMMA_SQUARED",
                           &kriging::WeightFunction::kOneOverModelGammaSquared);

  m.attr("__version__") = xstr(POLATORY_VERSION);

  auto one = m.def_submodule("one");
  auto two = m.def_submodule("two");
  auto three = m.def_submodule("three");

  define_module<1>(one);
  define_module<2>(two);
  define_module<3>(three);
}
