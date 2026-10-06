#include <Python.h>
#undef _GNU_SOURCE
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <format>
#include <jizai/geostats.hpp>
#include <jizai/jizai.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#define str(s) #s
#define xstr(s) str(s)

#define VISIT(ANY, EXPR) [](ANY& self) { return self.visit([](auto& x) { return EXPR; }); }

using namespace jizai;
namespace py = pybind11;
using namespace py::literals;

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

template <class F>
auto with_dim(Index dim, const F& f) {
  switch (dim) {
    case 1:
      return f.template operator()<1>();
    case 2:
      return f.template operator()<2>();
    case 3:
      return f.template operator()<3>();
    default:
      throw py::value_error(std::format("dimension must be 1, 2, or 3, got {}", dim));
  }
}

template <template <int> class T>
class AnyDim {
 public:
  template <int Dim>
  explicit AnyDim(T<Dim> t) : t_(std::move(t)) {}

  template <std::size_t I, class... Args>
  explicit AnyDim(std::in_place_index_t<I> i, Args&&... args)
      : t_(i, std::forward<Args>(args)...) {}

  int dim() const { return static_cast<int>(t_.index()) + 1; }

  template <int Dim>
  T<Dim>& get() {
    check_dim(Dim);
    return std::get<Dim - 1>(t_);
  }

  template <int Dim>
  const T<Dim>& get() const {
    check_dim(Dim);
    return std::get<Dim - 1>(t_);
  }

  template <class F>
  auto visit(const F& f) {
    return std::visit(f, t_);
  }

  template <class F>
  auto visit(const F& f) const {
    return std::visit(f, t_);
  }

 private:
  void check_dim(int dim) const {
    if (dim != this->dim()) {
      throw py::value_error(
          std::format("expected a {}-dimensional object, got {}", dim, this->dim()));
    }
  }

  std::variant<T<1>, T<2>, T<3>> t_;
};

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

using AnyBbox = AnyDim<geometry::Bbox>;
using AnyDistanceFilter = AnyDim<point_cloud::DistanceFilter>;
using AnyInterpolant = AnyDim<Interpolant>;
using AnyModel = AnyDim<Model>;
using AnyRbf = AnyDim<rbf::Rbf>;
using AnyVariogram = AnyDim<geostats::Variogram>;
using AnyVariogramCalculator = AnyDim<geostats::VariogramCalculator>;
using AnyVariogramFitting = AnyDim<geostats::VariogramFitting>;
using AnyVariogramSet = AnyDim<geostats::VariogramSet>;

template <template <int> class R>
class AnyRbfOf : public AnyRbf {
 public:
  template <int Dim>
  explicit AnyRbfOf(R<Dim> rbf) : AnyRbf(std::move(rbf)) {}
};

template <int Dim, template <int> class T>
const T<Dim>* get_ptr(const AnyDim<T>* any) {
  return any != nullptr ? &any->template get<Dim>() : nullptr;
}

Index point_dim(const Array& point) {
  if (point.ndim() != 1) {
    throw py::value_error("expected an array of shape (dim,)");
  }
  return point.shape(0);
}

Index points_dim(const Array& points) {
  if (points.ndim() != 2) {
    throw py::value_error("expected an array of shape (n, dim)");
  }
  return points.shape(1);
}

template <int Dim>
Mat<Dim> to_mat(const Array& m) {
  if (m.ndim() != 2 || m.shape(0) != Dim || m.shape(1) != Dim) {
    throw py::value_error(std::format("expected an array of shape ({0}, {0})", Dim));
  }
  return Eigen::Map<const Mat<Dim>>(m.data());
}

template <int Dim>
geometry::Point<Dim> to_point(const Array& point) {
  if (point.ndim() != 1 || point.shape(0) != Dim) {
    throw py::value_error(std::format("expected an array of shape ({},)", Dim));
  }
  return Eigen::Map<const geometry::Point<Dim>>(point.data());
}

template <int Dim>
geometry::Points<Dim> to_points(const Array& points) {
  if (points.ndim() != 2 || points.shape(1) != Dim) {
    throw py::value_error(std::format("expected an array of shape (n, {})", Dim));
  }
  return Eigen::Map<const geometry::Points<Dim>>(points.data(), points.shape(0), Dim);
}

template <int Dim>
geometry::Vector<Dim> to_vector(const Array& vector) {
  if (vector.ndim() != 1 || vector.shape(0) != Dim) {
    throw py::value_error(std::format("expected an array of shape ({},)", Dim));
  }
  return Eigen::Map<const geometry::Vector<Dim>>(vector.data());
}

template <int Dim>
geometry::Vectors<Dim> to_vectors(const Array& vectors) {
  if (vectors.ndim() != 2 || vectors.shape(1) != Dim) {
    throw py::value_error(std::format("expected an array of shape (n, {})", Dim));
  }
  return Eigen::Map<const geometry::Vectors<Dim>>(vectors.data(), vectors.shape(0), Dim);
}

AnyModel make_model(const std::vector<AnyRbf>& any_rbfs, std::optional<int> poly_degree) {
  if (any_rbfs.empty()) {
    throw py::value_error("rbfs must not be empty");
  }

  return with_dim(any_rbfs.at(0).dim(), [&]<int Dim>() {
    std::vector<rbf::Rbf<Dim>> rbfs;
    rbfs.reserve(any_rbfs.size());
    for (const auto& rbf : any_rbfs) {
      rbfs.push_back(rbf.get<Dim>());
    }
    return AnyModel(Model<Dim>(std::move(rbfs), poly_degree));
  });
}

template <template <int> class R>
void define_params_init(py::class_<AnyRbfOf<R>, AnyRbf>& cls) {
  cls.def(py::init([](const std::vector<double>& params, int dim) {
            return with_dim(dim, [&]<int Dim>() { return AnyRbfOf(R<Dim>(params)); });
          }),
          "params"_a, py::kw_only(), "dim"_a);
}

template <template <int> class R>
void define_covariance_function(py::module& m, const std::string& name) {
  py::class_<AnyRbfOf<R>, AnyRbf> cls(m, name.c_str());
  cls.def(py::init([](double psill, double range, int dim) {
            return with_dim(dim, [&]<int Dim>() { return AnyRbfOf(R<Dim>(psill, range)); });
          }),
          "psill"_a, "range"_a, py::kw_only(), "dim"_a);
  define_params_init(cls);
}

template <template <int> class R>
void define_polyharmonic_rbf(py::module& m, const std::string& name) {
  py::class_<AnyRbfOf<R>, AnyRbf> cls(m, name.c_str());
  cls.def(py::init([](double scale, double c, int dim) {
            return with_dim(dim, [&]<int Dim>() { return AnyRbfOf(R<Dim>(scale, c)); });
          }),
          "scale"_a = 1.0, "c"_a = 0.0, py::kw_only(), "dim"_a);
  define_params_init(cls);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  using Mat = Mat3;
  using point_cloud::NormalEstimator;

  py::object orig_name = m.attr("__name__");
  m.attr("__name__") = "jizai";

  py::class_<NormalEstimator>(m, "NormalEstimator")
      .def(py::init([](const Array& points) {
             return std::make_unique<NormalEstimator>(to_points<3>(points));
           }),
           "points"_a)
      .def_property_readonly("normals", &NormalEstimator::normals, py::return_value_policy::copy)
      .def_property_readonly("plane_factors", &NormalEstimator::plane_factors,
                             py::return_value_policy::copy)
      .def("estimate_with_knn", py::overload_cast<Index>(&NormalEstimator::estimate_with_knn),
           "k"_a)
      .def("estimate_with_knn",
           py::overload_cast<const std::vector<Index>&>(&NormalEstimator::estimate_with_knn),
           "ks"_a)
      .def("estimate_with_radius",
           py::overload_cast<double>(&NormalEstimator::estimate_with_radius), "radius"_a)
      .def("estimate_with_radius",
           py::overload_cast<const std::vector<double>&>(&NormalEstimator::estimate_with_radius),
           "radii"_a)
      .def("filter_by_plane_factor", &NormalEstimator::filter_by_plane_factor, "threshold"_a = 1.8)
      .def(
          "orient_toward_direction",
          [](NormalEstimator& self, const Array& direction) {
            self.orient_toward_direction(to_vector<3>(direction));
          },
          "direction"_a)
      .def(
          "orient_toward_point",
          [](NormalEstimator& self, const Array& point) {
            self.orient_toward_point(to_point<3>(point));
          },
          "point"_a)
      .def("orient_closed_surface", &NormalEstimator::orient_closed_surface, "k"_a = 100);

  py::class_<point_cloud::SdfDataGenerator>(m, "SdfDataGenerator")
      .def(py::init([](const Array& points, const Array& normals, std::optional<double> offset,
                       const Array& aniso) {
             return std::make_unique<point_cloud::SdfDataGenerator>(
                 to_points<3>(points), to_vectors<3>(normals), offset, to_mat<3>(aniso));
           }),
           "points"_a, "normals"_a, "offset"_a = py::none(), "aniso"_a = Mat::Identity())
      .def_property_readonly("sdf_points", &point_cloud::SdfDataGenerator::sdf_points,
                             py::return_value_policy::copy)
      .def_property_readonly("sdf_values", &point_cloud::SdfDataGenerator::sdf_values,
                             py::return_value_policy::copy);

  py::class_<geostats::NormalScoreTransformation>(m, "NormalScoreTransformation")
      .def(py::init<int>(), "order"_a = 30)
      .def("transform", &geostats::NormalScoreTransformation::transform, "z"_a)
      .def("back_transform", &geostats::NormalScoreTransformation::back_transform, "y"_a);

  py::class_<geostats::WeightFunction>(m, "WeightFunction")
      .def(py::init<double, double, double>(), "exp_distance"_a = 0.0, "exp_model_gamma"_a = 0.0,
           "exp_num_pairs"_a = 0.0)
      .def_readonly_static("NUM_PAIRS", &geostats::WeightFunction::kNumPairs,
                           py::return_value_policy::copy)
      .def_readonly_static("NUM_PAIRS_OVER_DISTANCE_SQUARED",
                           &geostats::WeightFunction::kNumPairsOverDistanceSquared,
                           py::return_value_policy::copy)
      .def_readonly_static("NUM_PAIRS_OVER_MODEL_GAMMA_SQUARED",
                           &geostats::WeightFunction::kNumPairsOverModelGammaSquared,
                           py::return_value_policy::copy)
      .def_readonly_static("ONE", &geostats::WeightFunction::kOne, py::return_value_policy::copy)
      .def_readonly_static("ONE_OVER_DISTANCE_SQUARED",
                           &geostats::WeightFunction::kOneOverDistanceSquared,
                           py::return_value_policy::copy)
      .def_readonly_static("ONE_OVER_MODEL_GAMMA_SQUARED",
                           &geostats::WeightFunction::kOneOverModelGammaSquared,
                           py::return_value_policy::copy);

  m.attr("__version__") = xstr(JIZAI_VERSION);

  py::class_<AnyBbox>(m, "Bbox")
      .def(py::init([](int dim) {
             return with_dim(dim, []<int Dim>() { return AnyBbox(geometry::Bbox<Dim>()); });
           }),
           py::kw_only(), "dim"_a)
      .def(py::init([](const Array& min, const Array& max) {
             return with_dim(point_dim(min), [&]<int Dim>() {
               return AnyBbox(geometry::Bbox(to_point<Dim>(min), to_point<Dim>(max)));
             });
           }),
           "min"_a, "max"_a)
      .def_static(
          "from_points",
          [](const Array& points) {
            return with_dim(points_dim(points), [&]<int Dim>() {
              return AnyBbox(geometry::Bbox<Dim>::from_points(to_points<Dim>(points)));
            });
          },
          "points"_a)
      .def_property_readonly("dim", &AnyBbox::dim)
      .def_property_readonly("is_empty", VISIT(AnyBbox, x.is_empty()))
      .def_property_readonly("min", VISIT(AnyBbox, VecX(x.min().transpose())))
      .def_property_readonly("max", VISIT(AnyBbox, VecX(x.max().transpose())));

  py::class_<AnyRbf>(m, "Rbf")
      .def_property(
          "anisotropy", VISIT(AnyRbf, MatX(x.anisotropy())),
          [](AnyRbf& self, const Array& aniso) {
            self.visit([&]<int Dim>(rbf::Rbf<Dim>& x) { x.set_anisotropy(to_mat<Dim>(aniso)); });
          })
      .def_property_readonly("cpd_order", VISIT(AnyRbf, x.cpd_order()))
      .def_property_readonly("dim", &AnyRbf::dim)
      .def_property_readonly("is_covariance_function", VISIT(AnyRbf, x.is_covariance_function()))
      .def_property_readonly("num_parameters", VISIT(AnyRbf, x.num_parameters()))
      .def_property_readonly("parameter_lower_bounds", VISIT(AnyRbf, x.parameter_lower_bounds()))
      .def_property_readonly("parameter_names", VISIT(AnyRbf, x.parameter_names()))
      .def_property_readonly("parameter_upper_bounds", VISIT(AnyRbf, x.parameter_upper_bounds()))
      .def_property("parameters", VISIT(AnyRbf, x.parameters()),
                    [](AnyRbf& self, const std::vector<double>& params) {
                      self.visit([&](auto& x) { x.set_parameters(params); });
                    })
      .def_property_readonly("short_name", VISIT(AnyRbf, x.short_name()))
      .def(
          "evaluate",
          [](AnyRbf& self, const Array& diff) {
            return self.visit(
                [&]<int Dim>(rbf::Rbf<Dim>& x) { return x.evaluate(to_vector<Dim>(diff)); });
          },
          "diff"_a)
      .def(
          "evaluate_gradient",
          [](AnyRbf& self, const Array& diff) {
            return self.visit([&]<int Dim>(rbf::Rbf<Dim>& x) {
              return VecX(x.evaluate_gradient(to_vector<Dim>(diff)).transpose());
            });
          },
          "diff"_a)
      .def(
          "evaluate_hessian",
          [](AnyRbf& self, const Array& diff) {
            return self.visit([&]<int Dim>(rbf::Rbf<Dim>& x) {
              return MatX(x.evaluate_hessian(to_vector<Dim>(diff)));
            });
          },
          "diff"_a);

  // Depends on: Rbf
  define_polyharmonic_rbf<rbf::Biharmonic2D>(m, "Biharmonic2D");
  define_polyharmonic_rbf<rbf::Biharmonic3D>(m, "Biharmonic3D");
  define_covariance_function<rbf::CovCubic>(m, "CovCubic");
  define_covariance_function<rbf::CovExponential>(m, "CovExponential");
  define_covariance_function<rbf::CovGaussian>(m, "CovGaussian");
  define_covariance_function<rbf::CovGeneralizedCauchy3>(m, "CovGeneralizedCauchy3");
  define_covariance_function<rbf::CovGeneralizedCauchy5>(m, "CovGeneralizedCauchy5");
  define_covariance_function<rbf::CovGeneralizedCauchy7>(m, "CovGeneralizedCauchy7");
  define_covariance_function<rbf::CovGeneralizedCauchy9>(m, "CovGeneralizedCauchy9");
  define_covariance_function<rbf::CovSpherical>(m, "CovSpherical");
  define_covariance_function<rbf::CovSpheroidal3>(m, "CovSpheroidal3");
  define_covariance_function<rbf::CovSpheroidal5>(m, "CovSpheroidal5");
  define_covariance_function<rbf::CovSpheroidal7>(m, "CovSpheroidal7");
  define_covariance_function<rbf::CovSpheroidal9>(m, "CovSpheroidal9");
  define_polyharmonic_rbf<rbf::Triharmonic2D>(m, "Triharmonic2D");
  define_polyharmonic_rbf<rbf::Triharmonic3D>(m, "Triharmonic3D");

  // Depends on: Rbf
  py::class_<AnyModel>(m, "Model")
      .def(py::init([](const AnyRbf& rbf, std::optional<int> poly_degree) {
             return make_model({rbf}, poly_degree);
           }),
           "rbf"_a, "poly_degree"_a = py::none())
      .def(py::init(&make_model), "rbfs"_a, "poly_degree"_a = py::none())
      .def_property_readonly("cpd_order", VISIT(AnyModel, x.cpd_order()))
      .def_property_readonly("description", VISIT(AnyModel, x.description()))
      .def_property_readonly("dim", &AnyModel::dim)
      .def_property_readonly("is_covariance_model", VISIT(AnyModel, x.is_covariance_model()))
      .def_property(
          "nugget", VISIT(AnyModel, x.nugget()),
          [](AnyModel& self, double nugget) { self.visit([&](auto& x) { x.set_nugget(nugget); }); })
      .def_property_readonly("num_parameters", VISIT(AnyModel, x.num_parameters()))
      .def_property_readonly("num_rbfs", VISIT(AnyModel, x.num_rbfs()))
      .def_property_readonly("parameter_lower_bounds", VISIT(AnyModel, x.parameter_lower_bounds()))
      .def_property_readonly("parameter_names", VISIT(AnyModel, x.parameter_names()))
      .def_property_readonly("parameter_upper_bounds", VISIT(AnyModel, x.parameter_upper_bounds()))
      .def_property("parameters", VISIT(AnyModel, x.parameters()),
                    [](AnyModel& self, const std::vector<double>& params) {
                      self.visit([&](auto& x) { x.set_parameters(params); });
                    })
      .def_property_readonly("poly_basis_size", VISIT(AnyModel, x.poly_basis_size()))
      .def_property_readonly("poly_degree", VISIT(AnyModel, x.poly_degree()))
      .def_property_readonly("rbfs",
                             VISIT(AnyModel, std::vector<AnyRbf>(x.rbfs().begin(), x.rbfs().end())))
      .def_static(
          "load",
          [](const std::string& filename, int dim) {
            return with_dim(dim, [&]<int Dim>() { return AnyModel(Model<Dim>::load(filename)); });
          },
          "filename"_a, py::kw_only(), "dim"_a)
      .def(
          "save",
          [](AnyModel& self, const std::string& filename) {
            self.visit([&](auto& x) { x.save(filename); });
          },
          "filename"_a);

  // Depends on: Bbox, Model
  py::class_<AnyInterpolant>(m, "Interpolant")
      .def(py::init([](const AnyModel& model) {
             return model.visit(
                 []<int Dim>(const Model<Dim>& x) { return AnyInterpolant(Interpolant<Dim>(x)); });
           }),
           "model"_a)
      .def_property_readonly("bbox", VISIT(AnyInterpolant, AnyBbox(x.bbox())))
      .def_property_readonly("centers", VISIT(AnyInterpolant, MatX(x.centers())))
      .def_property_readonly("dim", &AnyInterpolant::dim)
      .def_property_readonly("grad_centers", VISIT(AnyInterpolant, MatX(x.grad_centers())))
      .def_property_readonly("model", VISIT(AnyInterpolant, AnyModel(x.model())))
      .def_property_readonly("weights", VISIT(AnyInterpolant, VecX(x.weights())))
      .def(
          "evaluate",
          [](AnyInterpolant& self, const Array& points, double accuracy) {
            return self.visit([&]<int Dim>(Interpolant<Dim>& x) {
              return x.evaluate(to_points<Dim>(points), accuracy);
            });
          },
          "points"_a, "accuracy"_a = kInfinity)
      .def(
          "evaluate",
          [](AnyInterpolant& self, const Array& points, const Array& grad_points, double accuracy,
             double grad_accuracy) {
            return self.visit([&]<int Dim>(Interpolant<Dim>& x) {
              return std::pair<VecX, MatX>(x.evaluate(
                  to_points<Dim>(points), to_points<Dim>(grad_points), accuracy, grad_accuracy));
            });
          },
          "points"_a, "grad_points"_a, "accuracy"_a = kInfinity, "grad_accuracy"_a = kInfinity)
      .def(
          "fit",
          [](AnyInterpolant& self, const Array& points, const VecX& values, double tolerance,
             int max_iter, double accuracy, const AnyInterpolant* initial) {
            self.visit([&]<int Dim>(Interpolant<Dim>& x) {
              x.fit(to_points<Dim>(points), values, tolerance, max_iter, accuracy,
                    get_ptr<Dim>(initial));
            });
          },
          "points"_a, "values"_a, "tolerance"_a, "max_iter"_a = 100, "accuracy"_a = kInfinity,
          "initial"_a = nullptr)
      .def(
          "fit",
          [](AnyInterpolant& self, const Array& points, const Array& grad_points,
             const VecX& values, const Array& grad_values, double tolerance, double grad_tolerance,
             int max_iter, double accuracy, double grad_accuracy, const AnyInterpolant* initial) {
            self.visit([&]<int Dim>(Interpolant<Dim>& x) {
              x.fit(to_points<Dim>(points), to_points<Dim>(grad_points), values,
                    to_vectors<Dim>(grad_values), tolerance, grad_tolerance, max_iter, accuracy,
                    grad_accuracy, get_ptr<Dim>(initial));
            });
          },
          "points"_a, "grad_points"_a, "values"_a, "grad_values"_a, "tolerance"_a,
          "grad_tolerance"_a, "max_iter"_a = 100, "accuracy"_a = kInfinity,
          "grad_accuracy"_a = kInfinity, "initial"_a = nullptr)
      .def(
          "fit_incrementally",
          [](AnyInterpolant& self, const Array& points, const VecX& values, double tolerance,
             int max_iter, double accuracy) {
            self.visit([&]<int Dim>(Interpolant<Dim>& x) {
              x.fit_incrementally(to_points<Dim>(points), values, tolerance, max_iter, accuracy);
            });
          },
          "points"_a, "values"_a, "tolerance"_a, "max_iter"_a = 100, "accuracy"_a = kInfinity)
      .def(
          "fit_incrementally",
          [](AnyInterpolant& self, const Array& points, const Array& grad_points,
             const VecX& values, const Array& grad_values, double tolerance, double grad_tolerance,
             int max_iter, double accuracy, double grad_accuracy) {
            self.visit([&]<int Dim>(Interpolant<Dim>& x) {
              x.fit_incrementally(to_points<Dim>(points), to_points<Dim>(grad_points), values,
                                  to_vectors<Dim>(grad_values), tolerance, grad_tolerance, max_iter,
                                  accuracy, grad_accuracy);
            });
          },
          "points"_a, "grad_points"_a, "values"_a, "grad_values"_a, "tolerance"_a,
          "grad_tolerance"_a, "max_iter"_a = 100, "accuracy"_a = kInfinity,
          "grad_accuracy"_a = kInfinity)
      .def(
          "fit_inequality",
          [](AnyInterpolant& self, const Array& points, const VecX& values, const VecX& values_lb,
             const VecX& values_ub, double tolerance, int max_iter, double accuracy,
             const AnyInterpolant* initial) {
            self.visit([&]<int Dim>(Interpolant<Dim>& x) {
              x.fit_inequality(to_points<Dim>(points), values, values_lb, values_ub, tolerance,
                               max_iter, accuracy, get_ptr<Dim>(initial));
            });
          },
          "points"_a, "values"_a, "values_lb"_a, "values_ub"_a, "tolerance"_a, "max_iter"_a = 100,
          "accuracy"_a = kInfinity, "initial"_a = nullptr)
      .def_static(
          "load",
          [](const std::string& filename, int dim) {
            return with_dim(
                dim, [&]<int Dim>() { return AnyInterpolant(Interpolant<Dim>::load(filename)); });
          },
          "filename"_a, py::kw_only(), "dim"_a)
      .def(
          "save",
          [](AnyInterpolant& self, const std::string& filename) {
            self.visit([&](auto& x) { x.save(filename); });
          },
          "filename"_a);

  py::class_<AnyDistanceFilter>(m, "DistanceFilter")
      .def(py::init([](const Array& points) {
             return with_dim(points_dim(points), [&]<int Dim>() {
               return std::make_unique<AnyDistanceFilter>(std::in_place_index<Dim - 1>,
                                                          to_points<Dim>(points));
             });
           }),
           "points"_a)
      .def_property_readonly("dim", &AnyDistanceFilter::dim)
      .def(
          "filtered_indices",
          [](AnyDistanceFilter& self, double distance) {
            return self.visit([&](auto& x) { return x.filtered_indices(distance); });
          },
          "distance"_a = 0.0)
      .def(
          "filtered_indices",
          [](AnyDistanceFilter& self, const std::vector<Index>& indices) {
            return self.visit([&](auto& x) { return x.filtered_indices(indices); });
          },
          "indices"_a)
      .def(
          "filtered_indices",
          [](AnyDistanceFilter& self, double distance, const std::vector<Index>& indices) {
            return self.visit([&](auto& x) { return x.filtered_indices(distance, indices); });
          },
          "distance"_a, "indices"_a);

  // Depends on: NormalScoreTransformation
  py::class_<AnyVariogram>(m, "Variogram")
      .def_property_readonly("bin_distance", VISIT(AnyVariogram, x.bin_distance()))
      .def_property_readonly("bin_gamma", VISIT(AnyVariogram, x.bin_gamma()))
      .def_property_readonly("bin_num_pairs", VISIT(AnyVariogram, x.bin_num_pairs()))
      .def_property_readonly("dim", &AnyVariogram::dim)
      .def_property_readonly("direction", VISIT(AnyVariogram, VecX(x.direction().transpose())))
      .def_property_readonly("num_bins", VISIT(AnyVariogram, x.num_bins()))
      .def_property_readonly("num_pairs", VISIT(AnyVariogram, x.num_pairs()))
      .def(
          "back_transform",
          [](AnyVariogram& self, const geostats::NormalScoreTransformation& nst) {
            self.visit([&](auto& x) { x.back_transform(nst); });
          },
          "nst"_a);

  // Depends on: NormalScoreTransformation, Variogram
  py::class_<AnyVariogramSet>(m, "VariogramSet")
      .def_property_readonly("dim", &AnyVariogramSet::dim)
      .def_property_readonly("num_pairs", VISIT(AnyVariogramSet, x.num_pairs()))
      .def_property_readonly("num_variograms", VISIT(AnyVariogramSet, x.num_variograms()))
      .def_property_readonly(
          "variograms", VISIT(AnyVariogramSet, std::vector<AnyVariogram>(x.variograms().begin(),
                                                                         x.variograms().end())))
      .def(
          "back_transform",
          [](AnyVariogramSet& self, const geostats::NormalScoreTransformation& nst) {
            self.visit([&](auto& x) { x.back_transform(nst); });
          },
          "nst"_a)
      .def_static(
          "load",
          [](const std::string& filename, int dim) {
            return with_dim(dim, [&]<int Dim>() {
              return AnyVariogramSet(geostats::VariogramSet<Dim>::load(filename));
            });
          },
          "filename"_a, py::kw_only(), "dim"_a)
      .def(
          "save",
          [](AnyVariogramSet& self, const std::string& filename) {
            self.visit([&](auto& x) { x.save(filename); });
          },
          "filename"_a);

  // Depends on: VariogramSet
  py::class_<AnyVariogramCalculator>(m, "VariogramCalculator")
      .def(py::init([](double lag_distance, Index num_lags, int dim) {
             return with_dim(dim, [&]<int Dim>() {
               return AnyVariogramCalculator(
                   geostats::VariogramCalculator<Dim>(lag_distance, num_lags));
             });
           }),
           "lag_distance"_a, "num_lags"_a, py::kw_only(), "dim"_a)
      .def_static(
          "anisotropic_directions",
          [](int dim) {
            return with_dim(dim, []<int Dim>() {
              return MatX(geostats::VariogramCalculator<Dim>::kAnisotropicDirections);
            });
          },
          py::kw_only(), "dim"_a)
      .def_static(
          "isotropic_directions",
          [](int dim) {
            return with_dim(dim, []<int Dim>() {
              return MatX(geostats::VariogramCalculator<Dim>::kIsotropicDirections);
            });
          },
          py::kw_only(), "dim"_a)
      .def_property("angle_tolerance", VISIT(AnyVariogramCalculator, x.angle_tolerance()),
                    [](AnyVariogramCalculator& self, std::optional<double> angle_tolerance) {
                      self.visit([&](auto& x) { x.set_angle_tolerance(angle_tolerance); });
                    })
      .def_property_readonly("dim", &AnyVariogramCalculator::dim)
      .def_property("directions", VISIT(AnyVariogramCalculator, MatX(x.directions())),
                    [](AnyVariogramCalculator& self, const Array& directions) {
                      self.visit([&]<int Dim>(geostats::VariogramCalculator<Dim>& x) {
                        x.set_directions(to_vectors<Dim>(directions));
                      });
                    })
      .def_property("lag_tolerance", VISIT(AnyVariogramCalculator, x.lag_tolerance()),
                    [](AnyVariogramCalculator& self, std::optional<double> lag_tolerance) {
                      self.visit([&](auto& x) { x.set_lag_tolerance(lag_tolerance); });
                    })
      .def(
          "calculate",
          [](AnyVariogramCalculator& self, const Array& points, const VecX& values) {
            return self.visit([&]<int Dim>(geostats::VariogramCalculator<Dim>& x) {
              return AnyVariogramSet(x.calculate(to_points<Dim>(points), values));
            });
          },
          "points"_a, "values"_a);

  // Depends on: Model, VariogramSet, WeightFunction
  py::class_<AnyVariogramFitting>(m, "VariogramFitting")
      .def(py::init([](const AnyVariogramSet& variog_set, const AnyModel& model,
                       const geostats::WeightFunction& weight_fn, bool fit_anisotropy) {
             return with_dim(variog_set.dim(), [&]<int Dim>() {
               return std::make_unique<AnyVariogramFitting>(std::in_place_index<Dim - 1>,
                                                            variog_set.get<Dim>(), model.get<Dim>(),
                                                            weight_fn, fit_anisotropy);
             });
           }),
           "variog_set"_a, "model"_a,
           py::arg_v("weight_fn", geostats::WeightFunction::kNumPairsOverDistanceSquared,
                     "jizai.WeightFunction.NUM_PAIRS_OVER_DISTANCE_SQUARED"),
           "fit_anisotropy"_a = true)
      .def_property_readonly("brief_report", VISIT(AnyVariogramFitting, x.brief_report()))
      .def_property_readonly("dim", &AnyVariogramFitting::dim)
      .def_property_readonly("full_report", VISIT(AnyVariogramFitting, x.full_report()))
      .def_property_readonly("final_cost", VISIT(AnyVariogramFitting, x.final_cost()))
      .def_property_readonly("model", VISIT(AnyVariogramFitting, AnyModel(x.model())));

  // Depends on: Model
  m.def(
      "cross_validate",
      [](const AnyModel& model, const Array& points, const VecX& values,
         const std::vector<Index>& set_ids, double tolerance, int max_iter, double accuracy) {
        return model.visit([&]<int Dim>(const Model<Dim>& x) {
          return geostats::cross_validate(x, to_points<Dim>(points), values, set_ids, tolerance,
                                          max_iter, accuracy);
        });
      },
      "model"_a, "points"_a, "values"_a, "set_ids"_a, "tolerance"_a, "max_iter"_a = 100,
      "accuracy"_a = kInfinity);

  m.def(
      "detrend",
      [](const Array& points, const VecX& values, int degree) {
        return with_dim(points_dim(points), [&]<int Dim>() {
          return geostats::detrend(to_points<Dim>(points), values, degree);
        });
      },
      "points"_a, "values"_a, "degree"_a);

  py::class_<isosurface::FieldFunction>(m, "_FieldFunction");

  // Depends on: Interpolant, _FieldFunction
  py::class_<isosurface::RbfFieldFunction, isosurface::FieldFunction>(m, "RbfFieldFunction")
      .def(py::init([](AnyInterpolant& interpolant, double accuracy, double grad_accuracy) {
             return std::make_unique<isosurface::RbfFieldFunction>(interpolant.get<3>(), accuracy,
                                                                   grad_accuracy);
           }),
           "interpolant"_a, "accuracy"_a = kInfinity, "grad_accuracy"_a = kInfinity,
           py::keep_alive<1, 2>());

  // Depends on: Interpolant, _FieldFunction
  py::class_<isosurface::RbfFieldFunction25D, isosurface::FieldFunction>(m, "RbfFieldFunction25D")
      .def(py::init([](AnyInterpolant& interpolant, double accuracy, double grad_accuracy) {
             return std::make_unique<isosurface::RbfFieldFunction25D>(interpolant.get<2>(),
                                                                      accuracy, grad_accuracy);
           }),
           "interpolant"_a, "accuracy"_a = kInfinity, "grad_accuracy"_a = kInfinity,
           py::keep_alive<1, 2>());

  py::class_<isosurface::Mesh>(m, "Mesh")
      .def("export_obj", &isosurface::Mesh::export_obj, "filename"_a)
      .def_property_readonly("faces", &isosurface::Mesh::faces, py::return_value_policy::copy)
      .def_property_readonly("is_empty", &isosurface::Mesh::is_empty)
      .def_property_readonly("is_entire", &isosurface::Mesh::is_entire)
      .def_property_readonly("vertices", &isosurface::Mesh::vertices,
                             py::return_value_policy::copy);

  // Depends on: Bbox, Mesh, _FieldFunction
  py::class_<isosurface::Isosurface>(m, "Isosurface")
      .def(py::init([](const AnyBbox& bbox, double resolution, const Array& aniso) {
             return std::make_unique<isosurface::Isosurface>(bbox.get<3>(), resolution,
                                                             to_mat<3>(aniso));
           }),
           "bbox"_a, "resolution"_a, "aniso"_a = Mat::Identity())
      .def("generate", &isosurface::Isosurface::generate, "field_fn"_a, "isovalue"_a = 0.0,
           "refine"_a = true)
      .def(
          "generate_from_seed_points",
          [](isosurface::Isosurface& self, const Array& seed_points,
             isosurface::FieldFunction& field_fn, double isovalue, bool refine) {
            return self.generate_from_seed_points(to_points<3>(seed_points), field_fn, isovalue,
                                                  refine);
          },
          "seed_points"_a, "field_fn"_a, "isovalue"_a = 0.0, "refine"_a = true)
      .def(
          "set_snap_points",
          [](isosurface::Isosurface& self, const Array& points, const VecX& relative_tolerances) {
            self.set_snap_points(to_points<3>(points), relative_tolerances);
          },
          "points"_a, "relative_tolerances"_a = VecX());

  m.attr("__name__") = orig_name;
}
