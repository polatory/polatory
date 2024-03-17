#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#undef _GNU_SOURCE
#include <polatory/polatory.hpp>
#include <vector>

#define xstr(s) str(s)
#define str(s) #s

using namespace polatory;
namespace py = pybind11;
using namespace py::literals;

template <int Dim>
void define_module(py::module& m) {
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

  py::class_<rbf::biharmonic2d<Dim>, rbf::rbf_proxy<Dim>>(m, "Biharmonic2D")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::biharmonic3d<Dim>, rbf::rbf_proxy<Dim>>(m, "Biharmonic3D")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_cauchy3<Dim>, rbf::rbf_proxy<Dim>>(m, "CovCauchy3")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_cauchy5<Dim>, rbf::rbf_proxy<Dim>>(m, "CovCauchy5")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_cauchy7<Dim>, rbf::rbf_proxy<Dim>>(m, "CovCauchy7")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_cauchy9<Dim>, rbf::rbf_proxy<Dim>>(m, "CovCauchy9")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_cubic<Dim>, rbf::rbf_proxy<Dim>>(m, "CovCubic")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_exponential<Dim>, rbf::rbf_proxy<Dim>>(m, "CovExponential")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_gaussian<Dim>, rbf::rbf_proxy<Dim>>(m, "CovGaussian")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_spherical<Dim>, rbf::rbf_proxy<Dim>>(m, "CovSpherical")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_spheroidal3<Dim>, rbf::rbf_proxy<Dim>>(m, "CovSpheroidal3")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_spheroidal5<Dim>, rbf::rbf_proxy<Dim>>(m, "CovSpheroidal5")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_spheroidal7<Dim>, rbf::rbf_proxy<Dim>>(m, "CovSpheroidal7")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::cov_spheroidal9<Dim>, rbf::rbf_proxy<Dim>>(m, "CovSpheroidal9")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::inverse_multiquadric1<Dim>, rbf::rbf_proxy<Dim>>(m, "InverseMultiquadric1")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::multiquadric1<Dim>, rbf::rbf_proxy<Dim>>(m, "Multiquadric1")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::multiquadric3<Dim>, rbf::rbf_proxy<Dim>>(m, "Multiquadric3")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::triharmonic2d<Dim>, rbf::rbf_proxy<Dim>>(m, "Triharmonic2D")
      .def(py::init<const std::vector<double>&>(), "params"_a);
  py::class_<rbf::triharmonic3d<Dim>, rbf::rbf_proxy<Dim>>(m, "Triharmonic3D")
      .def(py::init<const std::vector<double>&>(), "params"_a);

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

  m.attr("__version__") = xstr(POLATORY_VERSION);
}
