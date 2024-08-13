#pragma once

#include <Eigen/Core>
#include <boost/container_hash/hash.hpp>
#include <format>
#include <limits>
#include <memory>
#include <polatory/common/io.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/interpolation/rbf_incremental_fitter.hpp>
#include <polatory/interpolation/rbf_inequality_fitter.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace polatory {

template <int Dim>
class Interpolant {
  static constexpr int kDim = Dim;
  static constexpr double kInfinity = std::numeric_limits<double>::infinity();
  using Bbox = geometry::Bbox<kDim>;
  using Evaluator = interpolation::Evaluator<kDim>;
  using Fitter = interpolation::Fitter<kDim>;
  using IncrementalFitter = interpolation::IncrementalFitter<kDim>;
  using InequalityFitter = interpolation::InequalityFitter<kDim>;
  using Model = Model<kDim>;
  using Point = geometry::Point<kDim>;
  using Points = geometry::Points<kDim>;

 public:
  explicit Interpolant(const Model& model) : model_(model) {}

  const Bbox& bbox() const {
    throw_if_not_fitted();

    return bbox_;
  }

  const Points& centers() const {
    throw_if_not_fitted();

    return centers_;
  }

  VecX evaluate(const Points& points, double accuracy = kInfinity) {
    return evaluate(points, Points(0, kDim), accuracy, kInfinity);
  }

  VecX evaluate(const Points& points, const Points& grad_points, double accuracy = kInfinity,
                double grad_accuracy = kInfinity) {
    throw_if_not_fitted();

    check_accuracy(accuracy, grad_accuracy);

    set_evaluation_bbox_impl(Bbox::from_points(points).convex_hull(Bbox::from_points(grad_points)),
                             accuracy, grad_accuracy);
    return evaluate_impl(points, grad_points);
  }

  VecX evaluate_impl(const Points& points) const { return evaluate_impl(points, Points(0, kDim)); }

  VecX evaluate_impl(const Points& points, const Points& grad_points) const {
    throw_if_not_fitted();

    return evaluator_->evaluate(points, grad_points);
  }

  void fit(const Points& points, const VecX& values, double tolerance, int max_iter = 100,
           double accuracy = kInfinity, const Interpolant* initial = nullptr) {
    fit(points, Points(0, kDim), values, tolerance, kInfinity, max_iter, accuracy, kInfinity,
        initial);
  }

  void fit(const Points& points, const Points& grad_points, const VecX& values, double tolerance,
           double grad_tolerance, int max_iter = 100, double accuracy = kInfinity,
           double grad_accuracy = kInfinity, const Interpolant* initial = nullptr) {
    check_num_points(points, grad_points);

    auto n_rhs = points.rows() + kDim * grad_points.rows();
    if (values.rows() != n_rhs) {
      throw std::invalid_argument(std::format("values.rows() must be equal to {}", n_rhs));
    }

    check_tolerance(tolerance, grad_tolerance);
    check_max_iter(max_iter);
    check_accuracy(accuracy, grad_accuracy);

    VecX initial_weights;
    if (initial != nullptr) {
      initial_weights = build_initial_weights(points, grad_points, *initial);
    }

    // Clear after the initial weights are built as `initial` can be `this`.
    clear();

    Fitter fitter(model_, points, grad_points);
    weights_ = fitter.fit(values, tolerance, grad_tolerance, max_iter, accuracy, grad_accuracy,
                          initial != nullptr ? &initial_weights : nullptr);

    fitted_ = true;
    centers_ = points;
    grad_centers_ = grad_points;
    bbox_ = Bbox::from_points(centers_).convex_hull(Bbox::from_points(grad_centers_));
  }

  void fit_incrementally(const Points& points, const VecX& values, double tolerance,
                         int max_iter = 100, double accuracy = kInfinity) {
    fit_incrementally(points, Points(0, kDim), values, tolerance, kInfinity, max_iter, accuracy,
                      kInfinity);
  }

  void fit_incrementally(const Points& points, const Points& grad_points, const VecX& values,
                         double tolerance, double grad_tolerance, int max_iter = 100,
                         double accuracy = kInfinity, double grad_accuracy = kInfinity) {
    check_num_points(points, grad_points);

    if (values.rows() != points.rows() + kDim * grad_points.rows()) {
      throw std::invalid_argument(std::format("values.rows() must be equal to {}",
                                              points.rows() + kDim * grad_points.rows()));
    }

    check_tolerance(tolerance, grad_tolerance);
    check_max_iter(max_iter);
    check_accuracy(accuracy, grad_accuracy);

    clear();

    IncrementalFitter fitter(model_, points, grad_points);
    std::vector<Index> center_indices;
    std::vector<Index> grad_center_indices;
    std::tie(center_indices, grad_center_indices, weights_) =
        fitter.fit(values, tolerance, grad_tolerance, max_iter, accuracy, grad_accuracy);

    fitted_ = true;
    centers_ = points(center_indices, Eigen::all);
    grad_centers_ = grad_points(grad_center_indices, Eigen::all);
    bbox_ = Bbox::from_points(centers_).convex_hull(Bbox::from_points(grad_centers_));
  }

  void fit_inequality(const Points& points, const VecX& values, const VecX& values_lb,
                      const VecX& values_ub, double tolerance, int max_iter = 100,
                      double accuracy = kInfinity, const Interpolant* initial = nullptr) {
    if (model_.nugget() != 0.0) {
      throw std::runtime_error("Non-zero nugget is not supported");
    }

    check_num_points(points, Points(0, kDim));

    if (values.rows() != points.rows()) {
      throw std::invalid_argument("values.rows() must be equal to points.rows()");
    }

    if (values_lb.rows() != points.rows()) {
      throw std::invalid_argument("values_lb.rows() must be equal to points.rows()");
    }

    if (values_ub.rows() != points.rows()) {
      throw std::invalid_argument("values_ub.rows() must be equal to points.rows()");
    }

    check_tolerance(tolerance, kInfinity);
    check_max_iter(max_iter);
    check_accuracy(accuracy, kInfinity);

    VecX initial_weights;
    if (initial != nullptr) {
      initial_weights = build_initial_weights(points, Points(0, kDim), *initial);
    }

    // Clear after the initial weights are built as `initial` can be `this`.
    clear();

    InequalityFitter fitter(model_, points);
    std::vector<Index> center_indices;
    std::tie(center_indices, weights_) =
        fitter.fit(values, values_lb, values_ub, tolerance, max_iter, accuracy,
                   initial != nullptr ? &initial_weights : nullptr);

    fitted_ = true;
    centers_ = points(center_indices, Eigen::all);
    bbox_ = Bbox::from_points(centers_);
  }

  const Points& grad_centers() const {
    throw_if_not_fitted();

    return grad_centers_;
  }

  const Model& model() const { return model_; }

  void set_evaluation_bbox_impl(const Bbox& bbox, double accuracy = kInfinity,
                                double grad_accuracy = kInfinity) {
    throw_if_not_fitted();

    auto union_bbox = bbox.convex_hull(bbox_);

    evaluator_ = std::make_unique<Evaluator>(model_, centers_, grad_centers_, union_bbox, accuracy,
                                             grad_accuracy);
    evaluator_->set_weights(weights_);
  }

  const VecX& weights() const {
    throw_if_not_fitted();

    return weights_;
  }

  POLATORY_IMPLEMENT_LOAD_SAVE(Interpolant);

 private:
  POLATORY_FRIEND_READ_WRITE;

  struct PointHash {
    std::size_t operator()(const Point& p) const noexcept {
      return boost::hash_range(p.data(), p.data() + p.size());
    }
  };

  // For deserialization.
  Interpolant() = default;

  VecX build_initial_weights(const Points& points, const Points& grad_points,
                             const Interpolant& initial) const {
    auto l = model().poly_basis_size();
    auto mu = points.rows();
    auto sigma = grad_points.rows();
    VecX weights = VecX::Zero(mu + kDim * sigma + l);

    if (model() != initial.model()) {
      std::cerr << "warning: ignoring the initial interpolant because the model is different"
                << std::endl;
      return weights;
    }

    std::unordered_map<Point, Index, PointHash> ini_points;
    std::unordered_map<Point, Index, PointHash> ini_grad_points;

    auto ini_mu = initial.centers_.rows();
    auto ini_sigma = initial.grad_centers_.rows();
    for (Index i = 0; i < ini_mu; ++i) {
      ini_points.emplace(initial.centers_.row(i), i);
    }
    for (Index i = 0; i < ini_sigma; ++i) {
      ini_grad_points.emplace(initial.grad_centers_.row(i), i);
    }

    for (Index i = 0; i < mu; ++i) {
      auto it = ini_points.find(points.row(i));
      if (it != ini_points.end()) {
        weights(i) = initial.weights()(it->second);
      }
    }
    for (Index i = 0; i < sigma; ++i) {
      auto it = ini_grad_points.find(grad_points.row(i));
      if (it != ini_grad_points.end()) {
        weights.segment<kDim>(mu + kDim * i) =
            initial.weights().template segment<kDim>(ini_mu + kDim * it->second);
      }
    }
    weights.tail(l) = initial.weights().tail(l);

    return weights;
  }

  void check_accuracy(double accuracy, double grad_accuracy) const {
    if (!(accuracy > 0.0)) {
      throw std::invalid_argument("accuracy must be positive");
    }

    if (!(grad_accuracy > 0.0)) {
      throw std::invalid_argument("grad_accuracy must be positive");
    }
  }

  void check_max_iter(int max_iter) const {
    if (max_iter < 0) {
      throw std::invalid_argument("max_iter must be nonnegative");
    }
  }

  void check_num_points(const Points& points, const Points& grad_points) const {
    auto l = model_.poly_basis_size();
    auto mu = points.rows();
    auto sigma = grad_points.rows();

    if (model_.poly_degree() == 1 && mu == 1 && sigma >= 1) {
      // The special case.
      return;
    }

    if (mu < l) {
      throw std::invalid_argument(
          std::format("points.rows() must be greater than or equal to {}", l));
    }
  }

  void check_tolerance(double tolerance, double grad_tolerance) const {
    if (!(tolerance > 0.0)) {
      throw std::invalid_argument("tolerance must be positive");
    }

    if (!(grad_tolerance > 0.0)) {
      throw std::invalid_argument("grad_tolerance must be positive");
    }
  }

  void clear() {
    fitted_ = false;
    centers_ = Points();
    grad_centers_ = Points();
    bbox_ = Bbox();
    weights_ = VecX();
  }

  void throw_if_not_fitted() const {
    if (!fitted_) {
      throw std::runtime_error("interpolant has not been fitted");
    }
  }

  Model model_;
  bool fitted_{};
  Points centers_;
  Points grad_centers_;
  Bbox bbox_;
  VecX weights_;

  std::unique_ptr<Evaluator> evaluator_;
};

}  // namespace polatory

namespace polatory::common {

template <int Dim>
struct Read<Interpolant<Dim>> {
  void operator()(std::istream& is, Interpolant<Dim>& t) {
    read(is, t.model_);
    read(is, t.fitted_);
    read(is, t.centers_);
    read(is, t.grad_centers_);
    read(is, t.bbox_);
    read(is, t.weights_);
  }
};

template <int Dim>
struct Write<Interpolant<Dim>> {
  void operator()(std::ostream& os, const Interpolant<Dim>& t) {
    write(os, t.model_);
    write(os, t.fitted_);
    write(os, t.centers_);
    write(os, t.grad_centers_);
    write(os, t.bbox_);
    write(os, t.weights_);
  }
};

}  // namespace polatory::common
