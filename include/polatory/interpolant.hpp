#pragma once

#include <Eigen/Core>
#include <boost/container_hash/hash.hpp>
#include <format>
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
class interpolant {
  static constexpr int kDim = Dim;
  using Bbox = geometry::bboxNd<kDim>;
  using Evaluator = interpolation::rbf_evaluator<kDim>;
  using Fitter = interpolation::rbf_fitter<kDim>;
  using IncrementalFitter = interpolation::rbf_incremental_fitter<kDim>;
  using InequalityFitter = interpolation::rbf_inequality_fitter<kDim>;
  using Model = model<kDim>;
  using Point = geometry::pointNd<kDim>;
  using Points = geometry::pointsNd<kDim>;

 public:
  explicit interpolant(const Model& model) : model_(model) {}

  const Bbox& bbox() const {
    throw_if_not_fitted();

    return bbox_;
  }

  const Points& centers() const {
    throw_if_not_fitted();

    return centers_;
  }

  vectord evaluate(const Points& points) { return evaluate(points, Points(0, kDim)); }

  vectord evaluate(const Points& points, const Points& grad_points) {
    throw_if_not_fitted();

    set_evaluation_bbox_impl(Bbox::from_points(points).convex_hull(Bbox::from_points(grad_points)));
    return evaluate_impl(points, grad_points);
  }

  vectord evaluate_impl(const Points& points) const {
    return evaluate_impl(points, Points(0, kDim));
  }

  vectord evaluate_impl(const Points& points, const Points& grad_points) const {
    throw_if_not_fitted();

    return evaluator_->evaluate(points, grad_points);
  }

  void fit(const Points& points, const vectord& values, double absolute_tolerance,
           int max_iter = 100, const interpolant* initial = nullptr) {
    fit(points, Points(0, kDim), values, absolute_tolerance, absolute_tolerance, max_iter, initial);
  }

  void fit(const Points& points, const Points& grad_points, const vectord& values,
           double absolute_tolerance, double grad_absolute_tolerance, int max_iter = 100,
           const interpolant* initial = nullptr) {
    check_num_points(points, grad_points);

    auto n_rhs = points.rows() + kDim * grad_points.rows();
    if (values.rows() != n_rhs) {
      throw std::invalid_argument(std::format("values.rows() must be equal to {}", n_rhs));
    }

    if (!(absolute_tolerance > 0.0)) {
      throw std::invalid_argument("absolute_tolerance must be positive");
    }

    if (!(grad_absolute_tolerance > 0.0)) {
      throw std::invalid_argument("grad_absolute_tolerance must be positive");
    }

    vectord initial_weights;
    if (initial != nullptr) {
      initial_weights = build_initial_weights(points, grad_points, *initial);
    }

    // Clear after the initial weights are built as `initial` can be `this`.
    clear();

    Fitter fitter(model_, points, grad_points);
    weights_ = fitter.fit(values, absolute_tolerance, grad_absolute_tolerance, max_iter,
                          initial != nullptr ? &initial_weights : nullptr);

    fitted_ = true;
    centers_ = points;
    grad_centers_ = grad_points;
    bbox_ = Bbox::from_points(centers_).convex_hull(Bbox::from_points(grad_centers_));
  }

  void fit_incrementally(const Points& points, const vectord& values, double absolute_tolerance,
                         int max_iter = 100) {
    fit_incrementally(points, Points(0, kDim), values, absolute_tolerance, absolute_tolerance,
                      max_iter);
  }

  void fit_incrementally(const Points& points, const Points& grad_points, const vectord& values,
                         double absolute_tolerance, double grad_absolute_tolerance,
                         int max_iter = 100) {
    check_num_points(points, grad_points);

    if (values.rows() != points.rows() + kDim * grad_points.rows()) {
      throw std::invalid_argument(std::format("values.rows() must be equal to {}",
                                              points.rows() + kDim * grad_points.rows()));
    }

    if (!(absolute_tolerance > 0.0)) {
      throw std::invalid_argument("absolute_tolerance must be positive");
    }

    if (!(grad_absolute_tolerance > 0.0)) {
      throw std::invalid_argument("grad_absolute_tolerance must be positive");
    }

    clear();

    IncrementalFitter fitter(model_, points, grad_points);
    std::vector<index_t> center_indices;
    std::vector<index_t> grad_center_indices;
    std::tie(center_indices, grad_center_indices, weights_) =
        fitter.fit(values, absolute_tolerance, grad_absolute_tolerance, max_iter);

    fitted_ = true;
    centers_ = points(center_indices, Eigen::all);
    grad_centers_ = grad_points(grad_center_indices, Eigen::all);
    bbox_ = Bbox::from_points(centers_).convex_hull(Bbox::from_points(grad_centers_));
  }

  void fit_inequality(const Points& points, const vectord& values, const vectord& values_lb,
                      const vectord& values_ub, double absolute_tolerance, int max_iter = 100) {
    if (model_.nugget() != 0.0) {
      throw std::runtime_error("Non-zero nugget is not supported");
    }

    auto min_n_points = model_.poly_basis_size();
    if (points.rows() < min_n_points) {
      throw std::invalid_argument(
          std::format("points.rows() must be greater than or equal to {}", min_n_points));
    }

    if (values.rows() != points.rows()) {
      throw std::invalid_argument("values.rows() must be equal to points.rows()");
    }

    if (values_lb.rows() != points.rows()) {
      throw std::invalid_argument("values_lb.rows() must be equal to points.rows()");
    }

    if (values_ub.rows() != points.rows()) {
      throw std::invalid_argument("values_ub.rows() must be equal to points.rows()");
    }

    if (!(absolute_tolerance > 0.0)) {
      throw std::invalid_argument("absolute_tolerance must be positive");
    }

    clear();

    InequalityFitter fitter(model_, points);
    std::vector<index_t> center_indices;
    std::tie(center_indices, weights_) =
        fitter.fit(values, values_lb, values_ub, absolute_tolerance, max_iter);

    fitted_ = true;
    centers_ = points(center_indices, Eigen::all);
    bbox_ = Bbox::from_points(centers_);
  }

  const Points& grad_centers() const {
    throw_if_not_fitted();

    return grad_centers_;
  }

  const Model& model() const { return model_; }

  void set_evaluation_bbox_impl(const Bbox& bbox) {
    throw_if_not_fitted();

    auto union_bbox = bbox.convex_hull(bbox_);

    evaluator_ = std::make_unique<Evaluator>(model_, centers_, grad_centers_, union_bbox);
    evaluator_->set_weights(weights_);
  }

  const vectord& weights() const {
    throw_if_not_fitted();

    return weights_;
  }

  POLATORY_IMPLEMENT_LOAD_SAVE(interpolant);

 private:
  POLATORY_FRIEND_READ_WRITE(model);

  struct point_hash {
    std::size_t operator()(const Point& p) const noexcept {
      return boost::hash_range(p.data(), p.data() + p.size());
    }
  };

  // For deserialization.
  interpolant() = default;

  vectord build_initial_weights(const Points& points, const Points& grad_points,
                                const interpolant& initial) const {
    auto l = model().poly_basis_size();
    auto mu = points.rows();
    auto sigma = grad_points.rows();
    vectord weights = vectord::Zero(mu + kDim * sigma + l);

    if (model() != initial.model()) {
      std::cerr << "warning: ignoring the initial interpolant because the model is different"
                << std::endl;
      return weights;
    }

    std::unordered_map<Point, index_t, point_hash> ini_points;
    std::unordered_map<Point, index_t, point_hash> ini_grad_points;

    auto ini_mu = initial.centers_.rows();
    auto ini_sigma = initial.grad_centers_.rows();
    for (index_t i = 0; i < ini_mu; ++i) {
      ini_points.emplace(initial.centers_.row(i), i);
    }
    for (index_t i = 0; i < ini_sigma; ++i) {
      ini_grad_points.emplace(initial.grad_centers_.row(i), i);
    }

    for (index_t i = 0; i < mu; ++i) {
      auto it = ini_points.find(points.row(i));
      if (it != ini_points.end()) {
        weights(i) = initial.weights()(it->second);
      }
    }
    for (index_t i = 0; i < sigma; ++i) {
      auto it = ini_grad_points.find(grad_points.row(i));
      if (it != ini_grad_points.end()) {
        weights.segment<kDim>(mu + kDim * i) =
            initial.weights().template segment<kDim>(ini_mu + kDim * it->second);
      }
    }
    weights.tail(l) = initial.weights().tail(l);

    return weights;
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

  void clear() {
    fitted_ = false;
    centers_ = Points();
    grad_centers_ = Points();
    bbox_ = Bbox();
    weights_ = vectord();
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
  vectord weights_;

  std::unique_ptr<Evaluator> evaluator_;
};

}  // namespace polatory

namespace polatory::common {

template <int Dim>
struct Read<interpolant<Dim>> {
  void operator()(std::istream& is, interpolant<Dim>& t) {
    read(is, t.model_);
    read(is, t.fitted_);
    read(is, t.centers_);
    read(is, t.grad_centers_);
    read(is, t.bbox_);
    read(is, t.weights_);
  }
};

template <int Dim>
struct Write<interpolant<Dim>> {
  void operator()(std::ostream& os, const interpolant<Dim>& t) {
    write(os, t.model_);
    write(os, t.fitted_);
    write(os, t.centers_);
    write(os, t.grad_centers_);
    write(os, t.bbox_);
    write(os, t.weights_);
  }
};

}  // namespace polatory::common
