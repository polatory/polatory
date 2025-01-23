#pragma once

#include <memory>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/rbf/rbf.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/container/particle_container.hpp>
#include <scalfmm/tree/box.hpp>
#include <scalfmm/utils/sort.hpp>
#include <tuple>

namespace polatory::fmm {

template <int Dim, int km>
class Resource {
  static constexpr int kDim = Dim;
  using Bbox = geometry::Bbox<kDim>;
  using Mat = Mat<Dim>;
  using Points = geometry::Points<kDim>;
  using Rbf = rbf::Rbf<kDim>;

  template <int kn>
  using GenericParticle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,
      /* outputs */ double, kn,
      /* variables */ Index>;

  using Particle = GenericParticle<1>;

  using Container = scalfmm::container::particle_container<Particle>;

  using Position = typename Particle::position_type;
  using Box = scalfmm::component::box<Position>;

 public:
  Resource(const Rbf& rbf, const Bbox& bbox) : a_{rbf.anisotropy()}, box_{make_box(bbox)} {}

  std::unique_ptr<point_cloud::KdTree<kDim>> get_kdtree() const {
    auto n = size();
    Points apoints(n, kDim);

    for (Index k = 0; k < n; k++) {
      const auto p = particles_.at(k);
      auto orig_idx = std::get<0>(p.variables());
      for (auto i = 0; i < kDim; i++) {
        apoints(orig_idx, i) = p.position(i);
      }
    }

    return std::make_unique<point_cloud::KdTree<kDim>>(apoints);
  }

  template <class Container, bool Source>
  Container get_particles(int level) const {
    static_assert(!Source || Container::particle_type::inputs_size == km);

    if (sorted_level_ < level) {
      scalfmm::utils::sort_container(box_, level, particles_);
      sorted_level_ = level;
    }

    auto n = size();
    Container result;
    result.resize(n);

    for (Index idx = 0; idx < n; idx++) {
      const auto p = particles_.at(idx);
      auto q = result.at(idx);
      q.position() = p.position();
      if constexpr (Source) {
        for (auto i = 0; i < km; i++) {
          q.inputs(i) = p.inputs(i);
        }
      }
      q.variables() = p.variables();
    }

    return result;
  }

  void set_points(const Points& points) {
    auto n = points.rows();
    particles_.resize(n);

    for (Index idx = 0; idx < n; idx++) {
      auto p = particles_.at(idx);
      auto ap = geometry::transform_point<kDim>(a_, points.row(idx));
      for (auto i = 0; i < kDim; i++) {
        p.position(i) = ap(i);
      }
      p.variables(idx);
    }
  }

  void set_weights(const Eigen::Ref<const VecX>& weights) {
    auto n = size();

    for (Index idx = 0; idx < n; idx++) {
      auto p = particles_.at(idx);
      auto orig_idx = std::get<0>(p.variables());
      for (auto i = 0; i < km; i++) {
        p.inputs(i) = weights(km * orig_idx + i);
      }
    }
  }

  Index size() const { return particles_.size(); }

 private:
  // IMPORTANT: Keep in sync with the function in src/fmm/utility.hpp.
  Box make_box(const Bbox& bbox) {
    auto a_bbox = bbox.transform(a_);

    auto width = 1.01 * a_bbox.width().maxCoeff();
    if (width == 0.0) {
      width = 1.0;
    }

    typename Box::position_type center;
    for (auto i = 0; i < kDim; ++i) {
      center.at(i) = a_bbox.center()(i);
    }

    return {width, center};
  }

  Mat a_;
  Box box_;
  mutable Container particles_;
  mutable int sorted_level_{};
};

}  // namespace polatory::fmm
