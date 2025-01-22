#pragma once

#include <memory>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
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

  using Particle = GenericParticle<0>;

  using Container = scalfmm::container::particle_container<Particle>;

  using Position = typename Particle::position_type;
  using Box = scalfmm::component::box<Position>;

 public:
  Resource(const Rbf& rbf, const Bbox& bbox) : a_{rbf.anisotropy()}, box_{make_box(bbox)} {}

  template <class Container>
  Container get_particles(int level) const {
    if (sorted_level_ < level) {
      scalfmm::utils::sort_container(box_, level, particles_);
      sorted_level_ = level;
    }

    Container result;
    result.resize(particles_.size());

    auto p_it = particles_.cbegin();
    auto p_end = particles_.cend();
    auto q_it = result.begin();
    for (; p_it != p_end; ++p_it, ++q_it) {
      q_it->position() = p_it.position();
      q_it->inputs() = p_it.inputs();
      q_it->variables() = p_it.variables();
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
    Index n = particles_.size();

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
  Index mu_{};
  mutable Container particles_;
  mutable int sorted_level_{};
};

}  // namespace polatory::fmm
