#pragma once

#include <numeric>
#include <polatory/common/io.hpp>
#include <polatory/kriging/normal_score_transformation.hpp>
#include <polatory/kriging/variogram.hpp>
#include <polatory/types.hpp>
#include <utility>
#include <vector>

namespace polatory::kriging {

template <int Dim>
class variogram_set {
  using Variogram = variogram<Dim>;

 public:
  explicit variogram_set(std::vector<Variogram>&& variograms)
      : variograms_{std::move(variograms)} {}

  ~variogram_set() = default;

  variogram_set(const variogram_set& variogram_set) = default;
  variogram_set(variogram_set&& variogram_set) = default;
  variogram_set& operator=(const variogram_set&) = default;
  variogram_set& operator=(variogram_set&&) = default;

  void back_transform(const normal_score_transformation& t) {
    for (auto& v : variograms_) {
      v.back_transform(t);
    }
  }

  index_t num_pairs() const {
    return std::reduce(variograms_.begin(), variograms_.end(), index_t{0},
                       [](auto acc, const auto& v) { return acc + v.num_pairs(); });
  }

  index_t num_variograms() const { return static_cast<index_t>(variograms_.size()); }

  const std::vector<Variogram>& variograms() const { return variograms_; }

  POLATORY_IMPLEMENT_LOAD_SAVE(variogram_set);

 private:
  POLATORY_FRIEND_READ_WRITE(variogram_set);

  // For deserialization.
  variogram_set() = default;

  std::vector<Variogram> variograms_;
};

}  // namespace polatory::kriging

namespace polatory::common {

template <int Dim>
struct Read<kriging::variogram_set<Dim>> {
  void operator()(std::istream& is, kriging::variogram_set<Dim>& t) const {
    read(is, t.variograms_);
  }
};

template <int Dim>
struct Write<kriging::variogram_set<Dim>> {
  void operator()(std::ostream& os, const kriging::variogram_set<Dim>& t) const {
    write(os, t.variograms_);
  }
};

}  // namespace polatory::common
