#pragma once

#include <jizai/common/io.hpp>
#include <jizai/geostats/normal_score_transformation.hpp>
#include <jizai/geostats/variogram.hpp>
#include <jizai/types.hpp>
#include <numeric>
#include <utility>
#include <vector>

namespace jizai::geostats {

template <int Dim>
class VariogramSet {
  using Variogram = Variogram<Dim>;

 public:
  explicit VariogramSet(std::vector<Variogram> variograms) : variograms_{std::move(variograms)} {}

  ~VariogramSet() = default;

  VariogramSet(const VariogramSet& variogram_set) = default;
  VariogramSet(VariogramSet&& variogram_set) = default;
  VariogramSet& operator=(const VariogramSet&) = default;
  VariogramSet& operator=(VariogramSet&&) = default;

  void back_transform(const NormalScoreTransformation& nst) {
    for (auto& v : variograms_) {
      v.back_transform(nst);
    }
  }

  Index num_pairs() const {
    return std::reduce(variograms_.begin(), variograms_.end(), Index{0},
                       [](auto acc, const auto& v) { return acc + v.num_pairs(); });
  }

  Index num_variograms() const { return static_cast<Index>(variograms_.size()); }

  const std::vector<Variogram>& variograms() const { return variograms_; }

  JIZAI_IMPLEMENT_LOAD_SAVE(VariogramSet);

 private:
  JIZAI_FRIEND_READ_WRITE;

  // For deserialization.
  VariogramSet() = default;

  std::vector<Variogram> variograms_;
};

}  // namespace jizai::geostats

namespace jizai::common {

template <int Dim>
struct Read<geostats::VariogramSet<Dim>> {
  void operator()(std::istream& is, geostats::VariogramSet<Dim>& t) const {
    read(is, t.variograms_);
  }
};

template <int Dim>
struct Write<geostats::VariogramSet<Dim>> {
  void operator()(std::ostream& os, const geostats::VariogramSet<Dim>& t) const {
    write(os, t.variograms_);
  }
};

}  // namespace jizai::common
