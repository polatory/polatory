#pragma once

namespace polatory {
namespace common {

template <class Forwarditerator>
class iterator_range {
public:
  iterator_range(Forwarditerator begin, Forwarditerator end)
    : begin_(begin)
    , end_(end) {
  }

  auto begin() const {
    return begin_;
  }

  auto end() const {
    return end_;
  }

private:
  Forwarditerator begin_;
  Forwarditerator end_;
};

template <class Forwarditerator>
iterator_range<Forwarditerator> make_range(Forwarditerator begin, Forwarditerator end) {
  return iterator_range<Forwarditerator>(begin, end);
}

}  // namespace common
}  // namespace polatory
