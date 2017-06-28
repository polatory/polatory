// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

namespace polatory {
namespace common {

template<typename T>
class vector_range_view {
   const std::vector<T>& v;
   const size_t begin_idx;
   const size_t end_idx;

public:
   typedef typename std::vector<T>::const_iterator iterator;

   vector_range_view(const std::vector<T>& vector, size_t begin_index, size_t end_index)
      : v(vector)
      , begin_idx(begin_index)
      , end_idx(end_index)
   {
   }

   iterator begin() const
   {
      return v.begin() + begin_idx;
   }

   iterator end() const
   {
      return v.begin() + end_idx;
   }

   const T& operator[](size_t index) const
   {
      return v[begin_idx + index];
   }

   size_t size() const
   {
      return end_idx - begin_idx;
   }
};

template<typename T>
auto make_range_view(const std::vector<T>& vector, size_t begin_index, size_t end_index)
{
   return vector_range_view<T>(vector, begin_index, end_index);
}

} // namespace common
} // namespace polatory
