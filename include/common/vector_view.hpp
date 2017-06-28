// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <boost/operators.hpp>

namespace polatory {
namespace common {

template<typename T>
class vector_view_iterator;

template<typename T>
class vector_view {
   friend class vector_view_iterator<T>;

   const std::vector<T>& v;
   const std::vector<size_t>& idcs;

public:
   typedef vector_view_iterator<T> iterator;

   vector_view(const std::vector<T>& vector, const std::vector<size_t>& indices)
      : v(vector)
      , idcs(indices)
   {
   }

   iterator begin() const
   {
      return vector_view_iterator<T>(*this, 0);
   }

   iterator end() const
   {
      return vector_view_iterator<T>(*this, idcs.size());
   }

   const T& operator[](size_t index) const
   {
      return v[idcs[index]];
   }

   size_t size() const
   {
      return idcs.size();
   }
};

template<typename T>
class vector_view_iterator
   : public boost::bidirectional_iterator_helper<vector_view_iterator<T>, T> {

   const vector_view<T>& view;
   size_t idx;

public:
   vector_view_iterator(const vector_view<T>& view, size_t index)
      : view(view)
      , idx(index)
   {
   }

   bool operator==(const vector_view_iterator& other) const
   {
      return idx == other.idx;
   }

   vector_view_iterator& operator++()
   {
      assert(idx < view.idcs.size());
      idx++;
      return *this;
   }

   vector_view_iterator& operator--()
   {
      assert(idx > 0);
      idx--;
      return *this;
   }

   const T& operator*() const
   {
      return view.v[view.idcs[idx]];
   }
};

template<typename T>
auto make_view(const std::vector<T>& vector, const std::vector<size_t>& indices)
{
   return vector_view<T>(vector, indices);
}

} // namespace common
} // namespace polatory
