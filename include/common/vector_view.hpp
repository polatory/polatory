// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <memory>
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
   : public boost::random_access_iterator_helper<vector_view_iterator<T>, T> {
public:
   vector_view_iterator(const vector_view<T>& view, size_t index)
      : view_(view)
      , idx_(index)
   {
   }

   bool operator==(const vector_view_iterator& other) const
   {
      return idx_ == other.idx_;
   }

   vector_view_iterator& operator++()
   {
      assert(idx_ < view_.idcs.size());
      idx_++;
      return *this;
   }

   vector_view_iterator& operator--()
   {
      assert(idx_ > 0);
      idx_--;
      return *this;
   }

   const T& operator*() const
   {
      return view_.v[view_.idcs[idx_]];
   }

   bool operator<(const vector_view_iterator& other) const
   {
      return idx_ < other.idx_;
   }

   vector_view_iterator& operator+=(difference_type n)
   {
      idx_ += n;
      return *this;
   }

   vector_view_iterator& operator-=(difference_type n)
   {
      idx_ -= n;
      return *this;
   }

   friend difference_type operator-(const vector_view_iterator& x, const vector_view_iterator& y)
   {
      assert(std::addressof(x.view_) == std::addressof(y.view_));
      return x.idx_ - y.idx_;
   }

private:
   const vector_view<T>& view_;
   size_t idx_;
};

template<typename T>
auto make_view(const std::vector<T>& vector, const std::vector<size_t>& indices)
{
   return vector_view<T>(vector, indices);
}

} // namespace common
} // namespace polatory
