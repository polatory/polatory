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
public:
   using iterator = vector_view_iterator<T>;

   vector_view(const std::vector<T>& vector, const std::vector<size_t>& indices)
      : v_(vector)
      , idcs_(indices)
   {
   }

   iterator begin() const
   {
      return vector_view_iterator<T>(*this, 0);
   }

   bool empty() const
   {
      return idcs_.empty();
   }

   iterator end() const
   {
      return vector_view_iterator<T>(*this, idcs_.size());
   }

   const T& operator[](size_t index) const
   {
      return v_[idcs_[index]];
   }

   size_t size() const
   {
      return idcs_.size();
   }

private:
   friend class vector_view_iterator<T>;

   const std::vector<T>& v_;
   const std::vector<size_t>& idcs_;
};

template<typename T>
class vector_view_iterator : public boost::random_access_iterator_helper<vector_view_iterator<T>, T> {
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
      assert(idx_ < view_.idcs_.size());
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
      return view_.v_[view_.idcs_[idx_]];
   }

   bool operator<(const vector_view_iterator& other) const
   {
      return idx_ < other.idx_;
   }

   vector_view_iterator& operator+=(typename vector_view_iterator::difference_type n)
   {
      idx_ += n;
      return *this;
   }

   vector_view_iterator& operator-=(typename vector_view_iterator::difference_type n)
   {
      idx_ -= n;
      return *this;
   }

   friend typename vector_view_iterator::difference_type operator-(const vector_view_iterator& lhs, const vector_view_iterator& rhs)
   {
      assert(std::addressof(lhs.view_) == std::addressof(rhs.view_));
      return lhs.idx_ - rhs.idx_;
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
