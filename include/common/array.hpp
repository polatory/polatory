// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <array>
#include <utility>

namespace polatory {
namespace common {

template<class T>
std::array<T, 0> make_array()
{
   return { };
}

template<class Head, class... Tail, class T = typename std::decay<Head>::type>
std::array<T, 1 + sizeof...(Tail)> make_array(Head&& head, Tail&&... tail)
{
   return { std::forward<Head>(head), std::forward<Tail>(tail)... };
}

} // namespace common
} // namespace polatory
