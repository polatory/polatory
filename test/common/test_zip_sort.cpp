// Copyright (c) 2016, GSI and The Polatory Authors.

#include <vector>

#include <gtest/gtest.h>

#include "polatory/common/zip_sort.hpp"

using namespace polatory::common;

TEST(zip_sort, sort_by_first) {
  std::vector<int> a{ 4, 2, 5, 1, 3 };
  std::vector<double> b{ 4., 2., 5., 1., 3. };
  std::vector<int> a_sorted{ 1, 2, 3, 4, 5 };
  std::vector<double> b_sorted{ 1., 2., 3., 4., 5. };

  zip_sort(a.begin(), a.end(), b.begin(), b.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

  ASSERT_EQ(a_sorted, a);
  ASSERT_EQ(b_sorted, b);
}

TEST(zip_sort, sort_by_second) {
  std::vector<int> a{ 4, 2, 5, 1, 3 };
  std::vector<double> b{ 4., 2., 5., 1., 3. };
  std::vector<int> a_sorted{ 1, 2, 3, 4, 5 };
  std::vector<double> b_sorted{ 1., 2., 3., 4., 5. };

  zip_sort(a.begin(), a.end(), b.begin(), b.end(), [](const auto& a, const auto& b) { return a.second < b.second; });

  ASSERT_EQ(a_sorted, a);
  ASSERT_EQ(b_sorted, b);
}
