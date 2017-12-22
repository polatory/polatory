// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <polatory/common/uncertain.hpp>

using polatory::common::uncertain;

namespace {

enum Result {
  False, True, Uncertain
};

void test(Result expected, uncertain<bool> actual) {
  switch (expected) {
  case False:
    ASSERT_TRUE(actual.is_certain());
    ASSERT_FALSE(actual.get());
    break;
  case True:
    ASSERT_TRUE(actual.is_certain());
    ASSERT_TRUE(actual.get());
    break;
  case Uncertain:
    ASSERT_FALSE(actual.is_certain());
    break;
  }
}

} // namespace

TEST(uncertain, operators) {
  auto f = uncertain<bool>(false);
  auto t = uncertain<bool>(true);
  auto u = uncertain<bool>();

  test(False, f);
  test(True, t);
  test(Uncertain, u);

  test(True, !f);
  test(False, !t);
  test(Uncertain, !u);

  test(False, f || f);
  test(True, f || t);
  test(Uncertain, f || u);
  test(True, t || f);
  test(True, t || t);
  test(True, t || u);
  test(Uncertain, u || f);
  test(True, u || t);
  test(Uncertain, u || u);

  test(False, f && f);
  test(False, f && t);
  test(False, f && u);
  test(False, t && f);
  test(True, t && t);
  test(Uncertain, t && u);
  test(False, u && f);
  test(Uncertain, u && t);
  test(Uncertain, u && u);
}

TEST(uncertain, predicates) {
  auto f = uncertain<bool>(false);
  auto t = uncertain<bool>(true);
  auto u = uncertain<bool>();

  ASSERT_FALSE(certainly(f));
  ASSERT_TRUE(certainly(t));
  ASSERT_FALSE(certainly(u));

  ASSERT_TRUE(certainly_not(f));
  ASSERT_FALSE(certainly_not(t));
  ASSERT_FALSE(certainly_not(u));

  ASSERT_FALSE(possibly(f));
  ASSERT_TRUE(possibly(t));
  ASSERT_TRUE(possibly(u));

  ASSERT_TRUE(possibly_not(f));
  ASSERT_FALSE(possibly_not(t));
  ASSERT_TRUE(possibly_not(u));
}
