// --------------------------------
// See LICENCE file at project root
// File : scalfmm/utils/static_assert_as_exception.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_STATIC_ASSERT_AS_EXCEPTION_H
#define SCALFMM_UTILS_STATIC_ASSERT_AS_EXCEPTION_H

/* Conditionally compilable apparatus for replacing `static_assert`
    with a runtime exception of type `exceptionalized_static_assert`
    within (portions of) a test suite.
    Code from Mike kinghan on Stackoverflow ;)
*/
#ifdef SCALFMM_TEST_EXCEPTIONALIZE_STATIC_ASSERT

#include <stdexcept>
#include <string>

namespace scalfmm::test
{
    /**
     * @brief
     *
     */
    struct exceptionalized_static_assert : std::logic_error
    {
        exceptionalized_static_assert(char const* what)
          : std::logic_error(what) {};
        virtual ~exceptionalized_static_assert() noexcept {}
    };

    /**
     * @brief
     *
     * @tparam Cond
     */
    template<bool Cond>
    struct exceptionalize_static_assert;

    /**
     * @brief
     *
     * @tparam
     */
    template<>
    struct exceptionalize_static_assert<true>
    {
        explicit exceptionalize_static_assert(char const* reason) { (void)reason; }
    };

    /**
     * @brief
     *
     * @tparam
     */
    template<>
    struct exceptionalize_static_assert<false>
    {
        explicit exceptionalize_static_assert(char const* reason)
        {
            std::string s("static_assert would fail with reason: ");
            s += reason;
            throw exceptionalized_static_assert(s.c_str());
        }
    };

}   // namespace scalfmm::test

// A macro redefinition of `static_assert`
#define static_assert(cond, gripe)                                                                                     \
    struct _1_test : scalfmm::test::exceptionalize_static_assert<cond>                                                 \
    {                                                                                                                  \
        _1_test()                                                                                                      \
          : scalfmm::test::exceptionalize_static_assert<cond>(gripe) {};                                               \
    };                                                                                                                 \
    _1_test _2_test

#endif   // SCALFMM_TEST_EXCEPTIONALIZE_STATIC_ASSERT == 1

#endif   // EOF
