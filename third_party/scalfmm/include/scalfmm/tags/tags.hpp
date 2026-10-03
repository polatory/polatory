// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tags/tags.hpp
// --------------------------------
#pragma once

namespace scalfmm
{
    namespace uniform::tags
    {
        // Registered interpolator tags
        struct generic
        {
        };
        struct uniform
        {
        };
        struct chebyshev
        {
        };
    }   // namespace uniform::tags

    namespace operators::tags
    {
        struct with_forces
        {
        };
        struct without_forces
        {
        };
        struct inner
        {
        };
        struct outer
        {
        };
        struct full_mutual
        {
        };
    }   // namespace operators::tags

    namespace models::tags
    {
        struct tsm
        {
        };
        struct ssm
        {
        };
    }   // namespace models::tags
}   // namespace scalfmm