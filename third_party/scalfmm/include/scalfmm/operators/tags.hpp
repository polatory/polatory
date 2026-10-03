// --------------------------------
// See LICENCE file at project root
// File : scalfmm/operators/tags.hpp
// --------------------------------
#pragma once

namespace scalfmm::operators::impl
{
    /**
    * @brief Tag dispatching for FMM operators.
    */

    // Common base type for tag dispatching
    struct tag
    {
    };

    // Tag for the P2M operator
    struct tag_p2m : tag
    {
    };

    // Tag for the M2M operator
    struct tag_m2m : tag
    {
    };

    // Tag for the M2L operator
    struct tag_m2l : tag
    {
    };

    // Tag for the L2L operator
    struct tag_l2l : tag
    {
    };

    // Tag for the L2P operator
    struct tag_l2p : tag
    {
    };

    // Tag for the M2P operator
    struct tag_m2p : tag
    {
    };

    // Tag for the P2L operator
    struct tag_p2l : tag
    {
    };

    // Tag for the P2P operator
    struct tag_p2p : tag
    {
    };
}   // namespace scalfmm::operators::impl
