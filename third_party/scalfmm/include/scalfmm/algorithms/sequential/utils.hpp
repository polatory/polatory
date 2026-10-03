// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/sequential/utils.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_SEQUENTIAL_UTILS_HPP
#define SCALFMM_ALGORITHMS_SEQUENTIAL_UTILS_HPP

#include "scalfmm/tree/group_tree_view.hpp"

namespace scalfmm::algorithms::sequential
{
    /**
    * @brief Resets to zero the particles in the tree.
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_particles(TreeType& tree) -> void
    {
        tree.reset_particles();
    }

    /**
    * @brief Resets to zero the positions of the particles in the tree.
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_positions(TreeType& tree) -> void
    {
        tree.reset_positions();
    }

    /**
    * @brief Resets to zero the inputs of the particles in the tree.
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_inputs(TreeType& tree) -> void
    {
        tree.reset_inputs();
    }

    /**
    * @brief Resets to zero the outputs of the particles in the tree.
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_outputs(TreeType& tree) -> void
    {
        tree.reset_outputs();
    }

    /**
    * @brief Resets to zero the variables of the particles in the tree.
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_variables(TreeType& tree) -> void
    {
        tree.reset_variables();
    }

    /**
    * @brief Reset to zero the multipole and the local values in the cells.
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_far_field(TreeType& tree) -> void
    {
        tree.reset_far_field();
    }

    /**
    * @brief Reset to zero the multipole values in the cells.
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_multipoles(TreeType& tree) -> void
    {
        tree.reset_multipoles();
    }

    /**
    * @brief Reset to zero the local values in the cells.
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_locals(TreeType& tree) -> void
    {
        tree.reset_locals();
    }
}   // namespace scalfmm::algorithms::sequential

#endif
