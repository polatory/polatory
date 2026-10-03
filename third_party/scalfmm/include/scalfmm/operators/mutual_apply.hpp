// --------------------------------
// See LICENCE file at project root
// File : operators/mutual_apply.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_MUTUAL_APPLY_HPP
#define SCALFMM_OPERATORS_MUTUAL_APPLY_HPP

#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/p2p.hpp"

namespace scalfmm::operators
{
    /**
     * @brief  This function handles out-of-group interactions for the current group
     *  and applies the p2p_full_mutual operator between the leaves of two different groups.
     *
     * @tparam GroupType
     * @tparam MatrixKernelType
     * @tparam ArrayType
     * @tparam ValueType
     * @param group   The group on which  we want to compute the interactions
     * @param matrix_kernel   the matrix kernel you need to pass over to the operator
     * @param[in] pbc   array of periodicity in each direction
     * @param[in] box_width   the maximal width of simulation box
     */
    template<typename GroupType, typename MatrixKernelType, typename ArrayType, typename ValueType>
    void apply_out_of_group_p2p(GroupType& group, MatrixKernelType const& matrix_kernel, ArrayType const& pbc,
                                ValueType const& box_width)
    {
        using operators::p2p_full_mutual;
        // Get interactions outside of the current group.
        const auto& outside_interactions = group.symbolics().outside_interactions;

        for(auto interact: outside_interactions)
        {
            // get the particles container of inside block interaction
            auto& current_leaf = group.component(interact.inside_index_in_block);
            // set the iterator of outside block interaction in vector
            std::array<typename GroupType::symbolics_type::iterator_source_type, 1> neighbors_keep{
              interact.outside_iterator};
            p2p_full_mutual(matrix_kernel, current_leaf, neighbors_keep, 1, pbc, box_width);
        }
    }

    /**
     * @brief  This function applies some out-of-group interactions for the current group.
     *
     * The goal of this function is to apply the interactions only on 2 groups to split
     *  the dependencies in OMP parallelization.
     *
     * @tparam GroupType
     * @tparam MatrixKernelType
     * @tparam ArrayType
     * @tparam ValueType
     * @param[inout] group:  The group on which  we want to compute the interactions
     * @param[in] first_out  the first iteration in the out-of-block iterations to treat
     * @param[in] last_out  the last iteration in the out-of-block iterations to treat
     * @param[in] matrix_kernel the matrix kernel you need to pass over to the operator
     * @param[in] pbc   array of periodicity in each direction
     * @param[in] box_width : the maximal width of simulation box
     */
    template<typename GroupType, typename MatrixKernelType, typename ArrayType, typename ValueType>
    void apply_out_of_group_p2p(GroupType& group, const int& first_out, const int& last_out,
                                MatrixKernelType const& matrix_kernel, ArrayType const& pbc, ValueType const& box_width)
    {
        using operators::p2p_full_mutual;
        // constexpr std::size_t sizeNeig = (std::pow(static_cast<const int>(3), Dimension) - static_cast<const
        // int>(1));
        // Get interactions outside of the current group.
        auto& outside_interactions = group.symbolics().outside_interactions;
#ifdef NEW_P2P_OUT_GRP
        static constexpr std::size_t Dimension = GroupType::component_type::dimension;
        constexpr std::size_t sizeNeig = (meta::Pow<3, Dimension>::value - 1);
        // Try to use a single call to perform p2p between leaves in the outer group with the same Morton index (inside)
        // as the leaf within the group.
        auto current_pos = first_out;
        auto morton_index = outside_interactions[current_pos].inside_index;
        std::array<typename GroupType::iterator_type, sizeNeig> neighbors_keep;
        // Start loop on the interactions inside the same group
        while(current_pos != last_out)
        {
            morton_index = outside_interactions[current_pos].inside_index;
            int number_of_leaves = 0;
            auto& interact = outside_interactions[current_pos];
            auto& current_leaf = group.component(interact.inside_index_in_block);
            //
            // add in neighbors_keep all the outside leaf
            while(outside_interactions[current_pos].inside_index == morton_index && current_pos < last_out)
            {
                neighbors_keep[number_of_leaves] = outside_interactions[current_pos].outside_iterator;
                ++number_of_leaves;
                ++current_pos;
            }
            // std::cout << " Morton   " << morton_index << "  -> nb " << number_of_leaves << std::endl;
            // if(number_of_leaves > 14)
            // {
            //     std::cout << "first   " << first_out << " last " << last_out << "    " << last_out << std::endl;
            //     for(int k = first_out; k < last_out; ++k)
            //     {
            //         std::cout << k << " " << outside_interactions[k] << std::endl;
            //     }
            //     std::exit(-1);
            // }
            p2p_full_mutual(matrix_kernel, current_leaf, neighbors_keep, number_of_leaves, pbc, box_width);
        }
#else

        for(int i = first_out; i < last_out; ++i)
        {
            auto& interact = outside_interactions.at(i);
            // get the particles container of inside block interaction
            auto& current_leaf = group.component(interact.inside_index_in_block);
            // set the iterator of outside block interaction in vector
            std::array<typename GroupType::iterator_type, 1> neighbors_keep{interact.outside_iterator};
            p2p_full_mutual(matrix_kernel, current_leaf, neighbors_keep, 1, pbc, box_width);
        }
#endif
    }

    /**
     * @brief  This function applies some out-of-group interactions for the current group.
     *
     * The goal of this function is to apply the interactions only on 2 groups to split
     *  the dependencies in OMP parallelization. The interactions is between the group and
     *  the group with Morton indexes between [first_morton_index, last_morton_index[
     *
     * Maybe unused
     *
     * @tparam GroupType
     * @tparam MatrixKernelType
     * @tparam MortonType
     * @tparam ArrayType
     * @tparam ValueType
     * @param[inout] group:  The group on which  we want to compute the interactions
     * @param[in] start  th index to start the iterations inside the group
     * @param[in] first_morton_index  the first morton  index of the group to treat
     * @param[in] last_morton_index  the last morton  index of the group to treat
     * @param[in] matrix_kernel the matrix kernel you need to pass over to the operator
     * @param[in] pbc   array of periodicity in each direction
     * @param[in] box_width : the maximal width of simulation box
     */
    template<typename GroupType, typename MatrixKernelType, typename MortonType, typename ArrayType, typename ValueType>
    void apply_out_of_group_p2p(GroupType& group, int& start, MortonType const& first_morton_index,
                                MortonType const& last_morton_index, MatrixKernelType const& matrix_kernel,
                                ArrayType const& pbc, ValueType const& box_width)
    {
        using operators::p2p_full_mutual;
        // Get interactions outside of the current group.
        auto& outside_interactions = group.symbolics().outside_interactions;

        for(int i = start; i < outside_interactions.size(); ++i)
        {
            auto& interact = outside_interactions.at(i);
            // if(interact.outside_index_in_block < first_morton_index)
            {
                // continue;

                if(interact.outside_index_in_block < last_morton_index)
                {
                    // get the particles container of inside block interaction
                    auto& current_leaf = group.component(interact.inside_index_in_block);
                    // set the iterator of outside block interaction in vector
                    std::array<typename GroupType::iterator_type, 1> neighbors_keep{interact.outside_iterator};
                    p2p_full_mutual(matrix_kernel, current_leaf, neighbors_keep, 1, pbc, box_width);
                }
                else
                {
                    start = i;
                    break;
                }
            }
        }
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_MUTUAL_APPLY_HPP
