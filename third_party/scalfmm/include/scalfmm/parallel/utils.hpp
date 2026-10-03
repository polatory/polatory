// --------------------------------
// See LICENCE file at project root
// File : scalfmm/parallel/utils.hpp
// --------------------------------
#pragma once

#include "scalfmm/utils/math.hpp"

namespace scalfmm::parallel::utils
{
    /**
     * @brief find the processor owning the index
     *
     * @tparam MortonIdx
     * @tparam MortonDistribution
     * @param[in] index
     * @param[in] distrib the index distribution
     * @param[in] start [optional]position to start in the distribution vector
     * @return the process number
     */
    template<typename MortonIdx, typename MortonDistribution>
    inline auto find_proc_for_index(const MortonIdx& index, const MortonDistribution& distrib,
                                    std::size_t start = 0) -> int
    {
        for(std::size_t i = start; i < distrib.size(); ++i)
        {
            if(math::between(index, distrib[i][0], distrib[i][1]))
            {
                return i;
            }
        }
        return -1;
    }

    /**
     * @brief  get theoretical p2p interaction list outside me
     *
     * We return the list of indexes of cells involved in P2P interaction that we do
     *  not have locally.  The cells on other processors may not exist.
     *
     * @tparam MortonIdx
     * @tparam MortonDistribution
     * @param morton_idx
     * @param leaves_distrib the leaves distribution on the processes.
     * @return the list of indexes on other processes
     */
    template<typename MortonIdx, typename MortonDistribution>
    inline auto is_inside_distrib(MortonIdx morton_idx, MortonDistribution const& leaves_distrib) -> bool
    {
        for(auto const& interval: leaves_distrib)
        {
            if(morton_idx > interval[1])
            {
                continue;
            }
            else if(morton_idx >= interval[0])
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        return false;
    }

    /**
     * @brief Check if morton index is inside the vector of distribution starting at start +1
     *
     * @tparam MortonIdx
     * @tparam MortonDistribution
     * @param morton_idx the index
     * @param start  We begin to we check in the distribution at position start+1
     * @param component_distrib the distribution of cells/leaves on the processes
     * @return true if morton_ids is inside otherwise false
     */
    template<typename MortonIdx, typename MortonDistribution>
    inline auto is_inside_distrib_right(MortonIdx morton_idx, int const& start,
                                        MortonDistribution const& component_distrib) -> bool
    {
        if(find_proc_for_index(morton_idx, component_distrib, start + 1) > 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    /**
     * @brief Check if morton index is inside the vector of distribution between 0 and end - 1
     *
     * @tparam MortonIdx
     * @tparam MortonDistribution
     * @param morton_idx the index
     * @param end  We check in the distribution between 0 and end -1
     * @param component_distrib the distribution of cells/leaves on the processes
     * @return true if morton_ids is inside otherwise false
     */
    template<typename MortonIdx, typename MortonDistribution>
    inline auto is_inside_distrib_left(MortonIdx morton_idx, int const& end,
                                       MortonDistribution const& component_distrib) -> bool
    {
        for(int i = end - 1; i >= 0; --i)
        {
            auto const& interval = component_distrib[i];
            if(math::between(morton_idx, interval[0], interval[1]))
            {
                return true;
            }
            else if(morton_idx >= interval[1])
            {
                return false;
            }
        }
        return false;
    }

    /**
     * @brief Get the proc id of the morton index
     *
     * @tparam MortonType
     * @tparam VectorDistrib
     * @param morton_idx  mordon index
     * @param distrib a vector of distribution (number of processes size)
     * @return the process id containing the Morton index
     */
    template<typename MortonType, typename VectorDistrib>
    inline auto get_proc_id(MortonType const& morton_idx, VectorDistrib const& distrib) -> int
    {
        int idx_proc{0};
        for(auto dist: distrib)
        {
            if(math::between(morton_idx, dist[0], dist[1]))
            {
                return idx_proc;
            }
            ++idx_proc;
        }

        return -1;
    }

    /**
     * @brief Set the data in leaf object
     *
     * @tparam VectorStructType
     * @tparam VectorType
     * @param dim
     * @param leaf_to_receive_access
     * @param buffer
     */
    template<typename VectorStructType, typename VectorType>
    inline auto set_data_in_leaf(int dim, VectorStructType const& leaf_to_receive_access,
                                 VectorType const& buffer) -> void
    {
        auto const& elt = leaf_to_receive_access;
        int nb_elt{0};   // size of the  buffer for one coordinate
        for(auto i = 0; i < elt.size(); ++i)
        {
            auto leaf = (*(elt[i].first))->component(elt[i].second);
            nb_elt += leaf.size();
        }
        auto begin_buffer = std::begin(buffer);
        for(auto i = 0; i < elt.size(); ++i)
        {
            auto leaf = (*(elt[i].first))->component(elt[i].second);
            // leaf[0] return a particle proxy on the first particle
            nb_elt = leaf.size();
            auto end_buffer = begin_buffer + nb_elt;
            for(int k = 0; k < dim; ++k)
            {
                auto ptr_coord = &(leaf[0].position()[k]);
                auto begin_buffer_k = begin_buffer + k * nb_elt;
                auto end_buffer_k = end_buffer + k * nb_elt;
                std::copy(begin_buffer_k, end_buffer_k, ptr_coord);
            }
            begin_buffer = end_buffer;
        }
    }
}   // namespace scalfmm::parallel::utils
