// --------------------------------
// See LICENCE file at project root
// File : scalfmm/utils/source_target.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_SOURCE_TARGET_HPP
#define SCALFMM_UTILS_SOURCE_TARGET_HPP

#include <scalfmm/tree/box.hpp>
#include <scalfmm/tree/for_each.hpp>

namespace scalfmm::utils
{
    /**
     * @brief Construct the geometric bounding box of box1 and box2
     *
     * @tparam BoxType
     * @param[in] box1 first box containing particles (sources)
     * @param[in] box2 second box containing particles (targets)
     * @return the box containing all particles (sources and targets)
     */
    template<typename BoxType>
    inline auto bounding_box(BoxType const& box1, BoxType const& box2) -> BoxType
    {
        using position_type = typename BoxType::position_type;
        using value_type = typename position_type::value_type;
        constexpr static const std::size_t dimension = position_type::dimension;

        const auto& box1_c1 = box1.c1();
        const auto& box1_c2 = box1.c2();
        const auto& box2_c1 = box2.c1();
        const auto& box2_c2 = box2.c2();
        position_type c1{}, c2{};
        for(std::size_t i = 0; i < dimension; ++i)
        {
            c1.at(i) = std::min(box1_c1.at(i), box2_c1.at(i));
            c2.at(i) = std::max(box1_c2.at(i), box2_c2.at(i));
        }

        auto diff = c2 - c1;
        value_type length{diff.max()};
        c2 = c1 + length;

        return BoxType(c1, c2);
    }

    /**
     * @brief Merge two containers of particles
     *
     * @tparam ContainerType
     * @param container1 first container
     * @param container2 second container
     * @return the merge of the two containers
     */
    template<typename ContainerType>
    inline auto merge(ContainerType const& container1, ContainerType const& container2) -> ContainerType
    {
        auto size = container1.size() + container2.size();
        ContainerType output(size);
        std::size_t idx = 0;
        for(std::size_t i = 0; i < container1.size(); ++i, ++idx)
        {
            output.insert_particle(idx, container1.particle(i));
        }
        for(std::size_t i{0}; i < container2.size(); ++i, ++idx)
        {
            output.insert_particle(idx, container2.particle(i));
        }
        return output;
    }

    /**
     * @brief
     *
     * @todo
     *
     * @tparam TreeType
     * @tparam IndexType
     * @param tree
     * @param index
     * @param level
     * @return auto
     */
    template<typename TreeType, typename IndexType>
    inline auto find_cell_group(TreeType const& tree, IndexType index, int& level)
    {
        std::size_t id_group{0};
        throw("find_cell_group: not implemented yet !");
    }

}   // namespace scalfmm::utils
#endif
