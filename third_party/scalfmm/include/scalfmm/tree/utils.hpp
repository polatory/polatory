// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tree/utils.hpp
// --------------------------------
#ifndef SCALFMM_TREE_UTILS_HPP
#define SCALFMM_TREE_UTILS_HPP

#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <tuple>
#include <type_traits>

#include "scalfmm/container/point.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"

// namespace scalfmm::utils
namespace scalfmm::index
{
    namespace impl
    {
        /**
         * @brief
         *
         * @tparam RelativePositionType
         * @tparam IndexType
         * @param coords
         * @return IndexType
         */
        template<typename RelativePositionType, typename IndexType = std::size_t>
        auto build_morton_index(RelativePositionType& coords) -> IndexType
        {
            using dim_t = typename RelativePositionType::size_type;
            constexpr dim_t Dim = std::tuple_size<RelativePositionType>::value;
            IndexType mask = 1;
            IndexType idx = 0;
            auto not_done = [&]
            {
                if(mask == 0)
                {
                    return false;
                }
                for(dim_t i = 0U; i < Dim; ++i)
                {
                    if((mask << i) <= coords.at(i))
                    {
                        return true;
                    }
                }
                return false;
            };

            while(not_done())
            {
                for(dim_t i = 0U; i < Dim; ++i)
                {
                    idx |= (coords.at(i) & mask);
                    mask <<= 1;
                    coords.at(i) <<= Dim - 1;
                }
            }
            return idx;
        }
    }   // namespace impl

    /**
     * @brief get the morton index of point pos inside box at level
     *
     * @tparam IndexType
     * @tparam BoxType
     * @tparam PositionType
     * @param pos the real coordinate of the point inside the box
     * @param box   the box containing the 2^d tree
     * @param level the level to compute the morton index
     * @return the morton index
     */
    template<typename IndexType = std::size_t, typename BoxType, typename PositionType>
    auto get_morton_index(const PositionType& pos, const BoxType& box, const std::size_t level) -> IndexType
    {
        constexpr static const std::size_t Dim = PositionType::dimension;
        using dim_t = std::size_t;

        // we set double type rather than Position::value_type for a better accuracy
        double cell_width = box.width(0) / (static_cast<IndexType>(1) << level);
        std::array<IndexType, Dim> coords{};

        for(dim_t i = 0; i < Dim; ++i)
        {
            coords.at(Dim - i - 1) = static_cast<IndexType>((pos.at(i) - box.c1().at(i)) / cell_width);
            coords.at(Dim - i - 1) <<= Dim - i - dim_t(1);
        }
        // std::cout << "   get_morton_index.    pos" << pos << " " << coords.at(0) << " " << coords.at(1) << " " << box
        //           << "   " << cell_width << std::endl;
        return impl::build_morton_index(coords);
    }

    /**
     * @brief get morton index from a relative position
     *
     * @tparam PositionType
     * @tparam IndexType
     * @param[in] pos the relative position in the d-grid. array of d index
     * @return  the morton index
     */
    template<typename PositionType, typename IndexType = std::size_t>
    auto get_morton_index(const PositionType& pos) -> IndexType
    {
        constexpr static const std::size_t Dim = PositionType::dimension;
        using dim_t = std::size_t;

        std::array<IndexType, Dim> coords{};

        for(dim_t i = 0; i < Dim; ++i)
        {
            coords.at(Dim - i - 1) = static_cast<IndexType>(pos.at(i)) << (Dim - i - 1);
        }
        return impl::build_morton_index(coords);
    }

    /**
     * @brief Get the tree coordinate object
     *
     * @tparam ValueType
     * @tparam CoordinateType
     * @param relative_position
     * @param box_width
     * @param box_width_at_leaf_level
     * @param tree_height
     * @return std::enable_if_t<std::is_integral_v<CoordinateType>, CoordinateType>
     */
    template<typename ValueType, typename CoordinateType = std::int64_t>
    inline auto
    get_tree_coordinate(ValueType relative_position, ValueType box_width, ValueType box_width_at_leaf_level,
                        std::size_t tree_height) -> std::enable_if_t<std::is_integral_v<CoordinateType>, CoordinateType>
    {
        assertm((relative_position >= 0 && relative_position <= box_width),
                "get_tree_coordinate : relative_position out of box");
        if(relative_position == box_width)
        {
            return static_cast<CoordinateType>(math::pow((tree_height - 1) - 1, 2));
        }
        return static_cast<CoordinateType>(relative_position / box_width_at_leaf_level);
    }

    /**
     * @brief Get the coordinate from position and corner object
     *
     * @tparam ValueType
     * @tparam CoordinateType
     * @tparam Dimension
     * @param position
     * @param corner_of_box
     * @param box_width
     * @param tree_height
     * @return std::enable_if_t<std::is_integral_v<CoordinateType>, container::point<CoordinateType, Dimension>>
     */
    template<typename ValueType, typename CoordinateType = std::int64_t, std::size_t Dimension>
    inline auto get_coordinate_from_position_and_corner(container::point<ValueType, Dimension> const& position,
                                                        container::point<ValueType, Dimension> const& corner_of_box,
                                                        ValueType box_width, std::size_t tree_height)
      -> std::enable_if_t<std::is_integral_v<CoordinateType>, container::point<CoordinateType, Dimension>>
    {
        const ValueType box_width_at_leaf_level{box_width /
                                                ValueType(static_cast<std::size_t>(1) << (tree_height - 1))};

        // box coordinate to host the particle
        container::point<CoordinateType, Dimension> host{};
        // position has to be relative to corner not center
        auto tmp = position - corner_of_box;
        meta::for_each(host, tmp,
                       [box_width, box_width_at_leaf_level, tree_height](ValueType p) {
                           return get_tree_coordinate<ValueType, CoordinateType>(p, box_width, box_width_at_leaf_level,
                                                                                 tree_height);
                       });
        return host;
    }

    /**
     * @brief Get the position from coordinate object
     *
     * @tparam ValueType
     * @tparam CoordinateType
     * @tparam Dimension
     * @param coordinate
     * @param center_of_box
     * @param box_width
     * @param tree_height
     * @return std::enable_if_t<std::is_integral_v<CoordinateType>, container::point<ValueType, Dimension>>
     */
    template<typename ValueType, typename CoordinateType, std::size_t Dimension>
    inline auto get_position_from_coordinate(container::point<CoordinateType, Dimension> const& coordinate,
                                             container::point<ValueType, Dimension> const& center_of_box,
                                             ValueType box_width, std::size_t tree_height)
      -> std::enable_if_t<std::is_integral_v<CoordinateType>, container::point<ValueType, Dimension>>
    {
        const container::point<ValueType, Dimension> box_corner{center_of_box - (box_width / ValueType(2))};
        const ValueType box_width_at_leaf_level{box_width /
                                                ValueType(static_cast<std::size_t>(1) << (tree_height - 1))};

        // box coordinate to host the particle
        container::point<ValueType, Dimension> host{coordinate * box_width_at_leaf_level};
        return host + box_corner;
    }

    /**
     * @brief Get the coordinate from morton index object
     *
     * @tparam Dimension
     * @tparam MortonIndex
     * @tparam CoordinateType
     * @param morton_index
     * @return std::enable_if_t<std::is_integral_v<CoordinateType>, container::point<CoordinateType, Dimension>>
     */
    template<std::size_t Dimension, typename MortonIndex = std::size_t, typename CoordinateType = std::int64_t>
    inline auto get_coordinate_from_morton_index(MortonIndex morton_index)
      -> std::enable_if_t<std::is_integral_v<CoordinateType>, container::point<CoordinateType, Dimension>>
    {
        using coordinate_type = container::point<CoordinateType, Dimension>;

        std::size_t mask = 0x1LL;

        coordinate_type coord(CoordinateType(0));

        while(morton_index >= mask)
        {
            for(int dim = static_cast<int>(Dimension - 1); dim >= 0; --dim)
            {
                if(dim == 0)
                {
                    coord.at(static_cast<std::size_t>(dim)) |= CoordinateType(morton_index & mask);
                }
                else
                {
                    coord.at(static_cast<std::size_t>(dim)) |= CoordinateType(morton_index & mask);
                    morton_index >>= 1;
                }
            }
            mask <<= 1;
        }
        return coord;
    }

    /**
     * @brief get_grid_index return the grid coordinate of a linear index
     *
     * get_grid_index return the grid coordinate of a linear index
     *   between 0 and nbNeigPerDim^dimension in   3x3 grid centered in 0.
     *
     * Each component od the grid index is between -a and a where a =
     * (nbNeigPerDim-1)/2, i.e. the number of neighbors
     *   in on direction (generally when we consider the first neighbors a = 1)
     *
     * @tparam Dimension
     * @tparam MortonIndex
     * @tparam CoordinateType
     * @param idx the linear index
     * @param nbNeigPerDim  the number of points per line
     * @return and array of size dimension
     */
    template<std::size_t Dimension, typename MortonIndex = std::size_t, typename CoordinateType = std::int64_t>
    inline auto get_grid_index(MortonIndex& idx, const int nbNeigPerDim) ->
      typename container::point<CoordinateType, Dimension>
    {
        container::point<CoordinateType, Dimension> coordinate{};
        auto tmp = idx;
        auto nbEltperDim = nbNeigPerDim;
        for(auto d = 0; d < Dimension - 1; ++d)
        {
            coordinate[d] = CoordinateType(tmp / nbEltperDim);
            tmp -= (coordinate[d]) * nbEltperDim;
            nbEltperDim /= nbNeigPerDim;
        }
        coordinate[Dimension - 1] = tmp;

        return coordinate;
    }

    /**
     * @brief
     *
     * @tparam Dimension
     * @tparam CoordinatePointType
     * @tparam ArrayType
     * @param coord
     * @param period
     * @param limite1d
     * @return true
     * @return false
     */
    template<std::size_t Dimension, typename CoordinatePointType, typename ArrayType>
    auto check_limit(CoordinatePointType& coord, const ArrayType& period, const int& limite1d) -> bool
    {
        using CoordinateType = typename CoordinatePointType::value_type;
        bool check = true;
        for(std::size_t d = 0; d < Dimension; ++d)
        {
            if(period[d])
            {
                if(coord[d] < 0)
                {
                    coord[d] += limite1d;
                }
                else if(coord[d] > limite1d - 1)
                {
                    coord[d] -= limite1d;
                }
            }
            else
            {
                check = check && scalfmm::math::between(coord[d], CoordinateType(0), CoordinateType(limite1d));
            }
        }
        return check;
    }

    /**
     * @brief get_grid_index return the grid coordinate of a linear index
     *
     * get_grid_index return the grid coordinate of a linear index
     *   between 0 and nbNeigPerDim^dimension in   3x3 grid centered in 0.
     *
     * Each component od the grid index is between -a and a where a =
     * (nbNeigPerDim-1)/2, i.e. the number of neighbors
     *   in on direction (generally when we consider the first neighbors a = 1)
     *
     *
     * @tparam Dimension
     * @tparam IndexType
     * @tparam CoordinateType
     * @param idx the linear index
     * @param nbNeigPerDim  the number of points per line
     * @return and array of size dimension
     */
    template<std::size_t Dimension, typename IndexType = std::size_t, typename CoordinateType = std::int64_t>
    inline auto get_grid_3x3_index(CoordinateType& idx, const int nbNeigPerDim = 3) ->
      typename container::point<CoordinateType, Dimension>
    {
        container::point<CoordinateType, Dimension> coordinate{};
        auto tmp = idx;
        auto nbEltperDim = std::pow(nbNeigPerDim, Dimension);
        for(std::size_t d = 0; d < Dimension - 1; ++d)
        {
            nbEltperDim /= nbNeigPerDim;
            coordinate[d] = CoordinateType(tmp / nbEltperDim) - 1;
            tmp -= (coordinate[d] + 1) * nbEltperDim;
        }
        coordinate[Dimension - 1] = tmp - 1;

        return coordinate;
    }

    /**
     * @ingroup get_interaction_neighbors
     * @brief Compute the neighbors of the coordinate component
     *
     * @todo Problem with neighbour_separation /= 1 and the use of the array
     * structure !!!
     *
     * @tparam Dimension
     * @tparam IndexType
     * @tparam ArrayType
     * @tparam CoordinateType
     * @param[in] coordinate the grid coordinate of the morton index of the current box
     * @param[in] level The level to compute the neighbors
     * @param[in] period the periodicity in the different directions (array of bool)
     * @param[in] neighbour_separation  the number of neighbors in one direction (default 1 = the neighbors at distance 1 of me)
     * @return a tuple containing
     *   the sorted morton index of the neighbors
     *   the number of neighbors
     *   the position of the index
     */
    template<std::size_t Dimension, typename IndexType = std::size_t, typename ArrayType,
             typename CoordinateType = std::int64_t>
    inline auto get_neighbors(container::point<CoordinateType, Dimension> const& coordinate, std::size_t level,
                              ArrayType const& period, int neighbor_separation)
    {
        const std::size_t nbNeigPerDim = 3;
        using position_type = container::point<CoordinateType, Dimension>;
        static constexpr CoordinateType interactions = math::pow(nbNeigPerDim, Dimension);
        const CoordinateType limite1d = static_cast<IndexType>(1) << (static_cast<CoordinateType>(level));
        std::array<IndexType, interactions> indexes{};
        std::array<position_type, interactions> idx_pos{};
        int idx_neig = 0;
        // We test all cells around
        // As interactions is now 3^d --> th bound is interactions and not interactions+1
        for(CoordinateType idx = 0; idx < interactions; ++idx)
        {
            const auto idx_grid = get_grid_3x3_index<Dimension>(idx, nbNeigPerDim);
            bool check = (idx_grid[0] == 0);
            for(std::size_t d = 1; d < Dimension; ++d)
            {
                check = check && (idx_grid[d] == 0);
            }
            if(check)
            {
                continue;
            }
            auto coord = coordinate + idx_grid;
            check = true;
            // A mettre dans une fonction
            for(std::size_t d = 0; d < Dimension; ++d)
            {
                if(period[d])
                {
                    if(coord[d] < 0)
                    {
                        coord[d] += limite1d;
                    }
                    else if(coord[d] > limite1d - 1)
                    {
                        coord[d] -= limite1d;
                    }
                }
                else
                {
                    check = check && math::between(coord[d], CoordinateType(0), limite1d);
                }
            }
            if(!check)
            {
                continue;
            }
            indexes[idx_neig] = get_morton_index(coord);
            idx_pos[idx_neig] = coord;
            ++idx_neig;
        }
        std::sort(std::begin(indexes), std::begin(indexes) + idx_neig);
        return std::make_tuple(indexes, idx_neig, idx_pos);
    }

    ///////////////////////////

    /**
     * @brief Get the neighbors new object
     *
     * @tparam Dimension
     * @tparam IndexType
     * @tparam ArrayType
     * @tparam CoordinateType
     * @param coordinate
     * @param level
     * @param period
     * @param true_pos
     * @param neighbour_separation
     * @return auto
     */
    template<std::size_t Dimension, typename IndexType = std::size_t, typename ArrayType,
             typename CoordinateType = std::int64_t>
    inline auto get_neighbors_new(container::point<CoordinateType, Dimension> const& coordinate, std::size_t level,
                                  ArrayType const& period, const bool true_pos, const int neighbour_separation)
    {
        const int nbNeigPerDim = 3 /*2* neighbour_separation + 1 */;
        using position_type = container::point<CoordinateType, Dimension>;
        // the right size od the array is  nbNeigPerDim = 2* neighbour_separation
        // + 1  and not 3 !!! -1 because we don't consider the current box static
        // const std::size_t interactions = math::pow(nbNeigPerDim, Dimension) -
        // 1;
        // static constexpr std::size_t interactions = math::pow(nbNeigPerDim, Dimension) - 1;
        static constexpr CoordinateType interactions = math::pow(nbNeigPerDim, Dimension);
        const CoordinateType limite1d = static_cast<IndexType>(1) << (static_cast<CoordinateType>(level));
        std::array<IndexType, interactions> indexes{};
        std::array<position_type, interactions> idx_pos{};
        int idx_neig = 0;

        // We test all cells around
        CoordinateType idx_skip = interactions / 2;

        for(CoordinateType idx = 0; idx < interactions /*+1*/; ++idx)
        {
            if(idx == idx_skip)
            {
                continue;
            }
            const auto idx_grid = get_grid_3x3_index<Dimension>(idx, nbNeigPerDim);

            auto coord = coordinate + idx_grid;
            auto coord_ori = coord;
            // checkLimit modify coord to obtain teh good morton index
            // auto  check = checkLimit<Dimension>(coord, period, limite1d );
            if(!check_limit<Dimension>(coord, period, limite1d))
            {
                continue;
            }
            indexes[idx_neig] = get_morton_index(coord);
            idx_pos[idx_neig] = coord_ori;
            ++idx_neig;
        }

        return std::make_tuple(indexes, idx_pos, idx_neig);
    }

    /**
     * @brief Get the index of a interaction neighbors (for M2L)
     *
     * @tparam Dimension
     * @tparam CoordinateType
     * @param p position in the interactions (from -3 to +3)^Dimension
     * @return the index (from 0 to 342)
     */
    template<std::size_t Dimension, typename CoordinateType = std::int64_t>
    inline auto neighbor_index(container::point<CoordinateType, Dimension> const& p) -> int
    {
        CoordinateType pos{p[0] + 3};
        for(int d = 1; d < Dimension; ++d)
        {
            pos = pos * 7 + p[d] + 3;
        }
        return pos;
    }

    /**
     * @brief
     *
     * @tparam Dimension
     * @tparam CoordinateType
     * @param p
     * @return int
     */
    template<std::size_t Dimension, typename CoordinateType = std::int64_t>
    inline auto neighbor_index(std::array<CoordinateType, Dimension> const& p) -> int
    {
        CoordinateType pos{p[0] + 3};
        for(std::size_t d = 1; d < Dimension; ++d)
        {
            pos = pos * 7 + p[d] + 3;
        }
        return pos;
    }

    /**
     * @ingroup get_m2l_list
     * @brief Compute the interaction list of coordinate box
     *
     * @warning Problem with neighbour_separation /! 1 and the use of the array
     * structure !!!
     *
     * @tparam Dimension
     * @tparam MortonIndex
     * @tparam ArrayType
     * @tparam CoordinateType
     * @param[in] coordinate; the grid coordinate of the morton index of the current box.
     * @param[in] level The level to compute the interaction list
     * @param[in] period the vector of periodicity
     * @param[in] neighbour_separation  the number of neighbors in one direction (default 1 = the neighbors at distance 1 of me).
     * @return a tuple containing
     *   the morton index of the cells in the interaction list
     *   the position in the d grid of size (1+3*neighbour_separation)^Dimension
     *   the number of neighbors
     */
    template<std::size_t Dimension, typename MortonIndex = std::size_t, typename ArrayType,
             typename CoordinateType = std::int64_t>
    inline auto get_m2l_list(container::point<CoordinateType, Dimension> const& coordinate, std::size_t level,
                             ArrayType const& period, const int neighbour_separation)
    {
        // neighbour_separation<< ")\n";
        //        const int nbNeigPerDim = 6 /* 2*(2* neighbour_separation + 1 ) */;
        constexpr int nb_sons = math::pow(2, Dimension);
        using position_type = container::point<CoordinateType, Dimension>;
        // the right size of the array is  nbNeigPerDim = 2* neighbour_separation
        // + 1  and not 3 !!! -1 because we don't consider the current box static
        // const std::size_t interactions = math::pow(nbNeigPerDim, Dimension) -
        // math::pow(2*neighbour_separation + 1, Dimension) ;
        static constexpr std::size_t interactions = math::pow(6, Dimension) - math::pow(3, Dimension);
        //        const auto nbNeigPerDim_level = 2 * neighbour_separation + 1;
        bool is_periodic = false;
        for(std::size_t d = 0; d < Dimension; ++d)
        {
            is_periodic = is_periodic || period[d];
        }
        // number of cells in one dimension
        //
        std::array<MortonIndex, interactions> indexes{};
        std::array<CoordinateType, interactions> indexes_in_array{};
        // Compute the parent cell at level -1
        position_type parent_cell_coordinate{};
        meta::for_each(parent_cell_coordinate, coordinate, [](auto c) { return c >> 1; });
        //
        auto neig_parent = get_neighbors_new(parent_cell_coordinate, level - 1, period, true, neighbour_separation);
        // We test all cells around

        CoordinateType number_of_neighbors_parent = std::get<2>(neig_parent);
        auto morton_parent_index = std::get<0>(neig_parent);
        auto morton_parent_pos = std::get<1>(neig_parent);
        //  Loop on the neighbors of the parents to construct the sons withoy the neighbors of the target cell.
        int idx_m2L_list = 0;

        for(CoordinateType idx_p = 0; idx_p < number_of_neighbors_parent; ++idx_p)
        {
            //  Build the sons of the parent cell
            const MortonIndex morton_son = morton_parent_index[idx_p] << Dimension;
            position_type first_son{};
            meta::for_each(first_son, morton_parent_pos[idx_p], [](auto c) { return c << 1; });
            // Loop on the child
            for(MortonIndex idxCousin = 0; idxCousin < nb_sons; ++idxCousin)
            {
                auto dd = Dimension - 1;
                bool check = false;
                std::array<CoordinateType, Dimension> diff{};
                // Check to remove first neighbors
                for(std::size_t d = 0; d < Dimension; ++d)
                {
                    diff[d] = (first_son[d] | (CoordinateType(idxCousin >> dd) & 1)) - coordinate[d];
                    check = check || (std::abs(diff[d]) > neighbour_separation);
                    --dd;
                }
                if(check)
                {
                    indexes.at(idx_m2L_list) = morton_son + idxCousin;
                    // CoordinateType pos{diff[0] + 3};
                    // for(int d = 1; d < Dimension; ++d)
                    // {
                    //     pos = pos * 7 + diff[d] + 3;
                    // }
                    // indexes_in_array[idx_m2L_list] = pos;
                    indexes_in_array[idx_m2L_list] = neighbor_index(diff);
                    ++idx_m2L_list;
                }
            }
        }
        // We need to sort the indexes anf the permutation to reorder the position
        // mandatory
        {
            std::array<std::pair<MortonIndex, int>, interactions> perm;
            std::array<int, interactions> tmp{};

            for(int i{0}; i < idx_m2L_list; ++i)
            {
                perm[i].first = indexes[i];
                perm[i].second = i;
                tmp[i] = indexes_in_array[i];
            }
            std::sort(std::begin(perm), std::begin(perm) + idx_m2L_list,
                      [](const auto& x, const auto& y) { return x.first < y.first; });
            // Build the two sorted arrays
            for(int i{0}; i < idx_m2L_list; ++i)
            {
                indexes[i] = perm[i].first;
                indexes_in_array[i] = tmp[perm[i].second];
            }
        }
        return std::make_tuple(indexes, indexes_in_array, idx_m2L_list);
    }

    /**
     * @brief Get the interaction neighbors object
     *
     * @defgroup get_interaction_neighbors get_interaction_neighbors
     *
     * @ingroup get_interaction_neighbors
     *
     * @todo use metaprogrammation.
     *
     * @tparam Dimension
     * @tparam MortonIndex
     * @tparam CoordinateType
     * @tparam Array
     * @param t
     * @param coordinate
     * @param level
     * @param period
     * @param neighbour_separtion
     * @return auto
     */
    template<std::size_t Dimension, typename MortonIndex = std::size_t, typename CoordinateType = std::int64_t,
             typename Array>
    inline auto get_interaction_neighbors(operators::impl::tag_m2l t,
                                          container::point<CoordinateType, Dimension> const& coordinate,
                                          std::size_t level, Array const& period, int neighbour_separtion)
    {
        return get_m2l_list(coordinate, level, period, neighbour_separtion);
    }

    /**
     * @brief Get the opposite inter index object
     *
     * @tparam Dimension
     * @tparam IndexType
     * @param index
     * @return IndexType
     */
    template<std::size_t Dimension, typename IndexType>
    inline auto get_opposite_inter_index(IndexType index) -> IndexType
    {
        static constexpr std::size_t i = math::pow(7, Dimension);
        return static_cast<IndexType>(i) - index - IndexType(1);
    }

    /**
     * @brief Get the opposite p2p inter index object
     *
     * @tparam Dimension
     * @tparam IndexType
     * @param index
     * @return IndexType
     */
    template<std::size_t Dimension, typename IndexType>
    inline auto get_opposite_p2p_inter_index(IndexType index) -> IndexType
    {
        static constexpr std::size_t i = math::pow(3, Dimension);
        return static_cast<IndexType>(i) - index - IndexType(1);
    }

    // /// @ingroup get_interaction_neighbors
    // /// @brief
    // ///
    // /// @tparam Dimension
    // /// @tparam IndexType
    // /// @tparam CoordinateType
    // /// @param t
    // /// @param coordinate
    // /// @param tree_height
    // /// @param neighbour_separtion
    // ///
    // /// @return
    // // TODO fused the 2 functions !
    // template<std::size_t Dimension, typename IndexType = std::size_t, typename CoordinateType = std::int64_t,
    // typename Array> inline auto get_interaction_neighbors(operators::impl::tag_p2p t,
    //                                       container::point<CoordinateType, Dimension> const& coordinate,
    //                                       std::size_t leaf_level,  Array const & period, int neighbour_separtion = 1)
    // {
    //     return get_neighbors(coordinate, leaf_level, period, neighbour_separtion);
    // }

    /**
     * @brief Get the interaction neighbors of the component located by its coordinate.
     *
     * @tparam Dimension
     * @tparam IndexType
     * @tparam ArrayType
     * @tparam CoordinateType
     * @param[in] t  tag to specialize the function.
     * @param[in] coordinate the grid coordinate of the morton index of the current component.
     * @param[in] level The level to compute the neighbors.
     * @param[in] period the periodicity in the different directions (array of bool).
     * @param[in] neighbour_separation  the number of neighbors in one direction (default 1 = the neighbors at distance 1 of me).
     * @return a tuple containing the sorted morton index of the neighbors
     *   the number of neighbors
     */
    template<std::size_t Dimension, typename IndexType = std::size_t, typename ArrayType,
             typename CoordinateType = std::int64_t>
    inline auto get_interaction_neighbors(operators::impl::tag_p2p t,
                                          container::point<CoordinateType, Dimension> const& coordinate,
                                          std::size_t leaf_level, ArrayType const& period, int neighbour_separation)
    {
        return get_neighbors(coordinate, leaf_level, period, neighbour_separation);
    }

    /**
     * @brief Get the box object
     *
     * @tparam ValueType
     * @tparam Dimension
     * @tparam CoordinateType
     * @param corner
     * @param width
     * @param coordinate
     * @param tree_height
     * @param level
     * @return std::tuple<ValueType, container::point<ValueType, Dimension>>
     */
    template<typename ValueType, std::size_t Dimension, typename CoordinateType>
    inline auto get_box(container::point<ValueType, Dimension> const& corner, ValueType width,
                        container::point<CoordinateType, Dimension> const& coordinate, std::size_t tree_height,
                        std::size_t level) -> std::tuple<ValueType, container::point<ValueType, Dimension>>
    {
        auto width_at_current_level{width / ValueType(math::pow(2, level))};
        auto width_at_current_level_div2 = width_at_current_level / ValueType(2.);
        container::point<ValueType, Dimension> new_center{};
        meta::for_each(new_center, corner, coordinate,
                       [&width_at_current_level, &width_at_current_level_div2](auto c, auto coord)
                       { return c + ValueType(coord) * width_at_current_level + width_at_current_level_div2; });
        return std::make_tuple(width_at_current_level, new_center);
    }

    /**
     * @brief set the good morton index at fake level in periodic
     *
     * @tparam Dimension
     * @param index morton to correct
     * @return std::size_t
     */
    template<std::size_t Dimension>
    auto correctFakeMorton(std::size_t index) -> std::size_t
    {
        return (index > 3) ? index - ((index >> Dimension) << Dimension) : index;
    }

    /**
     * @brief Get the parent morton indices object
     *
     * @tparam MortonType
     * @param vector_of_mortons
     * @param dimension
     * @param offset
     */
    template<typename MortonType>
    inline auto get_parent_morton_indices(std::vector<MortonType>& vector_of_mortons, std::size_t dimension,
                                          std::size_t offset = 0) -> void
    {
        for(std::size_t i = 0; i < vector_of_mortons.size(); ++i)
        {
            vector_of_mortons.at(i) = vector_of_mortons.at(i) >> dimension;
        }

        /// we have to remove some elements at the begining if start != 0
        auto last = std::unique(vector_of_mortons.begin(), vector_of_mortons.end());
        if(offset > 0)
        {
            std::vector<MortonType> new_cell_index(std::distance(vector_of_mortons.begin() + offset, last));
            std::move(vector_of_mortons.begin() + offset, last, new_cell_index.begin());
            vector_of_mortons = std::move(new_cell_index);
        }
        else
        {
            vector_of_mortons.erase(last, vector_of_mortons.end());
        }
    }

    /**
     * @brief check that the group intersects the interval [start, end[.
     *
     * @tparam GroupType
     * @param start starting morton index
     * @param end ending morton index
     * @param g the group
     * @return return true if the intersection in not empty
     */
    template<typename GroupType>
    auto is_in_range(std::size_t start, std::size_t end, GroupType const& g) -> bool
    {
        auto const& csym = g.csymbolics();
        if(end <= csym.starting_index or csym.ending_index <= start)
        {
            return false;
        }

        return true;
    }

    /**
     * @brief Get the parent group range object
     *
     * @tparam GroupIteratorType
     * @param begin_range
     * @param end_range
     * @param begin_groups
     * @param end_groups
     * @return auto
     */
    template<typename GroupIteratorType>
    auto get_parent_group_range(std::size_t begin_range, std::size_t end_range, GroupIteratorType begin_groups,
                                GroupIteratorType end_groups)
    {
        if(begin_groups == end_groups)
        {
            return std::make_tuple(begin_groups, end_groups);
        }
        else
        {
            while(!is_in_range(begin_range, end_range, **begin_groups))
            {
                ++begin_groups;
                if(begin_groups == end_groups)
                {
                    return std::make_tuple(--begin_groups, end_groups);
                }
            }

            GroupIteratorType first{begin_groups};

            while(is_in_range(begin_range, end_range, **begin_groups))
            {
                ++begin_groups;
                if(begin_groups == end_groups)
                {
                    return std::make_tuple(first, end_groups);
                }
            }

            GroupIteratorType last{begin_groups};

            return std::make_tuple(first, last);
        }
    }

    /**
     * @brief Get the child group range object
     *
     * @tparam Dimension
     * @tparam GroupChildIteratorType
     * @tparam GroupParentType
     * @param begin_groups
     * @param end_groups
     * @param parent
     * @param verbose
     * @return auto
     */
    template<std::size_t Dimension, typename GroupChildIteratorType, typename GroupParentType>
    auto get_child_group_range(GroupChildIteratorType begin_groups, GroupChildIteratorType end_groups,
                               GroupParentType const& parent, bool verbose = false)
    {
        static constexpr std::size_t dimension = Dimension;

        if(begin_groups == end_groups)
        {
            return std::make_tuple(begin_groups, end_groups);
        }
        else
        {
            // (*begin_groups)->csymbolics().ending_index - 1 to have the morton inside
            //  and we take +1 to have a range
            while(!is_in_range(((*begin_groups)->csymbolics().starting_index >> dimension),
                               (((*begin_groups)->csymbolics().ending_index - 1) >> dimension) + 1, parent))
            {
                ++begin_groups;
                if(begin_groups == end_groups)
                {
                    return std::make_tuple(--begin_groups, end_groups);
                }
            }

            GroupChildIteratorType first{begin_groups};

            while(is_in_range(((*begin_groups)->csymbolics().starting_index >> dimension),
                              (((*begin_groups)->csymbolics().ending_index - 1) >> dimension) + 1, parent))
            {
                ++begin_groups;
                if(begin_groups == end_groups)
                {
                    return std::make_tuple(first, end_groups);
                }
            }

            GroupChildIteratorType last{begin_groups};

            return std::make_tuple(first, last);
        }
    }

    /**
     * @brief Get the shift to apply on center2 when the simulation box is periodic
     *
     * The shift is used to move the center2 near center1 according to the periodicity
     *
     * @tparam PointType
     * @tparam PeriodicityVectorType
     * @param center1 the current box
     * @param center2  the box which could be deplaced due to the periodicity
     * @param pbc   array of periodic direction (true if periodic)
     * @param box_width the width of the simulation box
     * @return vector of shift
     */
    template<typename PointType, typename PeriodicityVectorType>
    inline auto get_shift(const PointType& center1, const PointType& center2, const PeriodicityVectorType pbc,
                          typename PointType::value_type const& box_width) -> const PointType
    {
        PointType shift(0.0);
        auto half_width{0.5 * box_width};
        for(int i = 0; i < PointType::dimension; ++i)
        {
            if(pbc[i])
            {
                shift[i] = box_width * std::round((center1[i] - center2[i]) / box_width);
            }
        }
        return shift;
    }
}   // namespace scalfmm::index

namespace scalfmm::utils
{

    /**
     * @brief
     *
     * @tparam CellType
     * @param cell
     * @param print_aux
     */
    template<typename CellType>
    auto print_cell(const CellType& cell, bool print_aux = false) -> void
    {
        auto m = cell.cmultipoles();

        auto nb_m = m.size();
        std::cout << "cell index: " << cell.index() << " level " << cell.csymbolics().level << "\n";
        for(std::size_t i{0}; i < nb_m; ++i)
        {
            auto ten = m.at(i);
            std::cout << "  multi(" << i << "): \n" << m.at(i) << std::endl;
        }
        auto loc = cell.clocals();
        auto nb_loc = loc.size();
        for(std::size_t i{0}; i < nb_loc; ++i)
        {
            auto ten = loc.at(i);
            std::cout << &(loc.at(i)) << std::endl;
            std::cout << "  loc(" << i << "): \n" << loc.at(i) << std::endl;
        }
    }

    /**
     * @brief  Display particles in leaf and compute the current Morton index of the particle
     *
     * @tparam LeafType
     * @tparam BoxType
     * @param leaf the current leaf
     * @param box  The box to compute the Morton index
     * @param level  The level to compute the Morton index
     */
    template<typename LeafType, typename BoxType>
    auto print_leaf(const LeafType& leaf, BoxType box, int level) -> void
    {
        auto morton = leaf.index();
        int i{0};
        for(auto const& pl: leaf)
        {
            const auto p = typename LeafType::const_proxy_type(pl);
            std::cout << i++ << " morton: " << morton << " part= " << p << " cmp_morton "
                      << scalfmm::index::get_morton_index(p.position(), box, level) << std::endl;
        }
    }

    /**
     * @brief Display particles in the leaf.
     *
     * @tparam LeafType
     * @param leaf the current leaf.
     */
    template<typename LeafType>
    auto print_leaf(const LeafType& leaf) -> void
    {
        auto morton = leaf.index();
        int i{0};
        for(auto const& p: leaf)
        {
            const auto pp = typename LeafType::const_proxy_type(p);
            std::cout << i++ << " morton: " << morton << " part= " << pp << std::endl;
        }
    }
}   // namespace scalfmm::utils

#endif   // SCALFMM_TREE_UTILS_HPP
