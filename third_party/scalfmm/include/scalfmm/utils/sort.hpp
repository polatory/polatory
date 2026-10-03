// --------------------------------
// See LICENCE file at project root
// File : scalfmm/utils/sort.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_SORT_HPP
#define SCALFMM_UTILS_SORT_HPP

#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/tree/utils.hpp"

#include <xtensor/views/xview.hpp>

namespace scalfmm::utils
{
    using indexing_structure = std::tuple<std::size_t, std::size_t>;

    /**
         * @brief sort_container compute the permutation to obtain the sorted particles according to the morton index
         *
         *         Each element of the returned vector is a tuple of std::size_t. The first element is the Morton index
         *  of the particle while the second element is original index
         *
         * @tparam BoxType
         * @tparam ContainerType
         * @tparam Int
         * @param[in] box  the box containing all the particles inside the container.
         * @param[in] level the level  of the d-tree to compute the morton index
         * @param[in] particle_container the particles container
         * @param[in] data_already_sorted specify if we only construct the identity permutation
         * @return the permutation a vector of tuple containing the morton index and the original position of the
         * particle
         */
    template<class BoxType, class ContainerType, typename Int>   //, class array_type>
    inline auto get_morton_permutation(const BoxType& box, const Int& level, const ContainerType& particle_container,
                                       const bool data_already_sorted = false) -> std::vector<indexing_structure>
    {
        auto number_of_particles = particle_container.size();
        std::vector<indexing_structure> tuple_of_indexes(number_of_particles);

        for(std::size_t part = 0; part < number_of_particles; ++part)
        {
            std::get<1>(tuple_of_indexes[part]) = part;
            std::get<0>(tuple_of_indexes[part]) =
              scalfmm::index::get_morton_index(particle_container.at(part).position(), box, level);
        }
        if(!data_already_sorted)
        {
            std::sort(std::begin(tuple_of_indexes), std::end(tuple_of_indexes),
                      [](auto& a, auto& b) { return std::get<0>(a) < std::get<0>(b); });
        }
        return tuple_of_indexes;
    }

    /**
         * @brief sort a particles container
         *
         * Partial sort the particle_container according to the Morton index of the
         * particle. The morton index is built at level level. At the end, the
         * particle_container contains the ordered particles
         *
         * @tparam BoxType the type of the box
         * @tparam ContainerType the type of the particles container
         * @tparam Int the type of an integer
         * @param[in] box  the box containing all the particles inside the
         * container.
         * @param[in] level the level of the d-tree to compute the morton index
         * @param[inout] particle_container the particles container
         */
    template<class BoxType, class ParticleType, typename Int>
    inline auto sort_container(const BoxType& box, const Int& level,
                               scalfmm::container::particle_container<ParticleType>*& particle_container) -> void
    {
        using ContainerType = scalfmm::container::particle_container<ParticleType>;
        auto perm = get_morton_permutation(box, level, *particle_container);
        const auto number_of_particles = particle_container->size();
        ContainerType* ordered_container = new ContainerType(number_of_particles);
        for(std::size_t i = 0; i < number_of_particles; ++i)
        {
            ordered_container->insert_particle(i, particle_container->particle(std::get<1>(perm[i])));
        }
        delete particle_container;
        particle_container = ordered_container;
    }

    /**
         * @brief
         *
         * @tparam BoxType
         * @tparam ParticleType
         * @tparam Int
         * @param box
         * @param level
         * @param particle_container
         */
    template<class BoxType, class ParticleType, typename Int>
    inline auto sort_container(const BoxType& box, const Int& level,
                               scalfmm::container::particle_container<ParticleType>& particle_container) -> void
    {
        using ContainerType = scalfmm::container::particle_container<ParticleType>;
        auto perm = get_morton_permutation(box, level, particle_container);
        const auto number_of_particles = particle_container.size();
        ContainerType tmp_container(number_of_particles);
        // for(std::size_t i = 0; i < number_of_particles; ++i)
        // {
        //     particle_container.insert_particle(i, particle_container.particle(std::get<1>(perm[i])));
        // }
        std::copy(particle_container.begin(), particle_container.end(), tmp_container.begin());
        for(std::size_t i = 0; i < number_of_particles; ++i)
        {
            particle_container.insert_particle(i, tmp_container.particle(std::get<1>(perm[i])));
        }
    }

    /**
         * @brief
         *
         * @tparam BoxType
         * @tparam ContainerType
         * @tparam Int
         * @param box
         * @param level
         * @param particle_container
         */
    template<class BoxType, class ContainerType, typename Int>
    inline auto sort_container(const BoxType& box, const Int& level, ContainerType& particle_container) -> void
    {
        auto perm = get_morton_permutation(box, level, particle_container);
        const auto number_of_particles = particle_container.size();
        ContainerType tmp_container(number_of_particles);
        std::copy(particle_container.begin(), particle_container.end(), tmp_container.begin());
        for(std::size_t i = 0; i < number_of_particles; ++i)
        {
            particle_container[i] = tmp_container[std::get<1>(perm[i])];
        }
    }
    /// \brief sort a vector particles
    ///
    /// Sort the vector of particles according to the Morton index of the
    /// particle. The morton index is built at the maximum level depending of the dimension.
    /// This level is int( sizeof(morton_type) / dimension).
    ///  The number of components (input + outputs values) associated to a particle is
    ///    array.size()/nbParticles - dimension. The type of all components is the same (float or double).
    ///
    /// \tparam BOX_T the type of the box
    /// \tparam MatrixPos The type of Matrix containig the particles
    /// \tparam VectorInt the type of the vector of permutation
    /// \tparam VectorMorton  the type of the element of Morton vector
    ///
    /// \param[in] box  the box containing all the particles inside the
    /// container.
    ///
    /// \param[in] nbParticles the number of particles stored in array.
    ///
    /// \param[in] pos the 2d-array of particles position (size nb_particles<dimension)
    /// \param[inout] permutation the vector of permutation to apply on pos in order to have a sorted array
    /// \param[out] morton the morton indexes of the particles(no sorted)
    /// \param[in] level The level used to compute the Morton index of a particle
    ///
    ///
    template<typename BoxType, typename MatrixPos, typename VectorInt, typename VectorMorton>
    void morton_sort(const BoxType& box, MatrixPos const& pos, VectorInt& permutation, VectorMorton& morton, int level)
    {
        using point_type = typename BoxType::position_type;
        constexpr std::size_t dimension = point_type::dimension;
        using morton_type = typename VectorMorton::value_type;
        //
        struct PARTICLE_AT
        {
            //
            morton_type m{};   //< The morton index
            int index{};       //< The old position
            const morton_type& getMortonIndex() { return m; }
        };

        // meta::td<VectorInt> u;
        // meta::td<VectorMorton> b;

        const int nbParticles = permutation.size();
        // const std::size_t max_level = sizeof(typename PARTICLE_AT::morton_type) * 8 / dimension - 1;
        std::vector<PARTICLE_AT> vect(nbParticles);
        point_type G;
        for(int i = 0; i < nbParticles; ++i)
        {
            auto xyz = xt::row(pos, i);
            for(int j = 0; j < dimension; ++j)
            {
                G[j] = xyz[j];
            }
            vect[i].index = i;   // Pour le fortran
            vect[i].m = scalfmm::index::get_morton_index(G, box, level);
        }
        std::sort(vect.begin(), vect.end(),
                  [&](PARTICLE_AT& a, PARTICLE_AT& b) { return (a.getMortonIndex() < b.getMortonIndex()); });

        for(int i = 0; i < nbParticles; ++i)
        {
            permutation[i] = vect[i].index;
            morton[i] = vect[i].m;
        }
        // std::vector<int> permback(3 * permutation.size(), -1);
        // int idx{0};
        // for(int i = 0; i < permutation.size(); ++i, idx += 3)
        // {
        //     auto index = permutation[i] - 1;
        //     permback[3 * index] = idx;
        //     permback[3 * index + 1] = idx + 1;
        //     permback[3 * index + 2] = idx + 2;
        // }
        // std::cout << "pb = [ " << permback[0];

        // for(int i = 1; i < permback.size(); ++i)
        // {
        //     std::cout << ", " << permback[i];
        // }
        // std::cout << "]\n";

        // std::cout << "p = [ " << permutation[0];

        // for(int i = 1; i < permutation.size(); ++i)
        // {
        //     std::cout << ", " << permutation[i];
        // }
        // std::cout << "]\n";
    }
    /// \brief sort a vector particles
    ///
    /// Sort the vector of particles according to the Morton index of the
    /// particle. The morton index is built at the maximum level depending of the dimension.
    /// This level is int( sizeof(morton_type) / dimension).
    ///  The number of components (input + outputs values) associated to a particle is
    ///    array.size()/nbParticles - dimension. The type of all components is the same (float or double).
    ///
    /// \tparam BoxType the type of the box
    /// \tparam VectorType the type of the vector of particles
    /// \tparam Int the type of an integer
    ///
    /// \param[in] box  the box containing all the particles inside the
    /// container.
    ///
    /// \param[in] nbParticles the number of particles stored in array.
    ///
    /// \param[inout] array the array of data (particles = position+inputs_values+outputs_values)
    ///
    ///
    template<class BoxType, class VectorType>   //, class array_type>
    void sort_raw_array_with_morton_index(const BoxType& box, const std::size_t& nbParticles, VectorType& array)
    {
        const int nb_val = array.size() / nbParticles;

        using points_type = typename BoxType::position_type;
        using morton_type = std::size_t;
        using value_type = typename VectorType::value_type;
        value_type* tmp_array = new value_type[array.size()];
        std::copy(array.begin(), array.end(), tmp_array);
        constexpr static const std::size_t dimension = points_type::dimension;
        //
        const std::size_t max_level = sizeof(morton_type) * 8 / dimension - 1;
        using pair_type = std::pair<morton_type, int>;
        std::vector<pair_type> tosort(nbParticles);
#pragma omp parallel for shared(tosort, nbParticles, box, max_level, array)
        for(std::size_t i = 0; i < nbParticles; ++i)
        {
            points_type pos(&(array[i * nb_val]));
            tosort[i].first = scalfmm::index::get_morton_index(pos, box, max_level);
            tosort[i].second = i;
        }

        std::sort(tosort.begin(), tosort.end(), [&](pair_type& a, pair_type& b) { return (a.first < b.first); });

        //
        // We fill the sorted array
#pragma omp parallel for shared(tosort, array, tmp_array)
        for(std::size_t i = 0; i < nbParticles; ++i)
        {
            auto start = tmp_array + nb_val * tosort[i].second;
            std::copy(start, start + nb_val, &(array[i * nb_val]));
        }
    }

    /**
         * @brief Print the position of the particles in the tree and in those in vector nodes
         *
         * @tparam TreeType
         * @tparam Permutation_type
         * @tparam VectorType
         * @param tree the tree containing the particles.
         * @param perm permutation obtain after sorting the particles at the leaf level.
         * @param nodes a sorted vector of particles
         */
    template<typename TreeType, typename Permutation_type, typename VectorType>
    inline auto check_positions(TreeType& tree, Permutation_type const& perm, VectorType const& nodes) -> void
    {
        // Set the fmm forces un vector the force structure
        auto box = tree.box();
        auto leaf_level = tree.height() - 1;

        std::size_t idx = 0;
        scalfmm::component::for_each_leaf(
          std::begin(tree), std::end(tree),
          [&nodes, &perm, &idx, leaf_level, box](auto& leaf)
          {
              std::cout << " -------------- leaf " << leaf.index() << " -------------- " << std::endl;
              for(auto const p_tuple_ref: leaf)
              {
                  // We construct a particle type for classical acces
                  const auto part = typename TreeType::leaf_type::const_proxy_type(p_tuple_ref);
                  auto p = part.position();
                  std::cout << " tree pos " << p << " morton " << scalfmm::index::get_morton_index(p, box, leaf_level)
                            << " nodes " << idx << "  " << nodes[idx] << "  perm: " << std::get<1>(perm[idx])
                            << " morton " << std::get<0>(perm[idx]) << std::endl;
                  ++idx;
              }
          });
    }
}   // namespace scalfmm::utils
#endif
