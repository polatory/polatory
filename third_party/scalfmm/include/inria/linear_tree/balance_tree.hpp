#ifndef _INRIA_BALANCE_TREE_
#define _INRIA_BALANCE_TREE_

#include "distributed_regions_to_linear_tree.hpp"
#include "gather_octant_weights.hpp"
#include "weight_traits.hpp"

#include "inria/algorithm/distributed/distribute.hpp"
#include "inria/algorithm/distributed/sort.hpp"

namespace inria
{
    namespace linear_tree
    {
        namespace details
        {
            /**
             * \brief Node info with added weigh information
             */
            template<class P>
            struct weighted_node_info : node::info<P::position_t::Dim>
            {
                /// Weight attribute type
                using weight_t = decltype(inria::meta::get_weight(std::declval<P>()));
                /// Node weight
                weight_t weight = 0;

                // Use parent constructor
                using node::info<P::position_t::Dim>::info;

                /**
                 * \brief Output stream operator
                 *
                 * \param os Output stream
                 * \param n  Weighted node info object to stream
                 */
                friend std::ostream& operator<<(std::ostream& os, const weighted_node_info& n)
                {
                    using node::level;
                    using node::morton_index;
                    return os << '(' << morton_index(n) << ", " << level(n) << ", " << n.weight << ')';
                }
            };

            /**
             * \brief Function object to access weight
             */
            struct weight_accessor
            {
                template<class T>
                auto operator()(const T& e) const noexcept(noexcept(inria::meta::get_weight(e)))
                  -> decltype(inria::meta::get_weight(e))
                {
                    return inria::meta::get_weight(e);
                }
            };

            /**
             * \brief Helper type aliases for create_balanced_linear_tree
             */
            namespace cblt
            {
                /// Alias to extract the node::info<Dim> type from a particle iterator
                template<class T>
                using node_info_from_it = node::info<std::iterator_traits<T>::value_type::position_t::Dim>;

                /// Alias to extract the node::info<Dim> type from a particle range
                template<class T>
                using node_info_from_range = node::info<T::value_type::position_t::Dim>;

            }   // namespace cblt

        }   // namespace details

        /**
         * [INFO] : Every leaf of the returned linear tree don't have necessarily
         *          the same level
         * \brief Create a distributed linear tree from a sorted particle list
         *
         * \warning The particle list must be sorted accross the processes.
         *
         * \param conf   MPI configuration
         * \param level_ Maximum tree depth
         * \param box    The space bounding box
         * \param sorted_particles Distributed list of particles sorted by morton index
         *
         * \note The is box expected to be a cube.
         *
         * \tparam Range Particle list, must define begin and end functions and a
         *               value_type type alias.
         * \tparam Box   Must define `Box::width(int axis)` and `Box::corner(int morton_index)`.
         *
         * \return A holding the local leaf information (inria::linear_tree::node::info)
         *         of the tree.
         */
        template<class Range, class Box>
        std::vector<details::cblt::node_info_from_range<Box>>
        create_balanced_linear_tree(inria::mpi_config conf, std::size_t level_, Box box, Range& sorted_particles)
        {
            using particle_t = typename Range::value_type;
            using node_info_t = details::weighted_node_info<particle_t>;

            // Compute minimal and maximal octant for local range.
            node_info_t min_oct{}, max_oct{};
            if(!sorted_particles.empty())
            {
                auto min_idx = get_morton_index(sorted_particles.front().position(), box, level_);
                min_oct = node_info_t{min_idx, level_};
                auto max_morton_idx = get_morton_index(sorted_particles.back().position(), box, level_);
                max_oct = node_info_t{max_morton_idx, level_};
            }

            // Complete the local region based on sorted particles
            std::vector<node_info_t> local_region;
            complete_region(min_oct, max_oct, local_region);

            // Keep only the coarsest octants
            std::size_t min_level =
              std::accumulate(begin(local_region), end(local_region), level_,
                              [](std::size_t l, const node_info_t& n) { return std::min(l, level(n)); });
            local_region.erase(std::remove_if(begin(local_region), end(local_region),
                                              [&](const node_info_t& n) { return level(n) > min_level; }),
                               end(local_region));

            // Complete distributed octree
            std::vector<node_info_t> local_tree;
            distributed_regions_to_linear_tree(conf, begin(local_region), end(local_region), local_tree);

            // Compute the weight of each octant to redistribute them among processes
            gather_octants_weight(conf, begin(local_tree), end(local_tree), begin(sorted_particles),
                                  end(sorted_particles), box);

            distribute(conf, local_tree, uniform_distribution<details::weight_accessor>{conf, local_tree});

            coarsen_region(local_tree);

            return {begin(local_tree), end(local_tree)};
        }

        /**
         * \brief Create a distributed linear tree from a particle list
         *
         * \warning A copy of the particle list is made then sorted.
         *
         * \param conf   MPI configuration
         * \param level_ Maximum tree depth
         * \param box    The space bounding box
         * \param first  Particle list beginning
         * \param last   Particle list end
         * \param comp   Comparison function object used to sort the particles
         *
         * \note The is box expected to be a cube.
         *
         * \tparam ParticleForwardIt Forward iterator
         * \tparam Box   Must define `Box::width(int axis)` and `Box::corner(int morton_index)`.
         *
         * \return A holding the local leaf information (inria::linear_tree::node::info)
         *         of the tree.
         */
        template<class ParticleForwardIt, class Comp, class Box>
        std::vector<details::cblt::node_info_from_it<ParticleForwardIt>>
        create_balanced_linear_tree(inria::mpi_config conf, std::size_t level_, Box box, ParticleForwardIt first,
                                    ParticleForwardIt last, Comp&& comp)
        {
            auto sorted_particles = inria::sort(conf, first, last, comp);
            return create_balanced_linear_tree(conf, level_, box, sorted_particles);
        }

        /**
         * send_get_max_morton_idx this function send the max morton index of the current proc
         * to the proc n+1.
         * The current proc recev the max morton index of the proc n-1 and return it
         * @author benjamin.dufoyer@inria.fr
         * @param  conf     MPI conf
         * @param  max_morton_idx  max morton_index of the current proc
         * @return [description]
         */
        std::size_t send_get_max_morton_idx(inria::mpi_config& conf, std::size_t& max_morton_idx)
        {
            // Setting parametter
            const int nb_proc = conf.comm.size();
            const int my_rank = conf.comm.rank();
            // compute the buffer size
            std::size_t buff_recev{max_morton_idx + 1};
            if(nb_proc != 1)
            {
                inria::mpi::request tab_mpi_status[1];
                // compute the buffer size
                int size_buff = (int)sizeof(std::size_t);

                // if i'm not the last proc
                if(my_rank != nb_proc - 1)
                {
                    // Sending my max
                    conf.comm.isend(&max_morton_idx, size_buff, MPI_CHAR, my_rank + 1, 1);
                }
                // if i'm not the first proc
                if(my_rank != 0)
                {
                    // Receiv the max of the left proc
                    tab_mpi_status[0] = conf.comm.irecv(&buff_recev, size_buff, MPI_CHAR, my_rank - 1, 1);
                }
                // Waiting for the result of the request, if my rank is 0
                // I don't need to wait
                if(my_rank != 0)
                    inria::mpi::request::waitall(1, tab_mpi_status);
            }
            return buff_recev;
        }

        /**
         * This function create a linear tree from a level, it computes every leaf
         * at the level put in parameter
         *
         * 1 ) The first step is to send my max morton index of leaf to the right (n+1)
         *  proc
         *
         * 2) Compute the number of morton index to alloc the vector
         *
         * 3) stock all morton index in vector
         *
         * @author benjamin.dufoyer@inria.fr
         * @param level_            to compute the leaf
         * @param box box           to compute morton index
         * @param sorted_particle   particles, they will be sorted BEFORE calling this
         *                          function
         * @return                  linear tree with leaf at the same level
         */
        template<class Range, class Box>
        //     std::vector<details::cblt::node_info_from_range<Range>>
        Range create_balanced_linear_tree_at_leaf(inria::mpi_config conf, std::size_t level_, Box& box,
                                                  Range& sorted_particles_index)
        {
            // define type
            using node_info_t = typename Range::value_type;
            //  using node_info_t = typename particle_t::value_type;   // details::weighted_node_info<particle_t>;
            // compute the max morton index to create the linear tree
            size_t max_morton_idx{};
            std::size_t last = sorted_particles_index.size() - 1;
            if(!sorted_particles_index.empty())
            {
                max_morton_idx = sorted_particles_index[last][0];
                // get_morton_index(sorted_particles.back().position(), box, level_);
            }
            // Send my max morton index
            size_t maxMortonBeforeMe{send_get_max_morton_idx(conf, max_morton_idx)};
            // Compute the number of morton index
            unsigned nb_leaf = 0;
            std::size_t last_morton_index = max_morton_idx;
            for(unsigned i = 0; i < sorted_particles_index.size(); i++)
            {
                size_t curr_idx_morton =
                  sorted_particles_index[i][0];   // get_morton_index(sorted_particles.at(i).position(),box,level_);
                if(curr_idx_morton != last_morton_index && curr_idx_morton != maxMortonBeforeMe)
                {
                    last_morton_index = curr_idx_morton;
                    ++nb_leaf;
                }
            }
            // initialise the linear tree
            std::vector<node_info_t> lin_tree{nb_leaf};
            // Compute the number of leaf
            // the idea is to get every morton_index at the level_
            // when we find a new morton_index, it's a new leaf so we put her in
            // the linear tree
            nb_leaf = 0;
            last_morton_index = -1;
            for(unsigned i = 0; i < sorted_particles_index.size(); ++i)
            {
                size_t curr_idx_morton =
                  sorted_particles_index[i][0];   // get_morton_index(sorted_particles.at(i).position(), box, level_);
                // Check if it's a new morton index
                if(curr_idx_morton != last_morton_index && curr_idx_morton != maxMortonBeforeMe)
                {   // creation of the leaf
                    node_info_t newleaf{curr_idx_morton, sorted_particles_index[i][1] /*level_*/};
                    // adding the leaf in the linear_tree
                    lin_tree.at(nb_leaf) = newleaf;
                    // increment variable
                    last_morton_index = curr_idx_morton;
                    ++nb_leaf;
                }
            }

            return {begin(lin_tree), end(lin_tree)};
        }

        namespace details
        {
            /**
             * \brief Distribution object using pivots to choose target process
             *
             * This class is to be used with the inria::distribute function. When called, it
             * returns the rank of the process that should hold the given element after the
             * sort.
             *
             * \tparam Comp   The comparison callable type.
             * \tparam Pivots The list of pivots type.
             */
            template<class Comp, class Pivots>
            struct interval_distribution
            {
                Comp comp;
                Pivots pivots;
                template<class U>
                std::uint64_t operator()(const U& e)
                {
                    auto p_it = std::lower_bound(std::begin(pivots), std::end(pivots), e, comp);
                    std::uint64_t p_idx = p_it - std::begin(pivots);
                    if(p_idx > pivots.size())
                    {
                        throw std::runtime_error("Element distributed to non existing "
                                                 "process " +
                                                 std::to_string(p_idx));
                    }
                    return p_idx;
                }
            };

            template<class Comp, class Pivots>
            interval_distribution<Comp, Pivots> make_interval_distribution(Comp&& comp, Pivots&& pivots)
            {
                return {std::forward<Comp>(comp), std::forward<Pivots>(pivots)};
            }

        }   // namespace details

        /**
         * \brief Redistribute particles using the linear tree structure as a criteria
         *
         * To redistribute the particles, a pivot morton index is found for each
         * process. The pivot is the morton index of the hypothetical last cell at the
         * maximum level in the tree for each process.
         *
         * Each particle's morton index is searched (using a binary search) for in the
         * pivot array. The greatest pivot equal or inferior to the particle morton
         * index is selected and the particle is sent to the corresponding process.
         *
         * \param conf        MPI configuration
         * \param linear_tree Local linear tree segment
         * \param particles   Particle list
         */
        template<class AssignableRange, class LinearTree,
                 class IsRange = inria::enable_if_t<inria::is_range<AssignableRange>::value>,
                 class IsAssignable = inria::enable_if_t<inria::is_assignable<
                   AssignableRange, typename AssignableRange::iterator, typename AssignableRange::iterator>::value>>
        void redistribute_particles(inria::mpi_config conf, const LinearTree& linear_tree, AssignableRange& particles)
        {
            using node::morton_index;
            using morton_index_t = typename std::remove_reference<decltype(morton_index(particles.front()))>::type;
            using particle_t = typename AssignableRange::value_type;

            // Theoretical tree max depth, assumes sizeof(morton_index_t) is the exact
            // storage size (i.e. morton_index_t does not store any other information)
            constexpr auto max_depth = 8 * sizeof(morton_index_t) / 3;
            // Setup MPI datatype for future sends
            inria::mpi::datatype_commit_guard MortonType{inria::mpi::get_datatype<morton_index_t>()};
            // Setup pivots for distribution, initial value is 0b111....111
            std::vector<morton_index_t> pivots(conf.comm.size(), ~morton_index_t{0});
            // Initialise current process pivot
            if(!linear_tree.empty())
            {
                auto& n = linear_tree.back();
                pivots[conf.comm.rank()] = morton_index(last_descendant(n, max_depth - n.level));
            }
            // Share pivots
            conf.comm.allgather(pivots.data(), 1, MortonType.datatype);

            // Correct pivots, if a process holds no elements, copy the previous process
            // pivot. Note: this breaks if process 0 does not hold elements
            for(std::size_t i = 1u; i < pivots.size(); ++i)
            {
                if(pivots[i] == ~morton_index_t{0})
                {
                    pivots[i] = pivots[i - 1];
                }
            }

            // Object for comparison between morton index and particle
            struct
            {
                bool operator()(const morton_index_t& m, const particle_t& p) { return m < morton_index(p); };
                bool operator()(const particle_t& p, const morton_index_t& m) { return morton_index(p) < m; }
            } comp;
            // Setup an run distribution algorithm
            auto dist = details::make_interval_distribution(comp, std::move(pivots));
            inria::distribute(conf, particles, dist);
        }

    }   // namespace linear_tree
}   // namespace inria

#endif /* _INRIA_BALANCE_TREE_ */
