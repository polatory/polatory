#ifndef SCALFMM_TREE_LET_HPP
#define SCALFMM_TREE_LET_HPP
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/parallel_manager/parallel_manager.hpp>

#include <scalfmm/tree/utils.hpp>
#include <scalfmm/utils/io_helpers.hpp>   // for io::print
#include <scalfmm/utils/math.hpp>

#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/lists/sequential.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/parallel/mpi/utils.hpp"
#include "scalfmm/parallel/utils.hpp"
#include "scalfmm/tree/for_each.hpp"
#ifdef SCALFMM_USE_MPI

#include <inria/algorithm/distributed/distribute.hpp>
#include <inria/algorithm/distributed/mpi.hpp>
#include <inria/algorithm/distributed/sort.hpp>
#include <inria/linear_tree/balance_tree.hpp>
#include <mpi.h>
#endif

namespace scalfmm::tree
{
    using morton_type = std::int64_t;   // typename Tree_type::

    template<typename MortonIdx>
    struct leaf_info_type
    {
        using morton_type = MortonIdx;
        MortonIdx morton{};
        std::size_t number_of_particles{};
        friend std::ostream& operator<<(std::ostream& os, const leaf_info_type& w)
        {
            os << "[" << w.morton << ", " << w.number_of_particles << "] ";
            return os;
        }
    };

    namespace let
    {

        template<typename Box, typename VectorLeafInfo, typename MortonDistribution>
        inline /*std::vector<morton_type>*/ VectorLeafInfo
        get_ghosts_p2p_interaction(cpp_tools::parallel_manager::parallel_manager& para, Box const& box,
                                   std::size_t const& level, int const& separation, VectorLeafInfo const& leaf_info,
                                   MortonDistribution const& leaves_distrib)
        {
            std::vector<morton_type> ghost_to_add;
            auto const& period = box.get_periodicity();
            const auto rank = para.get_process_id();
            auto const& my_distrib = leaves_distrib[rank];
            //
            for(auto const& info: leaf_info)
            {
                auto const& morton_index = info.morton;
                auto coordinate{index::get_coordinate_from_morton_index<Box::dimension>(morton_index)};
                auto interaction_neighbors = index::get_neighbors(coordinate, level, period, separation);
                auto& list = std::get<0>(interaction_neighbors);
                auto nb = std::get<1>(interaction_neighbors);
                int it{0};
                //io::print("rank(" + std::to_string(rank) + ") list idx(p2p)  : ", list);

                while(list[it] < my_distrib[0])
                {
                    // std::cout << "INSIDE left idx " << list[it] << "  " << std::boolalpha
                    //             << parallel::utils::is_inside_distrib(list[it], leaves_distrib) << std::endl;
                    if(parallel::utils::is_inside_distrib_left(list[it], rank, leaves_distrib))
                    {
                        ghost_to_add.push_back(list[it]);
                    }
                    ++it;
                }
                it = nb - 1;
                while(list[it] >= my_distrib[1])
                {
                    //      std::cout << "INSIDE right idx " << list[it] << "  " << std::boolalpha
                    //               << parallel::utils::is_inside_distrib(list[it], leaves_distrib) << std::endl;
                    if(parallel::utils::is_inside_distrib_right(list[it], rank, leaves_distrib))
                    {
                        ghost_to_add.push_back(list[it]);
                    }
                    --it;
                }
            }
            std::sort(ghost_to_add.begin(), ghost_to_add.end());
            auto last = std::unique(ghost_to_add.begin(), ghost_to_add.end());
            ghost_to_add.erase(last, ghost_to_add.end());
            VectorLeafInfo ghost_leaf_to_add(ghost_to_add.size());
            for(int i = 0; i < ghost_to_add.size(); ++i)
            {
                ghost_leaf_to_add[i] = {ghost_to_add[i], 0};
            }

            return ghost_leaf_to_add;
        }
        ///
        /// \brief  get theoretical m2l interaction list outside me
        ///
        /// We return the list of indexes of cells involved in P2P interaction that we do
        ///  not have locally.  The cells on other processors may not exist.
        ///
        /// \param[in]  para the parallel manager
        /// \param tree the tree used to compute the interaction
        /// \param local_morton_idx the local morton index of the cells
        /// \param cell_distrib the cells distribution on the processes
        /// \return the list of indexes on tother processes
        ///
        template<typename Box, typename VectorMortonIdx, typename MortonDistribution>
        inline VectorMortonIdx
        get_ghosts_m2l_interaction(cpp_tools::parallel_manager::parallel_manager& para, Box const& box,
                                   const std::size_t& level, int const& separation,
                                   VectorMortonIdx const& local_morton_vect, MortonDistribution const& cell_distrib)
        {
            VectorMortonIdx ghost_to_add;
            auto const& period = box.get_periodicity();
            const auto rank = para.get_process_id();
            auto const my_distrib = cell_distrib[rank];
            const bool last_proc = (rank == para.get_num_processes());
            //
            for(auto morton_index: local_morton_vect)
            {
                // for each index in the vector of cells in local_morton_vect we compute the m2l interactions
                auto coordinate{index::get_coordinate_from_morton_index<Box::dimension>(morton_index)};
                auto interaction_m2l_list = index::get_m2l_list(coordinate, level, period, separation);

                auto& list = std::get<0>(interaction_m2l_list);
                auto nb = std::get<2>(interaction_m2l_list);
                //
                // io::print("rank(" + std::to_string(rank) + ") list idx(m2l)  : ", list);
                // io::print("rank(" + std::to_string(rank) + ") my_distrib  : ", my_distrib);

                int it{0};
                // We check if the cells are in the distribution
                for(auto it = 0; it < nb; ++it)
                {
                    // if(list[it] > my_distrib[0])
                    // std::cout << list[it] << " " << std::boolalpha
                    //           << math::between(list[it], my_distrib[0], my_distrib[1]) << std::endl;

                    if(math::between(list[it], my_distrib[0], my_distrib[1]))
                    {
                        break;
                    }
                    bool check{false};
                    // for(int i = 0; i < rank; ++i)
                    for(int i = rank - 1; i >= 0; i--)
                    {
                        auto const& interval = cell_distrib[i];
                        // // if(rank == 2)
                        // {
                        //     std::cout << "parallel::utils::is_inside_distrib_left list[it]: " << interval[0] << " < "
                        //     << list[it]
                        //               << " < " << interval[1] << std::endl;
                        // }
                        check = math::between(list[it], interval[0], interval[1]);
                        if(check)
                        {
                            break;
                        }
                    }
                    // std::cout << "                 " << list[it] << "  " << std::boolalpha << check << std::endl;
                    if(check)   // parallel::utils::is_inside_distrib_left(list[it], rank, cell_distrib))
                    {
                        ghost_to_add.push_back(list[it]);
                    }
                }
                // while(list[it] < my_distrib[0])
                // {
                //     std::cout << it << " INSIDE left idx " << list[it] << "  " << std::boolalpha
                //               << parallel::utils::is_inside_distrib(list[it], cell_distrib) << std::endl;
                //     if(parallel::utils::is_inside_distrib_left(list[it], rank, cell_distrib))
                //     {
                //         ghost_to_add.push_back(list[it]);
                //     }
                //     ++it;
                //     if(it > nb)
                //     {
                //         break;
                //     }
                // }
                it = nb - 1;
                if(not last_proc)   // No ghost on the right on last process
                {
                    while(list[it] >= my_distrib[1])
                    {
                        if(parallel::utils::is_inside_distrib_right(list[it], rank, cell_distrib))
                        {
                            ghost_to_add.push_back(list[it]);
                        }
                        --it;
                    }
                }
                // if(rank == 2)
                // {
                //     io::print("rank(" + std::to_string(rank) + ") tmp ghost_to_add(m2l)  : ", ghost_to_add);
                // }
            }
            std::sort(ghost_to_add.begin(), ghost_to_add.end());
            auto last = std::unique(ghost_to_add.begin(), ghost_to_add.end());
            ghost_to_add.erase(last, ghost_to_add.end());
            // io::print("rank(" + std::to_string(rank) + ") cell_distrib: ", cell_distrib);
            // io::print("rank(" + std::to_string(rank) + ") ghost_to_add(m2l): ", ghost_to_add);

            return ghost_to_add;
        }

        template<typename VectorLeafInfoType>
        auto merge_split_structure(VectorLeafInfoType const& localLeaves, VectorLeafInfoType const& ghosts)
        {
            // compute the size of the merged vector
            using morton_type = typename VectorLeafInfoType::value_type::morton_type;
            std::size_t i{0}, j{0}, k{0};
            std::size_t size1{localLeaves.size()}, size2{ghosts.size()};

            std::vector<morton_type> morton(size1 + size2);
            std::vector<std::size_t> number_of_particles(size1 + size2);
            i = j = k = 0;
            while(i < size1 && j < size2)
            {
                if(localLeaves[i].morton < ghosts[j].morton)
                {
                    morton[k] = localLeaves[i].morton;
                    number_of_particles[k++] = localLeaves[i++].number_of_particles;
                }
                else if(localLeaves[i].morton > ghosts[j].morton)
                {
                    morton[k] = ghosts[j].morton;
                    number_of_particles[k++] = ghosts[j++].number_of_particles;
                }
                else
                {
                    morton[k] = localLeaves[i].morton;
                    number_of_particles[k++] = localLeaves[i++].number_of_particles;
                    j++;
                }
            }

            // Add the remaining elements of vector localLeaves (if any)
            while(i < size1)
            {
                morton[k] = localLeaves[i].morton;
                number_of_particles[k++] = localLeaves[i++].number_of_particles;
            }

            // Add the remaining elements of vector ghost_m2l_cells (if any)
            while(j < size2)
            {
                morton[k] = ghosts[j].morton;
                number_of_particles[k++] = ghosts[j++].number_of_particles;
            }

            return std::make_tuple(morton, number_of_particles);
        }
        /**
         * @brief Split the LeafInfo structure in two vectors (Morton, number_of_particles)
         *
         * @tparam VectorLeafInfoType
         * @param leaves
         * @return a tuple of two vectors the morton index and the number of particles in the leaves vector
         */
        template<typename VectorLeafInfoType>
        auto split_structure(VectorLeafInfoType const& leaves)
        {
            // compute the size of the merged vector
            using morton_type = typename VectorLeafInfoType::value_type::morton_type;
            std::size_t k{0};
            std::size_t size{leaves.size()};

            std::vector<morton_type> morton(size);
            std::vector<std::size_t> number_of_particles(size);
            for(auto& v: leaves)
            {
                morton[k] = v.morton;
                number_of_particles[k++] = v.number_of_particles;
            }
            return std::make_tuple(morton, number_of_particles);
        }
        template<typename VectorLeafInfoIteratorType>
        auto split_structure(const VectorLeafInfoIteratorType begin, const VectorLeafInfoIteratorType end)
        {
            // compute the size of the merged vector
            using VectorLeafInfoType = std::decay_t<decltype(*begin)>;
            using morton_type = typename VectorLeafInfoType::morton_type;
            std::size_t k{0};
            auto size{std::distance(begin, end)};
            std::vector<morton_type> morton(size);
            std::vector<std::size_t> number_of_particles(size);
            for(auto it = begin; it != end; ++it)
            {
                morton[k] = (*it).morton;
                number_of_particles[k++] = (*it).number_of_particles;
            }
            return std::make_tuple(morton, number_of_particles);
        }

        /**
         * @brief merge the three vector of blocs in one vector
         *
         * @tparam VectorBlockType
         * @param bloc1 first vector
         * @param bloc2 second vector
         * @param bloc3 third vector
         * @return the vector ot the three blocs
         */
        template<typename VectorBlockType>
        VectorBlockType merge_blocs(VectorBlockType const& bloc1, VectorBlockType const& bloc2,
                                    VectorBlockType const& bloc3)
        {
            // Merge the three block structure
            auto size = bloc1.size() + bloc2.size() + bloc3.size();

            VectorBlockType all_blocks(size);
            int k{0};
            for(int i = 0; i < bloc1.size(); ++i)
            {
                all_blocks[k++] = bloc1[i];
            }
            for(int i = 0; i < bloc2.size(); ++i)
            {
                all_blocks[k++] = bloc2[i];
            }
            for(int i = 0; i < bloc3.size(); ++i)
            {
                all_blocks[k++] = bloc3[i];
            }
            return all_blocks;
        }
        /**
         * @brief Construct the M2M ghost for the current level
         *
         *  The routine check if there is ghosts during the M2M operation.
         *  If yes, we exchange the ghost indexes
         * @tparam Box
         * @tparam VectorMortonIdx
         * @tparam MortonDistribution
         * @param para  the parallel manager
         * @param box  the simulation box
         * @param level the current level
         * @param local_morton_vect
         * @param cells_distrib teh distribution of cells
         * @param top if top is true nothing is down
         * @return VectorMortonIdx
         */
        template<typename Box, typename VectorMortonIdx, typename MortonDistribution>
        [[nodiscard]] auto build_ghost_m2m_let_at_level(cpp_tools::parallel_manager::parallel_manager& para, Box& box,
                                                        const int& level, VectorMortonIdx const& local_morton_vect,
                                                        MortonDistribution const& cells_distrib,
                                                        bool top = false) -> VectorMortonIdx
        {
            using morton_type = typename VectorMortonIdx::value_type;
            static constexpr int nb_children = math::pow(2, Box::dimension);
            VectorMortonIdx ghosts;
            std::clog << " begin build_ghost_m2m_let_at_level " << level << std::endl;
            io::print(std::clog, "local_morton_vect: ", local_morton_vect);
            std::clog << std::endl;

            if(top)
                return ghosts;
            const auto rank = para.get_process_id();
            const auto proc = para.get_num_processes();
            auto comm = para.get_communicator();
            // parallel::utils::print_distrib("level_dist[" + std::to_string(level) + "]: ", rank, cells_distrib);

            cpp_tools::parallel_manager::mpi::request mpi_status_left, mpi_status_right;

            auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<morton_type>();
            int tag = 100;
            // first element is the number of children to send
            //   the children belong on two processes
            std::array<morton_type, nb_children> send{}, recv{};
            //
            //             parallel::utils::print_distrib("level_dist[leaf_level]): ", rank, cells_distrib);
            bool comm_left{false}, comm_right{false};
            // Check on left
            std::clog << "check left\n ";

            if(rank > 0)
            {
                auto first_index = local_morton_vect[0];
                auto parent_first = first_index >> Box::dimension;
                auto last_parent_previous_proc = (cells_distrib[rank - 1][1] - 1) >> Box::dimension;
                // std::clog << "index : " << first_index << "   Parent ! " << parent_first << " "
                //           << last_parent_previous_proc << std::endl;

                if(parent_first == last_parent_previous_proc)
                {
                    comm_left = true;
                    //                    std::cout << "Need to exchange between " << rank << "  and " << rank - 1 << std::endl;
                    int idx{1};
                    send[idx] = local_morton_vect[0];
                    for(int i = 1; i < std::min(nb_children, int(local_morton_vect.size())); ++i)
                    {
                        auto parent_index = local_morton_vect[i] >> Box::dimension;
                        //			std::cout << "index : " << local_morton_vect[i] << "   Parent ! " << parent_first << " " << last_parent_previous_proc << std::endl;
                        if(parent_index == last_parent_previous_proc)
                        {
                            ++idx;
                            send[idx] = local_morton_vect[i];
                        }
                        else
                        {
                            break;
                        }
                    }
                    //	    std::cout << "nbindex to send " <<idx  <<std::endl;
                    send[0] = idx;
                    //                     io::print(" index to send ", send);
                    comm.isend(send.data(), nb_children, mpi_type, rank - 1, tag);
                }
            }
            // std::clog << "check right\n ";
            auto last_index = local_morton_vect[local_morton_vect.size() - 1];
            auto parent_last = last_index >> Box::dimension;
            // std::clog << "last_index " << last_index << " parent_last  " << parent_last << std::endl;
            ghosts.resize(0);
            if(rank < proc - 1)
            {
                // check on left
                auto first_parent_next_proc = cells_distrib[rank + 1][0] >> Box::dimension;
                // std::clog << "Parent ! " << parent_last << " " << first_parent_next_proc << std::endl;
                if(parent_last == first_parent_next_proc)
                {
                    comm_right = true;
                    //                    std::cout << "Need to exchange between " << rank << "  and " << rank + 1 << std::endl;
                    /*mpi_status_right =*/comm.recv(recv.data(), nb_children, mpi_type, rank + 1, tag);
                    // cpp_tools::parallel_manager::mpi::request::waitall(1, &mpi_status_right);
                    //		                 io::print("recv ",recv );
                    ghosts.resize(recv[0]);
                    for(int i = 0; i < ghosts.size(); ++i)
                    {
                        ghosts[i] = recv[i + 1];
                    }
                }
            }
            // io::print(std::clog, "m2m ghosts ", ghosts);
            // std::clog << " end build_ghost_m2m_let_at_level" << std::endl;
            return ghosts;
        }
        ///
        /// \brief construct the local essential tree (LET) at the level.
        ///
        ///  We start from a given Morton index distribution and we compute all
        ///  interactions needed
        ///   in the algorithm steps.
        ///  At the leaf level it corresponds to the interactions coming from the
        ///  direct pass (P2P operators)
        ///     and in the transfer pass (M2L operator). For the other levels we
        ///     consider only the M2L interactions.
        /// The leaves_distrib and the cells_distrib might be different
        ///  At the end the let has also all the interaction list computed
        ///
        /// \param[inout]  tree the tree to compute the let.
        /// \param[in]  local_morton_idx the morton index of the particles in the
        /// processors.
        ///
        ///
        ///  \param[in]  cells_distrib the morton index distribution for
        /// the cells at the leaf level.
        ///
        ///  \param[in]  level the level to construct the let
        ///
        template<typename Box, typename VectorMortonIdx, typename MortonDistribution>
        [[nodiscard]] auto build_let_at_level(cpp_tools::parallel_manager::parallel_manager& para, Box& box,
                                              const int& level, const VectorMortonIdx& local_morton_vect,
                                              const MortonDistribution& cells_distrib,
                                              const int& separation) -> VectorMortonIdx
        {
            const auto my_rank = para.get_process_id();
            // std::cout << cpp_tools::colors::red << " --> Begin let::build_let_at_level() at level = " << level
            //           << "dist: " << cells_distrib[my_rank] << cpp_tools::colors::reset << std::endl;
            // io::print("rank(" + std::to_string(my_rank) + ") local_morton_vect  : ", local_morton_vect);

            //  we compute the cells needed in the M2L operator

            auto needed_idx =
              std::move(get_ghosts_m2l_interaction(para, box, level, separation, local_morton_vect, cells_distrib));

            // io::print("rank(" + std::to_string(my_rank) + ") get_ghosts_m2l_interactions  : ", needed_idx);

            std::cout << std::flush;
            /// Look if the morton index really exists in the distributed tree
            parallel::utils::check_if_morton_index_exist(para, needed_idx, cells_distrib, local_morton_vect);
            ///
            // io::print("rank(" + std::to_string(my_rank) + ") check_if_morton_index_exist(m2l)  : ", needed_idx);
            //

            // std::cout << cpp_tools::colors::red
            //           << "rank(" + std::to_string(my_rank) + ")-- > End let::build_let_at_level() "
            //           << cpp_tools::colors::reset << std::endl;
            return needed_idx;
        }
        // template<typename OctreeTree, typename VectorMortonIdx, typename MortonDistribution>
        // void build_let_at_level(cpp_tools::parallel_manager::parallel_manager& para, OctreeTree& tree,
        //                         const VectorMortonIdx& local_morton_idx, const MortonDistribution& cells_distrib,
        //                         const int& level)
        // {
        //     std::cout << cpp_tools::colors::green << " --> Begin let::build_let_at_level() at level = " << level
        //               << cpp_tools::colors::reset << std::endl;

        //     // auto my_rank = para.get_process_id();
        //     // // stock in the variable if we are at the leaf level
        //     // bool leaf_level = (tree.leaf_level() == level);

        //     // //  we compute the cells needed in the M2L operators
        //     // auto needed_idx =
        //     //   std::move(distrib::get_m2l_interaction_at_level(para, tree, local_morton_idx, cells_distrib,
        //     //   level));
        //     // //           io::print("rank(" + std::to_string(my_rank) + ") needed_idx(m2l)  : ", needed_idx);
        //     // // std::cout << std::flush;
        //     // /// Look if the morton index really exists in the distributed tree
        //     // distrib::check_if_morton_index_exist(para, needed_idx, cells_distrib, local_morton_idx);

        //     // //            std::cout << std::flush;
        //     // ///
        //     // tree.insert_cells_at_level(level, needed_idx);
        //     std::cout << cpp_tools::colors::green << " --> End let::build_let_at_level() at level = " << level
        //               << cpp_tools::colors::reset << std::endl;
        // }
        /**
         * @brief
         *
         * @tparam Box
         * @tparam VectorMortonIdx
         * @tparam MortonDistribution
         * @param para
         * @param box
         * @param level
         * @param leaf_info
         * @param leaves_distrib
         * @param separation
         */
        template<typename Box, typename VectorLeafInfo, typename MortonDistribution>
        [[nodiscard]] auto build_let_leaves(cpp_tools::parallel_manager::parallel_manager& para, Box const& box,
                                            const std::size_t& level,
                                            const VectorLeafInfo& leaf_info /*local_morton_vect*/,
                                            MortonDistribution const& leaves_distrib, const int& separation)

          -> VectorLeafInfo
        {
            auto my_rank = para.get_process_id();
            //             std::cout << cpp_tools::colors::green
            //                       << "rank(" + std::to_string(my_rank) + ") --> Begin let::build_let_leaves() "
            //                       << cpp_tools::colors::reset << std::endl;
            //             io::print("rank(" + std::to_string(my_rank) + ") leaf_info  : ", leaf_info);

            //  we compute the leaves involved in the P2P operators
            auto leaf_info_to_add =
              std::move(get_ghosts_p2p_interaction(para, box, level, separation, leaf_info, leaves_distrib));
            //io::print("rank(" + std::to_string(my_rank) + ") leaf_info_to_add(p2p)  : ", leaf_info_to_add);

            std::vector<morton_type> needed_idx(leaf_info_to_add.size());
            for(int i = 0; i < leaf_info_to_add.size(); ++i)
            {
                needed_idx[i] = leaf_info_to_add[i].morton;
            }
            /// Look if the morton index really exists in the distributed tree
            /// needed_idx input  contains the Morton index of leaf
            ///            output   contains the number of particles in the leaf

            //           io::print("rank(" + std::to_string(my_rank) + ") 1 leaf_info_to_add(p2p)  : ", leaf_info_to_add);
            parallel::utils::check_if_leaf_morton_index_exist(para, needed_idx, leaves_distrib, leaf_info);
            //            io::print("rank(" + std::to_string(my_rank) + ") check needed_idx.size  : ", needed_idx);
            int idx{0};

            for(int i = 0; i < needed_idx.size(); ++i)
            {
                if(needed_idx[i] > 0)
                {
                    leaf_info_to_add[idx].morton = leaf_info_to_add[i].morton;
                    leaf_info_to_add[idx].number_of_particles = needed_idx[i];
                    ++idx;
                }
            }
            if(idx != needed_idx.size())
            {
                auto last = leaf_info_to_add.cbegin() + idx;
                leaf_info_to_add.erase(last, leaf_info_to_add.end());
            }
            ///
            //             io::print("rank(" + std::to_string(my_rank) + ") final leaf_info_to_add(p2p)  : ", leaf_info_to_add);
            //             std::cout << cpp_tools::colors::green
            //                       << "rank(" + std::to_string(my_rank) + ")-- > End let::build_let_leaves() "
            //                       << cpp_tools::colors::reset << std::endl;
            return leaf_info_to_add;
        }

        ///
        /// \brief buildLetTree  Build the let of the tree and the leaves and cells distributions
        ///
        /// The algorithm has 5 steps:
        ///   1) We sort the particles according to their Morton Index (leaf level)
        ///   2) Build the leaf morton vector of my local particles and construct either
        ///      the leaves distribution or the cell distribution according to parameter
        ///       use_leaf_distribution or use_particle_distribution
        ///   3) Fit the particles inside the use_leaf_distribution
        ///   4) Construct the  tree according to my particles and build the leaf
        ///       morton vector of my local particles
        ///   5) Constructing the let level by level
        ///
        /// \param[in]    manager   the parallel manager
        /// \param[in] number_of_particles  total number of particles in the simulation
        /// \param[in]  particle_container   vector of particles on my node. On output the
        ///                 array is sorted and correspond to teh distribution built
        /// \param[in]     box  size of the simulation box
        /// \param[in] leaf_level   level of the leaf in the tree
        /// \param[in] level_shared the level at which cells are duplicated on processors. If the level is negative,
        /// nothing is duplicated.
        /// \param[in] groupSizeLeaves  blocking parameter for the leaves (particles)
        /// \param[in] groupSizeCells    blocking parameter for the cells
        /// @param[in]    order order of the approximation to build the tree
        /// @param[in]    use_leaf_distribution to say if you consider the leaf distribution
        /// @param[in]    use_particle_distribution to say if you consider the particle distribution
        /// @return  localGroupTree  the LET of the octree

        /// processors
        template<typename Tree_type, typename Vector_type, typename Box_type>
        Tree_type
        buildLetTree(cpp_tools::parallel_manager::parallel_manager& manager, const std::size_t& number_of_particles,
                     Vector_type& particle_container, const Box_type& box, const int& leaf_level,
                     const int& level_shared, const int groupSizeLeaves, const int groupSizeCells, const int order,
                     const int separation, const bool use_leaf_distribution, const bool use_particle_distribution)
        {
            // std::cout << cpp_tools::colors::green << " --> Begin let::group_let() " << cpp_tools::colors::reset
            //           << std::endl;
            //
            static constexpr std::size_t dimension = Vector_type::value_type::dimension;
            const auto rank = manager.get_process_id();
            ////////////////////////////////////////////////////////////////////////////
            ///   Sort the particles at the leaf level according to their Morton index
#ifdef SCALFMM_USE_MPI
            inria::mpi_config conf_tmp(manager.get_communicator().raw_comm);

            inria::sort(conf_tmp.comm, particle_container,
                        [&box, &leaf_level](const auto& p1, const auto& p2)
                        {
                            auto m1 = scalfmm::index::get_morton_index(p1.position(), box, leaf_level);
                            auto m2 = scalfmm::index::get_morton_index(p2.position(), box, leaf_level);
                            return m1 < m2;
                        });
#else
            std::sort(particle_container.begin(), particle_container.end(),
                      [&box, &leaf_level](const auto& p1, const auto& p2)
                      {
                          auto m1 = scalfmm::index::get_morton_index(p1.position(), box, leaf_level);
                          auto m2 = scalfmm::index::get_morton_index(p2.position(), box, leaf_level);
                          return m1 < m2;
                      });
#endif
            // Build the morton index of the particles in order to find the
            //  existing leaves
            const std::size_t localNumberOfParticles = particle_container.size();
            std::vector<morton_type> particleMortonIndex(localNumberOfParticles);
            // As the particles are sorted the leafMortonIdx is sorted too
            // #pragma omp parallel for shared(localNumberOfParticles, box, leaf_level)
            for(std::size_t part = 0; part < localNumberOfParticles; ++part)
            {
                particleMortonIndex[part] =
                  scalfmm::index::get_morton_index(particle_container[part].position(), box, leaf_level);
                // std::cout << part << " m  " << particleMortonIndex[part] << particle_container[part] << std::endl;
            }
            auto leafMortonIdx(particleMortonIndex);
            // delete duplicate indexes
            auto last = std::unique(leafMortonIdx.begin(), leafMortonIdx.end());
            leafMortonIdx.erase(last, leafMortonIdx.end());
            ///////////////////////////////////////////////////////////////////////////////////
            // io::print("rank(" + std::to_string(rank) + ")  -->  init leafMortonIdx: ", leafMortonIdx);
            ///
            ////////////////////////////////////////////////////////////////////////////////////////////
            ////   Construct a uniform distribution for the leaves/cells at the leaves level
            ///
            /// A morton index should be own by only one process
            ///
            using morton_distrib_type = typename Tree_type::data_distrib_type;

            ///
            ///  Build a uniform distribution of the leaves/cells
            ///  Here the distribution is a closed interval and not semi open one !!!
            ///
            morton_distrib_type leaves_distrib;
            morton_distrib_type particles_distrib(manager.get_num_processes());
            if(use_leaf_distribution)
            {
                leaves_distrib = std::move(scalfmm::parallel::utils::balanced_leaves(manager, leafMortonIdx));
                for(auto& interval: leaves_distrib)
                {
                    interval[1] += 1;
                }
            }
            //            io::print("rank(" + std::to_string(rank) + ")  -->  leaves_distrib: ", leaves_distrib);
            ////                End
            ////////////////////////////////////////////////////////////////////////////////////////////
            ///
            ////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////
            ////   Construct a uniform distribution for the particles
            ///  On each process we have the same number of particles. The number of leaves might differ significally
            ///
            /// A morton index should be own by only one process
            ///
            if(use_particle_distribution)
            {
                particles_distrib = std::move(scalfmm::parallel::utils::balanced_particles(
                  manager, particle_container, particleMortonIndex, number_of_particles));
                for(auto& interval: particles_distrib)
                {
                    interval[1] += 1;
                }
                if(!use_leaf_distribution)
                {
                    leaves_distrib.resize(particles_distrib.size());
                    std::copy(particles_distrib.begin(), particles_distrib.end(), leaves_distrib.begin());
                }
            }
            else
            {
                particles_distrib = leaves_distrib;
            }
            if(manager.io_master())
            {
                std::cout << cpp_tools::colors::red;
                parallel::utils::print_distrib(" -->  particles_distrib:", rank, particles_distrib);
                parallel::utils::print_distrib(" -->  leaves_distrib:", rank, leaves_distrib);
                std::cout << cpp_tools::colors::reset << std::endl;
            }
            ////                End
            ////////////////////////////////////////////////////////////////////////////////////////////
            ///
            /// Check the two distributions
            ///
            ////////////////////////////////////////////////////////////////////////////////////////////
            ///   Set the particles on the good process according to the computed distribution
            ///
            parallel::utils::fit_particles_in_distrib(manager, particle_container, particleMortonIndex,
                                                      particles_distrib, box, leaf_level, number_of_particles);
            // io::print("rank(" + std::to_string(rank) + ")  --> particle_container: ", particle_container);
            ///    All the particles are located on the good process
            ////////////////////////////////////////////////////////////////////////////////////////////
            ///
            ///   Construct the local tree based on our set of particles
            // Build and empty tree
            Tree_type localGroupTree(manager, static_cast<std::size_t>(leaf_level + 1), level_shared, order,
                                     groupSizeLeaves, groupSizeCells, box);
            /// Set true because the particles are already sorted
            ///  In fact we have all the leaves to add in leafMortonIdx - could be used to construct
            /// the tree !!!
            ///

#ifdef SCALFMM_USE_MPI

            // std::cout << cpp_tools::colors::red;
            // io::print("rank(" + std::to_string(rank) + ") leafMortonIdx: ", leafMortonIdx);
            // std::cout << cpp_tools::colors::reset << std::endl;
            ///  End
            ////////////////////////////////////////////////////////////////////////////////////////////
            ///
            /// Compute the new morton indexes associated to the particles on the process
            ///
            leafMortonIdx.resize(particle_container.size());
            // #pragma omp parallel for shared(localNumberOfParticles, box, leaf_level)
            for(std::size_t i = 0; i < particle_container.size(); ++i)
            {
                leafMortonIdx[i] = scalfmm::index::get_morton_index(particle_container[i].position(), box, leaf_level);
            }
            // io::print("rank(" + std::to_string(rank) + ")  -->  leafMortonIdx:    ", leafMortonIdx);

            // localLeafInfo contains information on leaves (morton, number of particles) own by th current process
            std::vector<tree::leaf_info_type<morton_type>> localLeafInfo(leafMortonIdx.size());
            auto start{leafMortonIdx[0]};
            int idx{0};
            localLeafInfo[idx].number_of_particles = 1;
            localLeafInfo[idx].morton = start;
            for(std::size_t i = 1; i < particle_container.size(); ++i)
            {
                if(leafMortonIdx[i] == start)
                {
                    localLeafInfo[idx].number_of_particles += 1;
                }
                else
                {
                    idx++;
                    start = leafMortonIdx[i];
                    localLeafInfo[idx].number_of_particles = 1;
                    localLeafInfo[idx].morton = start;
                    leafMortonIdx[idx] = leafMortonIdx[i];
                }
            }
            leafMortonIdx.resize(idx + 1);
            localLeafInfo.resize(leafMortonIdx.size());
            // io::print("rank(" + std::to_string(rank) + ")  -->  localLeafInfo:    ", localLeafInfo);
            // io::print("rank(" + std::to_string(rank) + ")  -->  leafMortonIdx:    ", leafMortonIdx);
            ////////////////////////////////////////////////////////////////////////////////////////
            // Build the pointer of the tree with all parameters

            if(manager.get_num_processes() > 1)
            {
                ////////////////////////////////////////////////////////////////////////////////////////////
                ///   Step 5    Construct the let according to the distributions particles and cells
                ///
                /// Find and add the leaves to add at the leaves level
                ///   we consider the particles_distrib

                auto ghostP2P_leafInfo =
                  build_let_leaves(manager, box, leaf_level, localLeafInfo, particles_distrib, separation);
                // io::print("rank(" + std::to_string(rank) + ")  -->  final ghostP2P_leafInfo:    ",
                // ghostP2P_leafInfo); io::print("rank(" + std::to_string(rank) + ")  -->  final localLeafInfo:    ",
                // localLeafInfo);

                localGroupTree.set_leaf_distribution(particles_distrib);

                // std::cout << std::flush;
                // std::cout << cpp_tools::colors::red;
                // std::cout << "END LEAF LEVEL " << std::endl;
                // std::cout << cpp_tools::colors::reset;

                /// If the distribution is not the same for the leaf and the cell we redistribute the
                /// morton index according to the uniform distribution of morton index
                ///
                //////////////////////////////////////////////////////////////////
                ///  Construct a  uniform distribution of the morton index
                ///
#ifdef TEST_
                if(use_leaf_distribution && use_particle_distribution)
                {
                    std::cout << cpp_tools::colors::red << "WARNING\n" << cpp_tools::colors::reset << std::endl;
                    try
                    {
                        inria::mpi_config conf_tmp(manager.get_communicator().raw_comm);
                        inria::distribute(conf_tmp, leafMortonIdx,
                                          inria::uniform_distribution{conf_tmp, leafMortonIdx});
                    }
                    catch(std::out_of_range& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                }
#endif

                // ///
                // /// Find and add the cells to add at the leaves level
                std::vector<morton_distrib_type> level_dist(localGroupTree.height());
                level_dist[leaf_level] = leaves_distrib;
                localGroupTree.set_cell_distribution(leaf_level, level_dist[leaf_level]);

                auto ghost_m2l_cells =
                  build_let_at_level(manager, box, leaf_level, leafMortonIdx, level_dist[leaf_level], separation);
                // io::print("rank(" + std::to_string(rank) + ")  -->  final ghost_cells(m2l):    ", ghost_m2l_cells);
                auto ghost_m2m_cells =
                  build_ghost_m2m_let_at_level(manager, box, leaf_level, leafMortonIdx, level_dist[leaf_level]);
                //                 io::print("rank(" + std::to_string(rank) + ")  -->  ghost_cells(m2m):    ", ghost_m2m_cells);

                // distribution, particles
                // std::cout << "   $$$$$$$$$$$$$$$$$$$$$$$$$ leaf level " << leaf_level << " $$$$$$$$$$$$$$$$$$$$$$$$$ "
                //           << std::endl;

                localGroupTree.create_from_leaf_level(localLeafInfo, ghostP2P_leafInfo, ghost_m2l_cells,
                                                      ghost_m2m_cells, particles_distrib[rank],
                                                      level_dist[leaf_level][rank]);
                // std::cout << "   $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ leaf level $$$$$$$$$$$$$$$$$$$$$$$$$$ "
                //           << std::endl;
                // parallel::utils::print_distrib("leaf_cell distribution ", rank, level_dist[leaf_level]);

                // build all leaves between leaf_level - 1 and level_shared -1.
                //  we use the maximum because if we don't share certain levels this number is <0
                // std::cout << "std::max(level_shared, int(localGroupTree.top_level())) "
                //           << std::max(level_shared, int(localGroupTree.top_level())) << std::endl;
                // std::cout << "  XXXXXXXXXX -> std::max(level_shared, int(localGroupTree.top_level() - 1))"
                //           << std::max(level_shared, int(localGroupTree.top_level() - 1)) << std::endl;
                ;
                for(int level = leaf_level - 1; level >= localGroupTree.top_level(); --level)
                {
                    // std::cout << "   $$$$$$$$$$$$$$$$$$$$$$$$$ level " << level << "   $$$$$$$$$$$$$$$$$$$$$$$$$ "
                    //           << std::endl;
                    std::int64_t ghost_l2l_cell{-1};

                    // Get the distribution at the current level, the ghost cell involved in l2l operator
                    //  and the morton index of the existing cells at this level
                    level_dist[level] = std::move(parallel::utils::build_upper_distribution(
                      manager, dimension, level, leafMortonIdx, ghost_l2l_cell, level_dist[level + 1]));
                    // io::print("rank(" + std::to_string(rank) + ") MortonIdx(" + std::to_string(level) + "): ",
                    //           leafMortonIdx);
                    // std::cout << " ghost_l2l_cell: " << ghost_l2l_cell << std::endl;
                    // Set the distribution in tres tree
                    localGroupTree.set_cell_distribution(level, level_dist[level]);
                    // build the m2l ghost cells at this level
                    auto ghost_cells_level =
                      build_let_at_level(manager, box, level, leafMortonIdx, level_dist[level], separation);

                    // io::print("rank(" + std::to_string(rank) + ") level=" + std::to_string(level) +
                    //             " -->  final ghost_cells(m2l):    ",
                    //           ghost_cells_level);
                    // build the m2m ghost cells at this level

                    auto ghost_m2m_cells = build_ghost_m2m_let_at_level(
                      manager, box, leaf_level, leafMortonIdx, level_dist[level], level == localGroupTree.top_level());
                    // io::print("rank(" + std::to_string(rank) + ")  -->  ghost_cells(m2m):    ", ghost_m2m_cells);

                    // Create the groupe of cells structure for this level
                    localGroupTree.create_cells_at_level(level, leafMortonIdx, ghost_cells_level, ghost_m2m_cells,
                                                         ghost_l2l_cell, level_dist[level][rank]);
                    // std::cout << "   $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ " << std::endl
                    //           << std::flush;
                }
                // std::cout << " end loop\n" << std::flush;
                manager.get_communicator().barrier();
                // std::cout << " end barrier\n" << std::flush;
            }
            else
#endif          // SCALFMM_USE_MPI
            {   // we are in sequential 1 proc
                //
                std::vector<tree::leaf_info_type<morton_type>> ghostP2P_leafInfo;
                std::vector<tree::leaf_info_type<morton_type>> ghost_m2l_leafInfo;
                std::vector<tree::leaf_info_type<morton_type>> ghost_m2m_leafInfo;
                /*                localGroupTree.create_from_leaf_level(localLeafInfo, ghostP2P_leafInfo, ghost_m2l_cells,
                                                      ghost_m2m_cells, particles_distrib[rank],
                                                      level_dist[leaf_level][rank]);
	      */
                localGroupTree.construct(particleMortonIndex);
                // then, we fill each leaf with its particles (the particle container is sorted )
                localGroupTree.fill_leaves_with_particles(particle_container);

                //
                localGroupTree.set_leaf_distribution(particles_distrib);
                localGroupTree.set_cell_distribution(leaf_level, leaves_distrib);

                for(int l = leaf_level - 1; l >= localGroupTree.top_level(); --l)
                {
                    leaves_distrib[0][0] = leaves_distrib[0][0] >> dimension;
                    leaves_distrib[0][1] = leaves_distrib[0][1] >> dimension;
                    localGroupTree.set_cell_distribution(l, leaves_distrib);
                }
            }

            // std::cout << cpp_tools::colors::red << std::endl << std::flush;
            // std::cout << "set iterators \n" << std::flush << std::flush;
            localGroupTree.set_valid_iterators(true);
            // std::cout << "begin fill_leaves_with_particles \n" << std::flush;
            localGroupTree.fill_leaves_with_particles(particle_container);
            // std::cout << "end fill_leaves_with_particles \n" << std::flush;

            // std::cout << cpp_tools::colors::reset << std::endl;
            // std::cout << cpp_tools::colors::green << " --> End let::group_let() " << cpp_tools::colors::reset
            //           << std::endl
            //           << std::flush;

            return localGroupTree;
        }

    }   // namespace let
}   // namespace scalfmm::tree

#endif
