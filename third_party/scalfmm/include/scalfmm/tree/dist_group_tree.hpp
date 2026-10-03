// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tree/dist_group_tree.hpp
// --------------------------------

#ifndef SCALFMM_TREE_DIST_GROUP_TREE_HPP
#define SCALFMM_TREE_DIST_GROUP_TREE_HPP

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "scalfmm/tree/box.hpp"
#include <scalfmm/tree/group_let.hpp>
#include <scalfmm/tree/group_tree_view.hpp>
#include <scalfmm/utils/io_helpers.hpp>

#include <scalfmm/parallel/comm_access.hpp>

#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/parallel_manager/parallel_manager.hpp>

namespace scalfmm::component
{
    /**
     * @brief
     *
     * @tparam Cell
     * @tparam Leaf
     * @tparam box<typename Leaf::position_type>
     */
    template<typename Cell, typename Leaf, typename Box = box<typename Leaf::position_type>>
    class dist_group_tree : public group_tree_view<Cell, Leaf, Box>
    {
      public:
        using morton_type = std::int64_t;
        using data_distrib_value_type = std::array<morton_type, 2>;
        using data_distrib_type = std::vector<data_distrib_value_type>;
        using base_type = group_tree_view<Cell, Leaf, Box>;
        using leaf_iterator_type = typename base_type::leaf_iterator_type;
        using const_leaf_iterator_type = typename base_type::const_leaf_iterator_type;
        using cell_iterator_type = typename base_type::cell_iterator_type;
        using cell_group_level_iterator_type = typename base_type::cell_group_level_type::iterator;
        using iterator_type = typename base_type::iterator_type;
        using const_iterator_type = typename base_type::const_iterator_type;
        using grp_access_type = std::pair<cell_group_level_iterator_type, int>;

        static constexpr std::size_t dimension = base_type::box_type::dimension;

        /**
         * @brief Construct a new dist group tree object
         *
         * @param parallel_manager
         * @param tree_height
         * @param level_shared
         * @param order
         * @param size_leaf_blocking
         * @param size_cell_blocking
         * @param box
         */
        explicit dist_group_tree(cpp_tools::parallel_manager::parallel_manager& parallel_manager,
                                 std::size_t tree_height, const int level_shared, std::size_t order,
                                 std::size_t size_leaf_blocking, std::size_t size_cell_blocking, Box const& box)
          : base_type(tree_height, order, size_leaf_blocking, size_cell_blocking, box)
          , m_parallel_manager(parallel_manager)
          , m_level_shared{level_shared}
          , transfer_acces(tree_height, parallel_manager.get_num_processes())

        {
            m_cell_distrib.resize(tree_height);
            up_down_acces.resize(tree_height);

            // transfer_acces.m_receive_multipoles_type.resize(tree_height);
            // transfer_acces.m_send_multipoles_type.resize(tree_height);
            // transfer_acces.m_send_morton.resize(tree_height);
            // for(auto& vec: transfer_acces.m_send_morton)
            // {
            //     vec.resize(parallel_manager.get_num_processes());
            // }
            // m_receive_cells_access.resize(tree_height);
            // for(auto& vec: transfer_acces.m_receive_cells_access)
            // {
            //     vec.resize(parallel_manager.get_num_processes());
            // }
            // m_send_cells_access.resize(tree_height);
            // for(auto& vec: transfer_acces.m_send_cells_access)
            // {
            //     vec.resize(parallel_manager.get_num_processes());
            // }
        }

        // template<typename ParticleContainer>
        // explicit dist_group_tree(std::size_t tree_height, const int level_shared, std::size_t order, Box const& box,
        //                          std::size_t size_leaf_blocking, std::size_t size_cell_blocking,
        //                          ParticleContainer const& particle_container,
        //                          bool particles_are_sorted = false /*, int in_left_limit = -1*/)
        //   : base_type(tree_height, order, box, size_leaf_blocking, size_cell_blocking, particle_container,
        //               particles_are_sorted /*, in_left_limit*/)
        //   , m_level_shared{level_shared}

        // {
        //     m_cell_distrib.resize(tree_height);
        // }
        // template<typename ParticleContainer>
        // explicit dist_group_tree(cpp_tools::parallel_manager::parallel_manager& parallel_manager,
        //                          const std::size_t tree_height, const int level_shared, std::size_t order,
        //                          Box const& box, const std::size_t size_element_blocking,
        //                          const std::size_t size_cell_blocking)
        //   : base_type(tree_height, order, box, size_element_blocking, size_cell_blocking)
        //   , m_parallel_manager(parallel_manager)
        //   , m_level_shared{level_shared}
        // {
        // }

        /**
         * @brief Set the leaf distribution object
         *
         * @param in_leaf_distrib The leaf distribution
         */
        inline auto set_leaf_distribution(const data_distrib_type& in_leaf_distrib) -> void
        {
            m_leaf_distrib = in_leaf_distrib;
        }

        /**
         * @brief Get the leaf distribution object
         *
         * @param in_leaf_distrib The leaf distribution
         */
        inline auto get_leaf_distribution() -> data_distrib_type const& { return m_leaf_distrib; }

        /**
         * @brief Set the cell distribution object
         *
         * @param in_level  level
         * @param in_cell_distrib the cell distribution at in_level
         */
        inline auto set_cell_distribution(const int in_level, const data_distrib_type& in_cell_distrib) -> void
        {
            m_cell_distrib.at(in_level) = in_cell_distrib;
        }

        /**
         * @brief Get the cell distribution object at level in_level
         *
         * @param in_level the level
         * @return data_distrib_type const&
         */
        inline auto get_cell_distribution(const int in_level) -> data_distrib_type const&
        {
            return m_cell_distrib.at(in_level);
        }

        /**
         * @brief print the distribution of leaves and cells
         *
         * @param out the stream where we display
         * @param verbose to display explanations of what we print
         */
        auto print_distrib(std::ostream& out, bool verbose = true) -> void
        {
            if(m_cell_distrib.size() > 0)
            {
                std::string header;
                if(verbose)
                {
                    out << "Tree distribution" << std::endl;
                }
                out << " height" << base_type::height() << "top_level " << base_type::top_level() << std::endl;
                for(int l = base_type::top_level(); l < base_type::height(); ++l)
                {
                    if(verbose)
                    {
                        header = "  Level " + std::to_string(l) + " cell distribution: \n";
                    }
                    if(m_cell_distrib[l].size() > 0)
                    {
                        parallel::utils::print_distrib(out, std::move(header), m_cell_distrib[l]);
                    }
                    else
                    {
                        std::cout << "No cells at level " << l << std::endl;
                    }
                }
                if(verbose)
                {
                    header = "  leaf distribution: \n";
                }
                // io::print(out, std::move(header), m_leaf_distrib);
                parallel::utils::print_distrib(out, std::move(header), m_leaf_distrib);
            }
            else
            {
                std::cout << " m_cell_distrib.size() == 0 !!!!!\n";
            }
        }

        /**
         * @brief ghost cell level display on output stream
         *
         * @param out the stream where we display
         */
        auto print_morton_ghosts(std::ostream& out) -> void
        {
            out << "The ghost leaves " << std::endl << std::flush;

            for(auto it = this->begin_leaves(); it != this->begin_mine_leaves(); ++it)
            {
                for(std::size_t index = 0; index < (*it)->size(); ++index)
                {
                    auto const& morton = (*it)->ccomponent(index).csymbolics().morton_index;
                    out << morton << " ";
                }
            }

            for(auto it = this->end_mine_leaves(); it != this->end_leaves(); ++it)
            {
                for(std::size_t index = 0; index < (*it)->size(); ++index)
                {
                    auto const& morton = (*it)->ccomponent(index).csymbolics().morton_index;
                    out << morton << " ";
                }
            }
            out << std::endl << std::flush;
            //////////////////////////// cells
            for(int level = base_type::top_level(); level < base_type::m_tree_height; ++level)
            {
                out << "Level " << level << " ghost cells " << std::endl << std::flush;

                for(auto it = this->begin_cells(level); it != this->begin_mine_cells(level); ++it)
                {
                    for(std::size_t index = 0; index < (*it)->size(); ++index)
                    {
                        auto const& morton = (*it)->ccomponent(index).csymbolics().morton_index;
                        out << morton << " ";
                    }
                }
                // out << "\n nb grp= " << std::distance(this->end_mine_cells(level), this->end_cells(level)) <<
                // std::endl
                //     << std::flush;
                for(auto it = this->end_mine_cells(level); it != this->end_cells(level); ++it)
                {
                    for(std::size_t index = 0; index < (*it)->size(); ++index)
                    {
                        auto const& morton = (*it)->ccomponent(index).csymbolics().morton_index;
                        out << morton << " ";
                    }
                }
                out << std::endl << std::flush;
            }
        }

        /**
         * @brief Destroy the dist group tree object
         *
         */
        ~dist_group_tree()
        {
            // std::cout << cpp_tools::colors::red;
            // std::cout << " ~dist_group_tree() " << std::endl;
            // std::cout << cpp_tools::colors::reset;
            // std::cout << " end ~dist_group_tree() " << std::endl;
        }

        /**
         * @brief
         *
         * @return const int
         */
        inline auto start_duplicated_level() -> const int { return m_level_shared; }

        /**
         * @brief Create groups of cells at level object
         *
         *  ghosts_m2l indexes are slit in two sets one with indices before me and the second for indices after
         * me. we add the ghost cell involved in l2l operator on the left and the ghost cells involved in m2m
         * operator on the right. the Then we create separately the three set of groups (2 for the ghosts and one
         * for my cells)
         *
         * @tparam VectorMortonIndexType
         * @param level the level where we construct the groups of cells
         * @param mortonIdx the index cells I own
         * @param ghosts_m2l  the ghost cells needed in the m2l operator (VectorMortonIndexType)
         * @param ghosts_m2m  the ghost cells needed in the m2m operator (VectorMortonIndexType)
         * @param ghost_l2l  the ghost cell needed in the l2l operator (int64)
         * @param cell_distrib the cell distribution
         */
        template<typename VectorMortonIndexType>
        auto create_cells_at_level(const int level, VectorMortonIndexType const& mortonIdx,
                                   VectorMortonIndexType const& ghosts_m2l, VectorMortonIndexType const& ghosts_m2m,
                                   const std::int64_t& ghost_l2l, data_distrib_value_type const& cell_distrib) -> void
        {
            io::print(std::clog, "create_cells_at_level mortonIdx", mortonIdx);
            io::print(std::clog, "ghosts_m2l", ghosts_m2l);
            io::print(std::clog, "ghosts_m2m", ghosts_m2m);
            std::clog << std::endl << std::flush;

            // construct group of cells at leaf level
            auto first_index = cell_distrib[0];
            auto last =
              std::find_if(ghosts_m2l.begin(), ghosts_m2l.end(), [&first_index](auto& x) { return x > first_index; });
            //
            int to_add{0};
            if(std::distance(ghosts_m2l.begin(), last) > 0)
            {
                to_add = (ghost_l2l == -1) or (*(last - 1) == ghost_l2l) ? 0 : 1;
            }
            VectorMortonIndexType ghost_left_mortonIdx(std::distance(ghosts_m2l.begin(), last) + to_add);
            std::copy(ghosts_m2l.begin(), last, ghost_left_mortonIdx.begin());
            if(to_add == 1)
            {
                ghost_left_mortonIdx.back() = ghost_l2l;
            }

            // io::print("create_from_leaf : ghost_left_mortonIdx ", ghost_left_mortonIdx);
            this->build_groups_of_cells_at_level(ghost_left_mortonIdx, level, false);
            this->build_cells_in_groups_at_level(ghost_left_mortonIdx, base_type::m_box, level);

            // io::print("ghost_left_mortonIdx ", ghost_left_mortonIdx);
            // std::cout << std::endl << std::flush;

            auto left_block_cells = std::move(base_type::m_group_of_cell_per_level.at(level));
            // std::cout << " merge_unique_fast " << std::endl << std::flush;
            auto ghost_right_mortonIdx = scalfmm::parallel::utils::merge_unique_fast<VectorMortonIndexType>(
              last, ghosts_m2l.end(), ghosts_m2m.begin(), ghosts_m2m.end());

            // io::print("create_from_leaf : ghost_right_mortonIdx ", ghost_right_mortonIdx);
            std::cout << std::endl << std::flush;

            this->build_groups_of_cells_at_level(ghost_right_mortonIdx, level, false);
            this->build_cells_in_groups_at_level(ghost_right_mortonIdx, base_type::m_box, level);
            auto right_block_cells = std::move(base_type::m_group_of_cell_per_level.at(level));
            this->build_groups_of_cells_at_level(mortonIdx, level);
            this->build_cells_in_groups_at_level(mortonIdx, base_type::m_box, level);

            auto local_block_cells = std::move(base_type::m_group_of_cell_per_level.at(level));
            auto all_cells_blocks =
              scalfmm::tree::let::merge_blocs(left_block_cells, local_block_cells, right_block_cells);
            // std::cout << "  All cells blocks at level " << level << " size: " << all_cells_blocks.size() << std::endl
            //           << std::flush;
            // int tt{0};
            // for(auto pg: all_cells_blocks)
            // {
            //     std::cout << "block index " << tt++ << " ";
            //     pg->print();
            //     std::cout << std::endl << std::flush;
            //     // pg->cstorage().print_block_data(std::cout);
            // }
            // std::cout << std::endl << std::flush;
            base_type::m_group_of_cell_per_level.at(level) = std::move(all_cells_blocks);
            auto& grp_level = base_type::m_group_of_cell_per_level.at(level);
            int idx{0};
            for(auto it = std::begin(grp_level); it != std::end(grp_level); ++it, ++idx)
            {
                it->get()->symbolics().idx_global = idx;
            }
        }

        /**
         * @brief Create groups of cells and leaves from the leaf level
         *
         *  The ghosts involve in p2p and m2l are slit in two sets one with indices before me and the second for indices
         * after me. Then we create separately the three set of groups (2 for the ghosts and one for my cells)
         *
         * leaf_info is a structure that containsvthe Morton index and the number of particles inside the leaf.
         * @tparam VectorLeafInfoType
         * @tparam VectorMortonIndexType
         * @param localLeaves  the Leaf Info the lezves that I own
         * @param ghosts_p2p the ghosts (leaf_info structure) needed in the p2p operator
         * @param ghosts_m2l the ghost cells needed in the m2l operator
         * @param leaf_distrib the leaf distribution
         * @param cell_distrib the cell distribution
         */
        template<typename VectorLeafInfoType, typename VectorMortonIndexType>
        auto create_from_leaf_level(VectorLeafInfoType& localLeaves, VectorLeafInfoType& ghosts_p2p,
                                    VectorMortonIndexType const& ghosts_m2l, VectorMortonIndexType const& ghosts_m2m,
                                    data_distrib_value_type const& leaf_distrib,
                                    data_distrib_value_type const& cell_distrib) -> void
        {
            using morton_type = typename VectorLeafInfoType::value_type::morton_type;
            //
            // compute number of particles
            //
            std::vector<morton_type> mortonIdx;
            std::vector<std::size_t> number_of_part;
            std::tie(mortonIdx, number_of_part) = tree::let::split_structure(localLeaves.cbegin(), localLeaves.cend());
            // io::print("create_from_leaf :morton(p2p)  ", mortonIdx);
            // io::print("create_from_leaf :nbpart  ", number_of_part);

            this->build_groups_of_leaves(mortonIdx, number_of_part, base_type::m_box);
            auto localBlocks = std::move(base_type::m_group_of_leaves);

            // Build group on the left
            auto first_index = leaf_distrib[0];
            auto last = std::find_if(ghosts_p2p.begin(), ghosts_p2p.end(),
                                     [&first_index](auto& x) { return x.morton > first_index; });
            std::vector<morton_type> ghost_left_mortonIdx;
            std::vector<std::size_t> ghost_left_number_of_part;
            std::tie(ghost_left_mortonIdx, ghost_left_number_of_part) =
              tree::let::split_structure(ghosts_p2p.begin(), last);
            // io::print("create_from_leaf : left morton (p2)  ", ghost_left_mortonIdx);
            // io::print("create_from_leaf : left nbpart  ", ghost_left_number_of_part);

            this->build_groups_of_leaves(ghost_left_mortonIdx, ghost_left_number_of_part, base_type::m_box, false);
            auto ghost_left_Blocks = std::move(base_type::m_group_of_leaves);
            std::vector<morton_type> ghost_right_mortonIdx;
            std::vector<std::size_t> ghost_right_number_of_part;
            std::tie(ghost_right_mortonIdx, ghost_right_number_of_part) =
              tree::let::split_structure(last, ghosts_p2p.end());

            // io::print("create_from_leaf : right morton(p2p)  ", ghost_right_mortonIdx);
            // io::print("create_from_leaf : right nbpart  ", ghost_right_number_of_part);

            this->build_groups_of_leaves(ghost_right_mortonIdx, ghost_right_number_of_part, base_type::m_box, false);
            auto ghost_right_Blocks = std::move(base_type::m_group_of_leaves);

            // Merge the three block structure
            auto all_blocks = scalfmm::tree::let::merge_blocs(ghost_left_Blocks, localBlocks, ghost_right_Blocks);

            base_type::m_group_of_leaves = std::move(all_blocks);
            int idx{0};
            for(auto it = std::begin(base_type::m_group_of_leaves); it != std::end(base_type::m_group_of_leaves);
                ++it, ++idx)
            {
                it->get()->symbolics().idx_global = idx;
            }
            //  leaves are created
            /////////////////////////////////////////////////////////////////////////////////////
            // same code for the cells we change
            //  - ghosts_p2p in ghosts_m2l
            //  - base_type::m_group_of_leaf in base_type::m_group_of_cell_per_level[leaf_level]
            //
            // we construct the leaves in each group
            // io::print("create_from_leaf : m2l_ghost", ghosts_m2l);
            // io::print(" cell_distrib  ", cell_distrib);

            auto leaf_level = base_type::m_tree_height - 1;
            std::int64_t ghost_l2l_cell{-1};
            // std::cout << "       this->create_cells_at_level  \n" << std::flush;

            this->create_cells_at_level(leaf_level, mortonIdx, ghosts_m2l, ghosts_m2m, ghost_l2l_cell, cell_distrib);
            // std::cout << " END create_from_leaf_level \n" << std::flush;
            //
        }

        /**
         * @brief Construct the cells at levels in [level_shared, top_level]
         *
         * The data are duplicated on all processors  for level between level_shared and top_level.
         * The bounds are included.
         *
         * @tparam MortonDistVector
         * @tparam MortonIdxVector
         * @param para the manager of the parallelism
         * @param level_dist the vector of distribution on all levels
         * @param leafMortonIdx the morton indexes of the existing cells at level level_shared
         * @param level_shared the level where the cells  are duplicated on the first levels
         */
        template<typename MortonDistVector, typename MortonIdxVector>
        auto build_other_shared_levels(cpp_tools::parallel_manager::parallel_manager& para,
                                       MortonDistVector& level_dist, MortonIdxVector& leafMortonIdx,
                                       int const& level_shared) -> void
        {
            // Get all cells on all process
            const int nproc = para.get_num_processes();
            const auto rank = para.get_process_id();
            // parallel::utils::print_distrib("level < level_shared:", rank, level_dist[level_shared]);

            int local_nbcells{int(leafMortonIdx.size())}, global_nbcells{};
            // nbElt the number of cells on eachu process
            std::vector<int> nbElt(nproc), displs(nproc);
            auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<int>();
            para.get_communicator().allgather(&local_nbcells, 1, mpi_type, &nbElt[0], 1, mpi_type);
            // io::print("rank(" + std::to_string(rank) + ")  -->  nbElt:", nbElt);
            // All gather the cells
            displs[0] = 0;
            global_nbcells = nbElt[0];
            for(int i = 1; i < nproc; ++i)
            {
                displs[i] = global_nbcells;   // displs[i - 1] + nbElt[i - 1];
                global_nbcells += nbElt[i];
            }
            std::vector<morton_type> shared_cell_index(global_nbcells);
            auto mpi_morton_type = cpp_tools::parallel_manager::mpi::get_datatype<morton_type>();
            // io::print("rank(" + std::to_string(rank) + ") leafMortonIdx:", leafMortonIdx);

            MPI_Allgatherv(leafMortonIdx.data(), nbElt[rank], mpi_morton_type, shared_cell_index.data(), nbElt.data(),
                           displs.data(), mpi_morton_type, para.get_communicator());

            // meta::td<decltype(level_dist[level_shared])> u;
            //
            // The set of cells is the same on each process
            for(int level = level_shared; level >= base_type::top_level(); --level)
            {
                // std::cout << "   TO DO SEQUENTIAL level=" << level << std::endl << std::flush;
                // io::print("rank(" + std::to_string(rank) + ") shared_cell_index:", shared_cell_index);
                // allocate the distribution at the current level and set it
                level_dist[level].resize(nproc);
                for(auto& p: level_dist[level])
                {
                    p[0] = shared_cell_index[0];
                    p[1] = shared_cell_index[shared_cell_index.size() - 1] + 1;
                }
                // parallel::utils::print_distrib(" dist[" + std::to_string(level) + "]: ", rank,
                // level_dist[level]);
                this->set_cell_distribution(level, level_dist[level]);

                // construct group of cells at current level
                base_type::build_groups_of_cells_at_level(shared_cell_index, level);
                // construct cells in group of current level
                base_type::build_cells_in_groups_at_level(shared_cell_index, base_type::m_box, level);
                // Move indexes to next level
                parallel::utils::move_index_to_upper_level<base_type::dimension>(shared_cell_index);
            }
        }

        /**
         * @brief Set the valid begin and end iterators on cell and leaf group
         *
         * Here the iterators begin_mine_{cells,leaves} and begin_{cells,leaves} may be different
         *
         */
        auto set_valid_iterators(bool verbose = false) -> void
        {
            // set iterators for leaf
            //  begin_mine_leaf
            auto& vectG = base_type::m_group_of_leaves;
            //
            base_type::m_view_on_my_leaf_groups[0] = std::end(base_type::m_group_of_leaves);
            for(auto it = std::begin(base_type::m_group_of_leaves); it != std::end(base_type::m_group_of_leaves); ++it)
            {
                if(it->get()->csymbolics().is_mine)
                {
                    base_type::m_view_on_my_leaf_groups[0] = it;
                    break;
                }
            }
            // set iterator end_mine_leaf
            auto it_group = std::end(base_type::m_group_of_leaves);
            base_type::m_view_on_my_leaf_groups[1] = it_group;
            --it_group;   // to have a valid iterator
            for(auto it = it_group; it != --(std::begin(base_type::m_group_of_leaves)); --it)
            {
                if(it->get()->csymbolics().is_mine)
                {
                    auto itpp = it;
                    ++itpp;   // We increment by one to get the final iterator.
                    base_type::m_view_on_my_leaf_groups[1] = itpp;
                    break;
                }
            }
            auto leaf_level = base_type::m_tree_height - 1;

            if(verbose)
            {
                std::cout << " set_valid_iterators(leaves):\n ";
                std::cout << " level = " << leaf_level << " total grp.size= " << vectG.size()
                          << " total grp= " << std::distance(vectG.begin(), base_type::end_leaves())
                          << " total grp= " << std::distance(base_type::begin_leaves(), base_type::end_leaves())
                          << "  begin_mine at idx " << std::distance(vectG.begin(), base_type::begin_mine_leaves())
                          << " end_mine at idx " << std::distance(vectG.begin(), base_type::end_mine_leaves())
                          << std::endl;
            }
            ///////////////// End leaves
            // std::cout << " begin barrier set_valid_iterators cells\n" << std::flush;
            // m_parallel_manager.get_communicator().barrier();
            // std::cout << " end barrier set_valid_iterators cells\n" << std::flush;
            if(verbose)
            {
                std::cout << " set_valid_iterators(cells):\n " << std::flush;
            }
            auto cell_level_it = this->begin_cells() + leaf_level;

            for(int level = leaf_level; level >= base_type::m_top_level; --level)
            {
                auto group_of_cell_begin = std::begin(*(cell_level_it));
                auto group_of_cell_end = std::end(*(cell_level_it));
                auto& my_iterator_cells_at_level = base_type::m_view_on_my_cell_groups[level];
                auto& vect_cell_group_level = base_type::m_group_of_cell_per_level[level];

                my_iterator_cells_at_level[0] = group_of_cell_end;

                int idx{0};
                for(auto it = group_of_cell_begin; it != group_of_cell_end; ++it, ++idx)
                {
                    if(it->get()->csymbolics().is_mine)
                    {
                        my_iterator_cells_at_level[0] = it;
                        break;
                    }
                }
                // Search the last mine group
                my_iterator_cells_at_level[1] = group_of_cell_end;
                int last = vect_cell_group_level.size() - 1;
                for(int idx = vect_cell_group_level.size() - 1; idx >= 0; --idx)
                {
                    auto& grp = *(vect_cell_group_level[idx]);
                    if(grp.csymbolics().is_mine)
                    {
                        my_iterator_cells_at_level[1] = group_of_cell_end - (last - idx);
                        break;
                    }
                }
                if(verbose)
                {
                    std::cout << " level = " << level << " total grp= " << vect_cell_group_level.size()
                              << "    numbers of ghost left grp="
                              << std::distance(group_of_cell_begin, base_type::m_view_on_my_cell_groups[level][0])
                              << ", my grp="
                              << std::distance(base_type::m_view_on_my_cell_groups[level][0],
                                               base_type::m_view_on_my_cell_groups[level][1])
                              << ", ghost right grp= "
                              << std::distance(base_type::m_view_on_my_cell_groups[level][1], group_of_cell_end)
                              << "  pos begin_mine at idx "

                              << std::distance(group_of_cell_begin, base_type::m_view_on_my_cell_groups[level][0]) - 1
                              << " pps end_mine at idx "
                              << std::distance(group_of_cell_begin, base_type::m_view_on_my_cell_groups[level][1]) - 1
                              << std::endl;
                }
                --cell_level_it;
            }
        }

        /**
         * @brief
         *
         * @tparam ParticleContainer
         * @param particle_container
         */
        template<typename ParticleContainer>
        auto fill_leaves_with_particles(ParticleContainer const& particle_container) -> void
        {
#ifndef TOTO
            //	  using scalfmm::details::tuple_helper;
            // using proxy_type = typename particle_type::proxy_type;
            // using const_proxy_type = typename particle_type::const_proxy_type;
            // using outputs_value_type = typename particle_type::outputs_value_type;
            auto begin_container = std::begin(particle_container);
            std::size_t group_index{0}, part_src_index{0};
            // std::cout << cpp_tools::colors::red << " particle_container.size() " << particle_container.size()
            //           << std::endl;
            // std::cout << " nb of mine grp "
            //           << std::distance(base_type::cbegin_mine_leaves(), base_type::cend_mine_leaves())
            //           << cpp_tools::colors::reset << std::endl
            //           << std::flush;
            for(auto pg = base_type::cbegin_mine_leaves(); pg != base_type::cend_mine_leaves(); ++pg)
            {
                auto& group = *(pg->get());

                std::size_t leaf_index{0};
                auto leaves_view = group.components();
                // loop on leaves
                for(auto const& leaf: leaves_view)
                {
                    // get the leaf container
                    auto leaf_container_begin = leaf.particles().first;
                    // std::cout << " leaf index " << leaf.index() << " leaf.size() " << leaf.size() << std::endl
                    //           << std::flush;
                    // copy the particle in the leaf
                    for(std::size_t index_part = 0; index_part < leaf.size(); ++index_part)
                    {
                        // std::cout << " --> index_part " << index_part << " part_src_index " << part_src_index
                        //           << std::endl
                        //           << std::flush;
                        // get the source index in the source container
                        // auto source_index = std::get<1>(tuple_of_indexes.at(part_src_index));
                        // jump to the index in the source container
                        auto jump_to_particle = begin_container;
                        std::advance(jump_to_particle, int(part_src_index));
                        // copy the particle

                        // *leaf_container_begin = particle_container.particle(source_index).as_tuple();
                        // std::cout << " -->            " << part_src_index << " p "
                        // << particle_container.at(part_src_index) << std::endl;
                        *leaf_container_begin = particle_container.at(part_src_index).as_tuple();

                        //         proxy_type particle_in_leaf(*leaf_container_begin);
                        //         // set the outputs to zero
                        //         for(std::size_t ii{0}; ii < particle_type::outputs_size; ++ii)
                        //         {
                        //             particle_in_leaf.outputs(ii) = outputs_value_type(0.);
                        //         }

                        ++part_src_index;
                        ++leaf_container_begin;
                    }
                    ++leaf_index;
                }
                ++group_index;
                // std::cout << " group " << group << std::endl;
            }
#ifdef _DEBUG_BLOCK_DATA
            std::clog << "  FINAL block\n";
            int tt{0};
            for(auto pg: m_group_of_leaf)
            {
                std::clog << "block index " << tt++ << std::endl;
                pg->cstorage().print_block_data(std::clog);
            }
            std::clog << "  ---------------------------------------------------\n";
#endif
#endif
        }

        /**
         * @brief Get the parallel manager object
         *
         * @return cpp_tools::parallel_manager::parallel_manager&
         */
        inline auto get_parallel_manager() -> cpp_tools::parallel_manager::parallel_manager&
        {
            return m_parallel_manager;
        }

        auto inline get_send_multipole_types(const int& level) -> std::vector<MPI_Datatype>&
        {
            return transfer_acces.get_send_multipole_types(level);
        }

        // auto inline print_send_multipole_types(const int& level) -> void
        // {
        //     auto const& type = m_send_multipoles_type[level];
        //     for(int p = 0; p < type.size(); ++p)
        //     {
        //         std::cout << "   ptr_data_type(" << p << ") " << &(type[p]) << " level: " << level << std::endl
        //                   << std::flush;
        //     }
        // }
        auto inline get_receive_multipole_types(const int& level) -> std::vector<MPI_Datatype>&
        {
            return transfer_acces.get_receive_multipole_types(level);
        }
        auto inline send_morton_indexes(int const& level, int const& proc) -> std::vector<morton_type>&
        {
            return transfer_acces.send_morton_indexes(level, proc);
        }
        auto inline send_morton_indexes(int const& level) -> std::vector<std::vector<morton_type>>&
        {
            return transfer_acces.send_morton_indexes(level);
        }
        auto inline receive_cells_access(int const& level) -> std::vector<std::vector<grp_access_type>>&
        {
            return transfer_acces.receive_cells_access(level);
        }

        auto inline send_cells_access(int const& level) -> std::vector<std::vector<grp_access_type>>&
        {
            return transfer_acces.send_cells_access(level);
        }
        // auto inline set_receive_access(const int level, std::vector<std::vector<grp_access_type>>& access) -> void
        // {
        //     m_receive_access[level] = std::move(access);
        // }
        // auto inline get_ptr_send_access(const int& level) -> grp_access_type*
        // {
        //     return m_send_multipoles_type[level].data();
        // }
        // auto inline get_ptr_receive_access(const int& level) -> grp_access_type*
        // {
        //     return m_receive_multipoles_type[level].data();
        // }
        auto get_up_down_access(const int level) -> UpDownDataAccess& { return up_down_acces[level]; }
        auto get_up_down_access(const int level) const -> UpDownDataAccess const& { return up_down_acces[level]; }

      private:
        /**
         * @brief a reference on the parallel manager
         *
         */
        cpp_tools::parallel_manager::parallel_manager& m_parallel_manager;

        /**
         * @brief Distribution of leaves at different level. The interval is a range (open on the right)
         *
         */
        data_distrib_type m_leaf_distrib;

        /**
         * @brief Distribution of cells at different level
         *
         */
        std::vector<data_distrib_type> m_cell_distrib;

        /**
         * @brief The level at which cells are duplicated on processors. If the level is negative, nothing is duplicated.
         *
         */
        int m_level_shared;
        ///
        // Structure for communications in transfert step

        // std::vector<std::vector<MPI_Datatype>> m_send_multipoles_type;
        // std::vector<std::vector<MPI_Datatype>> m_receive_multipoles_type;
        // // vector per level and per process
        // std::vector<std::vector<std::vector<morton_type>>> m_send_morton;
        // std::vector<std::vector<std::vector<grp_access_type>>> m_receive_cells_access;
        // std::vector<std::vector<std::vector<grp_access_type>>> m_send_cells_access;

        std::vector<UpDownDataAccess> up_down_acces;
        transferDataAccess<morton_type, grp_access_type> transfer_acces;
    };
}   // namespace scalfmm::component

#endif
