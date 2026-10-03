// --------------------------------
// See LICENCE file at project root
// File : algorithm/mpi/downward.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_MPI_DOWNWARD_HPP
#define SCALFMM_ALGORITHMS_MPI_DOWNWARD_HPP

#include "scalfmm/operators/l2l.hpp"
#ifdef _OPENMP
#include "scalfmm/algorithms/omp/downward.hpp"
#include <omp.h>
#endif   // _OPENMP

#include <cpp_tools/parallel_manager/parallel_manager.hpp>

namespace scalfmm::algorithms::mpi::pass
{
    /// @brief Construct the vector of dependencies (child group)
    /// @tparam IteratorType
    /// @tparam MortonType
    /// @tparam Dependencies_t
    /// @tparam dimension
    /// @param begin first iterator on the groups of cells (child)
    /// @param end  last iterator on the groups of cells (child)
    /// @param parent_morton_index the parent index
    /// @param dependencies  the vector of dependencies
    template<int dimension, typename IteratorType, typename MortonType, typename Dependencies_t>
    void build_downward_dependencies(IteratorType begin, IteratorType end, MortonType const& parent_morton_index,
                                     Dependencies_t& dependencies)
    {
        for(auto grp_ptr = begin; grp_ptr != end; ++grp_ptr)
        {
            auto const& csymb = (*grp_ptr)->csymbolics();
            // iterate on the cells in the same group
            // we move forward in the index vector
            // std::cout << "[" << csymb.starting_index << " < " << parent_morton_index << " < " << csymb.ending_index
            //           << "] ?" << std::endl
            //           << std::flush;
            if(parent_morton_index == ((csymb.ending_index - 1) >> dimension))
            {
                // std::cout << parent_morton_index << " add depend for grp with Int [" << csymb.starting_index << ", "
                //           << csymb.ending_index << "]" << std::endl;
                dependencies.push_back(&(grp_ptr->get()->ccomponent(0).clocals(0)));
            }
            else
            {
                break;
            }
        }
    }
    /**
     * @brief perform the l2l communications for the father level
     *
     * @tparam Tree
     *
     * @param level current father level
     * @param tree the tree
     */
    template<typename Tree>
    inline auto downward_communications_level(const int& level, Tree& tree) -> void
    {
        //  value_type value of the local array
        using value_type = typename Tree::base_type::cell_type::value_type;
        using dep_type = typename Tree::group_of_cell_type::symbolics_type::ptr_multi_dependency_type;

        static constexpr int nb_outputs = Tree::cell_type::storage_type::outputs_size;
        static constexpr std::size_t dimension = Tree::base_type::box_type::dimension;
        // number of theoretical children
        constexpr int nb_children = math::pow(2, dimension);
        static constexpr auto prio{omp::priorities::max};
        //
        auto child_level = level + 1;

        // compute the size of the locals to send (generic) versus  math::pow(order, dimension)
        auto it_last_parent_group = tree.end_mine_cells(level) - 1;       // last group  I own father
        auto pos = it_last_parent_group->get()->size() - 1;               // index of the last cell
        auto const& cell = it_last_parent_group->get()->component(pos);   // the cell
        auto const& m = cell.clocals();
        auto size_local{int(nb_outputs * m.at(0).size())};   // size of a local
        //
        // For the communications
        auto& para = tree.get_parallel_manager();
        auto* comm = &(para.get_communicator());
        auto rank = comm->rank();
        int nb_proc = comm->size();
        int tag_data = 2201 + 10 * level;
        std::vector<dep_type> dependencies_in;
        //
        auto ptr_tree = &tree;
        auto const& distrib = tree.get_cell_distribution(child_level);
        // std::clog << "distrib me [" << distrib[rank][0] << "," << distrib[rank][1] << "]\n";
        // Send to the right the last locals
        if(rank != nb_proc - 1)
        {
            // std::clog << "   Send step " << level << "\n";
            // get the  distribution at child  level
            auto last_child_index = distrib[rank][1] - 1;
            auto first_child_index_after_me = distrib[rank + 1][0];
            // dependencies in on the group
            // Check if the last mine and the first right ghost have the same father
            auto parent_morton_index = last_child_index >> dimension;
            // std::clog << " downward last_child_index " << last_child_index << " its parent " << parent_morton_index
            //           << "  first_child_index_after_me " << first_child_index_after_me << " its parent "
            //           << (first_child_index_after_me >> dimension) << std::endl
            //           << std::flush;
            if(parent_morton_index == (first_child_index_after_me >> dimension))
            {
                // Two processes share the same parent
                // iterator on the my first child
                auto first_group_of_child = tree.begin_mine_cells(child_level);
                auto first_index_child = first_group_of_child->get()->component(0).index();
                auto parent_of_last_index_child = first_index_child >> dimension;

                std::cout << std::flush;
                // dependencies on the parent group
                auto dep_parent = &(it_last_parent_group->get()->ccomponent(0).clocals(0));
                // std::cout << " downward dep(in) on groupe dep_parent  " << dep_parent << std::endl << std::flush;
                // depend(iterator(std::size_t it = 0 dependencies.size()), inout : (dependencies[it])[0]),

#pragma omp task default(none) firstprivate(comm, rank, tag_data, it_last_parent_group, last_child_index)              \
  shared(std::clog) depend(in : dep_parent[0], ptr_tree[0]) priority(prio)
                {
                    // I have to send a message from my right to update the multipoles of the first
                    // cells of the right ghosts.
                    // temporary buffer
                    auto pos = it_last_parent_group->get()->size() - 1;
                    auto const& cell = it_last_parent_group->get()->component(pos);   // the cell

                    auto const& m = cell.clocals();
                    auto size_local{int(nb_outputs * m.at(0).size())};
                    auto nb_m = m.size();
                    std::vector<value_type> buffer(size_local);

                    // std::cout << "cell index: " << cell.index() << " = parent " << (last_child_index >> dimension)
                    //           << "\n";
                    // loop to serialize the locals
                    auto it = std::begin(buffer);
                    for(std::size_t i{0}; i < nb_m; ++i)
                    {
                        auto const& ten = m.at(i);
                        std::copy(std::begin(ten), std::end(ten), it);
                        it += ten.size();
                    }
                    // io::print("buffer(send) ", buffer);

                    auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<value_type>();
                    // std::clog << "   send buffer to " << rank + 1 << std::endl;
                    comm->isend(buffer.data(), size_local, mpi_type, rank + 1, tag_data);
                    // std::cout << " downward(task_send)  buffer(rank=" << std::to_string(rank) << "): " << std::flush;
                    // for(int i = 0; i < buffer.size(); ++i)
                    // {
                    //     std::cout << "  " << buffer[i] << std::flush;
                    // }
                    // std::cout << std::endl << std::flush;
                }   // end task
            }   // end same parent
        }   // end rank !- proc -1
        // Receive
        if(rank > 0)
        {
            // std::clog << "Receive step\n";

            auto last_child_index_before_me = distrib[rank - 1][1] - 1;
            auto first_child_index = distrib[rank][0];
            // dependencies out on the group
            // check if same parent
            // std::clog << "downward receive comm  last_child_index_before_me " << last_child_index_before_me
            //           << " parent " << (last_child_index_before_me >> dimension) << " first_child_index "
            //           << first_child_index << " its parent " << (first_child_index >> dimension) << std::endl
            //           << std::flush;
            if((last_child_index_before_me >> dimension) == (first_child_index >> dimension))
            {
                // task to do
                // std::cout << " downward receive task to do perform " << std::endl << std::flush;

                // dependencies on left ghost parent
                auto gs = it_last_parent_group->get()->size();
                // std::cout << " gs = " << gs << std::endl << std::flush;
                int nb_grp_dep =
                  std::min(static_cast<int>(nb_children / gs + 1),
                           static_cast<int>(std::distance(it_last_parent_group, tree.end_cells(child_level))));
                auto it_last_parent_group = tree.begin_mine_cells(level) - 1;
                auto dep_ghost_parent = &(it_last_parent_group->get()->ccomponent(0).clocals(0));
                // std::cout << " downward(receive) dependencies(out): " << dep_ghost_parent << std::endl << std::flush;

#pragma omp task default(none) firstprivate(comm, rank, tag_data, size_local, it_last_parent_group) shared(std::clog)  \
  depend(out : dep_ghost_parent[0], ptr_tree[0]) priority(prio)
                {
                    // std::clog << "      Same parent\n ";
                    // Same parent, I have to receive a message from my left
                    // to update the locals of the last cells of the left ghosts.
                    std::vector<value_type> buffer(size_local);
                    // blocking receive ( We block the submission of L2L tasks at this level )
                    auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<value_type>();
                    comm->recv(buffer.data(), size_local, mpi_type, rank - 1, tag_data);
                    // std::cout << " downward(task receive) buffer:";
                    // for(int i = 0; i < buffer.size(); ++i)
                    // {
                    //     std::cout << "  " << buffer[i];
                    // }
                    // std::cout << std::endl;
                    /// set the locals in the last left ghosts and in last cell
                    auto it_group = it_last_parent_group;
                    auto pos = it_group->get()->size() - 1;   // index of the last cell in the group
                    auto& cell = it_group->get()->component(pos);
                    auto& m = cell.locals();
                    // std::clog << "cell index: " << cell.index() << " = parent " << (cell.index() >> dimension) << "\n";

                    auto nb_m = m.size();
                    // std::cout << " cell index: " << cell.index() << " level " << cell.csymbolics().level << "\n";
                    // io::print("buffer(recv) ", buffer);
                    auto it = std::begin(buffer);
                    for(std::size_t i{0}; i < nb_m; ++i)
                    {
                        auto& ten = m.at(i);
                        // std::cout << " ten before " << ten << std::endl;
                        std::copy(it, it + ten.size(), std::begin(ten));
                        // std::transform(it, it + ten.size(), std::begin(ten), std::begin(ten), std::plus<>{});
                        // std::cout << " ten after " << ten << std::endl;
                        it += ten.size();
                    }
                }   // end task
            }   // end same parent
        }   // end rank > 0
    }

    /**
    * @brief This function constructs the local approximation for all the cells of the tree by applying the operator l2l
    *
    * @tparam Tree
    * @tparam Approximation
    *
    * @param tree the target tree.
    * @param approximation the approximation to construct the local approximation.
    */
    template<typename Tree, typename Approximation>
    inline auto downward(Tree& tree, Approximation const& approximation) -> void
    {
        // upper working level is
        const auto top_height = tree.box().is_periodic() ? 0 : 2;
        const auto leaf_level = tree.leaf_level();

        for(std::size_t level = top_height; level < leaf_level; ++level)
        {
            // std::cout << "   L2L downward  : " << level << " -> " << level + 1 << std::endl << std::flush;
            // std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
            // update the ghost at the current level (father)
            downward_communications_level(level, tree);
            // std::cout << "   end downward comm  " << level << std::endl << std::flush;
            // std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;

            //  compute at the level
            omp::pass::downward_level(level, tree, approximation);
        }
    }
}   // namespace scalfmm::algorithms::mpi::pass

#endif   // SCALFMM_ALGORITHMS_MPI_DOWNWARD_HPP
