// --------------------------------
// See LICENCE file at project root
// File : algorithm/omp/upward.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_MPI_UPWARD_HPP
#define SCALFMM_ALGORITHMS_MPI_UPWARD_HPP

#ifdef _OPENMP

#include <omp.h>

#include "scalfmm/algorithms/omp/upward.hpp"
// #include "scalfmm/operators/m2m.hpp"
// #include "scalfmm/operators/tags.hpp"
// #include "scalfmm/tree/utils.hpp"
// #include "scalfmm/utils/massert.hpp"
// #include "scalfmm/utils/math.hpp"

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
    void build_dependencies(IteratorType begin, IteratorType end, MortonType const& parent_morton_index,
                            Dependencies_t& dependencies)
    {
        for(auto grp_ptr = begin; grp_ptr != end; ++grp_ptr)
        {
            auto const& csymb = (*grp_ptr)->csymbolics();
            // iterate on the cells in the same group
            // we move forward in the index vector
            if(parent_morton_index == (csymb.starting_index >> dimension))
            {
                // std::cout << " add depend for grp with Int [" << csymb.starting_index << ", " << csymb.ending_index
                //           << "]" << std::endl;
                dependencies.push_back(&(grp_ptr->get()->ccomponent(0).cmultipoles(0)));
            }
            else
            {
                break;
            }
        }
    }
    /**
     * @brief Perform the communications for the children level 
     *
     *  Check if the parent of my first Morton index at child level is own by the 
     * previous process. if yes, we send the multipoles 
     * 
     * @tparam Tree
     * @tparam Approximation
     * @input[in] level the level to build the multipoles
     * @input[inout] tree to update the multipoles
     */
    template<typename Tree>
    inline auto start_communications(const int& level, Tree& tree) -> void
    {
        using value_type = typename Tree::base_type::cell_type::value_type;
        using dep_type = typename Tree::group_of_cell_type::symbolics_type::ptr_multi_dependency_type;

        static constexpr std::size_t dimension = Tree::base_type::box_type::dimension;
        static constexpr int nb_inputs = Tree::cell_type::storage_type::inputs_size;
        //
        // number of theoretical children
        constexpr int nb_children = math::pow(2, dimension);
        static constexpr auto prio{omp::priorities::max};
        //
        auto& para = tree.get_parallel_manager();
        auto& comm = para.get_communicator();
        auto ptr_comm = &comm;
        auto rank = comm.rank();
        int nb_proc = comm.size();
        int tag_nb = 1200 + 10 * level;
        int tag_data = 1201 + 10 * level;
        auto mpi_int_type = cpp_tools::parallel_manager::mpi::get_datatype<int>();
        // level where the multipoles are known
        auto level_child = level + 1;
        // iterator on the child cell groups
        auto it_first_group_of_child = tree.begin_mine_cells(level_child);
        // get size of multipoles
        auto it_first_cell_child = it_first_group_of_child->get()->begin();
        auto const& m1 = it_first_cell_child->cmultipoles();
        int size{int(nb_inputs * m1.at(0).size()) * nb_children};   //only nb_children -1 is needed in the worse case

        std::vector<dep_type> dependencies_out, dependencies_in;

        auto const& distrib = tree.get_cell_distribution(level);
        auto const& distrib_child = tree.get_cell_distribution(level_child);
        bool send_data{false}, receive_data{false};

        auto ptr_tree = &tree;

        if(rank > 0)
        {
            // If we send the multipoles, they must have been updated by the M2M of the previous level!
            // Check if you have to send something on the left (rank-1)
            // We are at level l and the children are at level l+1.
            // The ghost at the child level(on the right)
            //   must be updated if the parent index of the first child index is not equal to(lower than)
            //     the first index at level l.
            // serialization
            // check if we have some child to send (if there exists they starts at the first group)

            // first_index_child = morton index of the first child cell (at level = level_child)
            auto first_index_child = distrib_child[rank][0];
            // index_parent = first index of the cell at current level  on my process
            auto index_parent = distrib[rank][0];
            // parent_morton_index = last index of the cell on previous process at current level
            auto previous_parent_morton_index = distrib[rank - 1][1] - 1;
            // first_index_child = morton index of the first child cell (at level = level_child)
            auto last_index_child_previous = distrib_child[rank - 1][1] - 1;
            //
            send_data = (first_index_child >> dimension) == previous_parent_morton_index;
            if(send_data)
            {
                // std::cout << "upward send comm  first_index_child " << first_index_child << " parent "
                //           << (first_index_child >> dimension) << " previous child " << last_index_child_previous
                //           << " its parent " << previous_parent_morton_index << std::endl
                //           << std::flush;
                // construct the dependencies on the group of multipoles
                build_dependencies<dimension>(it_first_group_of_child, tree.end_mine_cells(level_child),
                                              previous_parent_morton_index, dependencies_in);
// io::print("upward(send) dependencies(in) ", dependencies_in);
// std::cout << std::flush;
// task to send the multipoles
#pragma omp task default(none) shared(std::cout, std::clog)                                                            \
  firstprivate(ptr_comm, rank, level, tag_data, tag_nb, mpi_int_type, size, dimension, nb_children,                    \
                 previous_parent_morton_index, first_index_child, it_first_group_of_child)                             \
  depend(iterator(std::size_t it = 0 : dependencies_in.size()), in : (dependencies_in[it])[0]),                        \
  depend(inout : ptr_tree[0]) priority(prio)
                {
                    // std::clog << "upward(task send(" << level << ")) start \n";
                    std::vector<value_type> buffer(size);
                    int count{0};

                    // std::clog << "upward(task send(" << level << ")) index " << first_index_child
                    //           << "  parent_of_first_index_child " << previous_parent_morton_index << std::endl
                    //           << std::flush;
                    // index is now the parent of the first child
                    auto index = first_index_child >> dimension;
                    // I have to send
                    // find the number of children to send (get pointer on multipoles !!)
                    // Construct an MPI datatype

                    // serialization
                    auto it_group = it_first_group_of_child;
                    //
                    auto it_cell = it_first_group_of_child->get()->begin();
                    auto next = scalfmm::component::generate_linear_iterator(1, it_group, it_cell);

                    auto it = std::begin(buffer);
                    //
                    // compute the number of cells to send and copy the multipoles in the buffer
                    for(int i = 0; i < nb_children - 1; ++i, next())
                    {
                        if(previous_parent_morton_index < (it_cell->index() >> dimension))
                        {
                            break;
                        }
                        // std::clog << "upward(task send) Check children  P " << index << " C " << it_cell->index()
                        //           << " level " << it_cell->csymbolics().level << std::endl
                        //           << std::flush;
                        // copy the multipoles in the buffer
                        auto const& m = it_cell->cmultipoles();
                        auto nb_m = m.size();
                        for(std::size_t i{0}; i < nb_m; ++i)
                        {
                            auto const& ten = m.at(i);
                            std::copy(std::begin(ten), std::end(ten), it);
                            it += ten.size();
                        }
                        ++count;
                    }

                    // std::clog << "upward(task send) nb_send = " << count << std::endl;

                    ptr_comm->isend(&count, 1, mpi_int_type, rank - 1, tag_nb);

                    {
                        // loop to serialize the multipoles
                        auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<value_type>();

                        ptr_comm->isend(buffer.data(), size, mpi_type, rank - 1, tag_data);
                        // std::cout << "upward(task send) buffer:";
                        // for(int i = 0; i < buffer.size(); ++i)
                        // {
                        //     std::cout << "  " << buffer[i];
                        // }
                        // std::cout << " \n" << std::flush;
                    }
                    // std::clog << "upward(task send(" << level << ")) end \n";
                }   // end task
            }
            // else
            // {
            //     std::cout << "upward no send comm first_index_child " << first_index_child << " previous child "
            //               << last_index_child_previous << std::endl
            //               << std::flush;
            // }
        }

        if(rank < nb_proc - 1)
        {
            // last_index_child = morton index of the last child cell (at level = level_child)
            auto last_index_child = distrib_child[rank][1] - 1;
            // parent_morton_index = last index of the cell on previous process at current level
            auto parent_morton_index = distrib[rank][1] - 1;
            // first_index_child_next = morton index of the first child cell (at level = level_child) on next mpi process
            auto first_index_child_next = distrib_child[rank + 1][0];
            receive_data = (last_index_child >> dimension) == (first_index_child_next >> dimension);
            if(receive_data)
            {
                // std::cout << "upward receive comm  last_index_child " << last_index_child << " parent "
                //           << (last_index_child >> dimension) << " next child " << first_index_child_next
                //           << " its parent " << (first_index_child_next >> dimension) << std::endl
                //           << std::flush;
                // dependencies_out contains the ghosts group with parent morton index
                auto it_first_ghost = tree.end_mine_cells(level_child);
                build_dependencies<dimension>(it_first_ghost, tree.end_cells(level_child), parent_morton_index,
                                              dependencies_out);
                // io::print(std::clog, "upward(receive) dependencies(out) ", dependencies_out);
                // io::print(std::cout, "upward(receive) dependencies(out) ", dependencies_out);
                // std::clog << std::flush;
// dependencies_out
#pragma omp task default(none) shared(std::cout, std::clog)                                                            \
  firstprivate(ptr_comm, rank, level, tag_data, tag_nb, mpi_int_type, size, dimension, it_first_ghost)                 \
  depend(iterator(std::size_t it = 0 : dependencies_out.size()), out : (dependencies_out[it])[0]),                     \
  depend(inout : ptr_tree[0]) priority(prio)
                {
                    // std::clog << "upward(task receive(" << level << ")) start \n";
                    int count{0};
                    std::vector<value_type> buffer(size);
                    ptr_comm->recv(&count, 1, mpi_int_type, rank + 1, tag_nb);
                    // std::clog << "upward(task receive(" << level << "))  " << count << " cell(s)\n" << std::flush;

                    auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<value_type>();

                    cpp_tools::parallel_manager::mpi::request recept_mpi_status(
                      ptr_comm->irecv(buffer.data(), size, mpi_type, rank + 1, tag_data));
                    cpp_tools::parallel_manager::mpi::request::waitall(1, &recept_mpi_status);

                    // ptr_comm->recv(buffer.data(), size, mpi_type, rank + 1, tag_data);

                    // std::cout << "upward(task receive)     buffer:";
                    // for(int i = 0; i < buffer.size(); ++i)
                    // {
                    //     std::cout << "  " << buffer[i];
                    // }
                    // std::cout << std::endl << std::flush;
                    // first right ghost
                    auto it_group = it_first_ghost;
                    auto it_cell = it_group->get()->begin();
                    // linear iterator on cells
                    auto next1 = scalfmm::component::generate_linear_iterator(1, it_group, it_cell);
                    auto it = std::begin(buffer);

                    for(int i = 0; i < count; ++i, next1())
                    {
                        // copy the multipoles in the buffer
                        auto& m = it_cell->multipoles();
                        auto nb_m = m.size();
                        // std::clog << "upward((" << level << ")) cell index: " << it_cell->index() << " level "
                        //           << it_cell->csymbolics().level << "\n"
                        //           << std::flush;
                        for(std::size_t i{0}; i < nb_m; ++i)
                        {
                            auto& ten = m.at(i);
                            // std::cout << "upward()    ten(before) " << ten << std::endl << std::flush;
                            std::copy(it, it + ten.size(), std::begin(ten));
                            it += ten.size();
                            // std::cout << "upward()    ten(after) " << ten << std::endl << std::flush;
                        }
                    }
                    // std::clog << "upward(task receive(" << level << ")) end \n";
                }   // end task
            }
            // else
            // {
            //     std::cout << "no receive comm last_index_child " << last_index_child << " next child "
            //               << first_index_child_next << std::endl
            //               << std::flush;
            // }
        }
    }

    template<typename Tree>
    void prepare_comm_up(Tree tree, const int level)
    {
        using value_type = typename Tree::base_type::cell_type::value_type;
        using dep_type = typename Tree::group_of_cell_type::symbolics_type::ptr_multi_dependency_type;

        static constexpr std::size_t dimension = Tree::base_type::box_type::dimension;
        static constexpr int nb_inputs = Tree::cell_type::storage_type::inputs_size;
        //
        // number of theoretical children
        constexpr int nb_children = math::pow(2, dimension);
        static constexpr auto prio{omp::priorities::max};
        //
        auto& para = tree.get_parallel_manager();
        auto& comm = para.get_communicator();
        auto ptr_comm = &comm;
        auto rank = comm.rank();
        int nb_proc = comm.size();
        int tag_nb = 1200 + 10 * level;
        int tag_data = 1201 + 10 * level;
        auto mpi_int_type = cpp_tools::parallel_manager::mpi::get_datatype<int>();
        // level where the multipoles are known
        auto level_child = level + 1;
        // iterator on the child cell groups
        auto it_first_group_of_child = tree.begin_mine_cells(level_child);
        // get size of multipoles
        auto it_first_cell_child = it_first_group_of_child->get()->begin();
        auto const& m1 = it_first_cell_child->cmultipoles();
        int size{int(nb_inputs * m1.at(0).size()) * nb_children};   //only nb_children -1 is needed in the worse case

        auto msg = tree.get_up_down_access(level);

        auto const& distrib = tree.get_cell_distribution(level);
        auto const& distrib_child = tree.get_cell_distribution(level_child);
        bool send_data{false}, receive_data{false};

        auto ptr_tree = &tree;

        if(rank > 0)
        {
            // If we send the multipoles, they must have been updated by the M2M of the previous level!
            // Check if you have to send something on the left (rank-1)
            // We are at level l and the children are at level l+1.
            // The ghost at the child level(on the right)
            //   must be updated if the parent index of the first child index is not equal to(lower than)
            //     the first index at level l.
            // serialization
            // check if we have some child to send (if there exists they starts at the first group)

            // first_index_child = morton index of the first child cell (at level = level_child)
            auto first_index_child = distrib_child[rank][0];
            // index_parent = first index of the cell at current level  on my process
            auto index_parent = distrib[rank][0];
            // parent_morton_index = last index of the cell on previous process at current level
            auto previous_parent_morton_index = distrib[rank - 1][1] - 1;
            // first_index_child = morton index of the first child cell (at level = level_child)
            auto last_index_child_previous = distrib_child[rank - 1][1] - 1;
            //
            send_data = (first_index_child >> dimension) == previous_parent_morton_index;

            if(send_data)
            {
                std::clog << "upward(task send(" << level << ")) start \n";
                int count{0};
                std::clog << "upward(task send(" << level << ")) index " << first_index_child
                          << "  parent_of_first_index_child " << previous_parent_morton_index << std::endl
                          << std::flush;
                // index is now the parent of the first child
                auto index = first_index_child >> dimension;
                // I have to send
                // find the number of children to send (get pointer on multipoles !!)
                // Construct an MPI datatype

                // serialization
                auto it_group = it_first_group_of_child;
                //
                auto it_cell = it_first_group_of_child->get()->begin();
                auto next = scalfmm::component::generate_linear_iterator(1, it_group, it_cell);
                //
                // compute the number of cells to send and copy the multipoles in the buffer
                for(int i = 0; i < nb_children - 1; ++i, next())
                {
                    if(previous_parent_morton_index < (it_cell->index() >> dimension))
                    {
                        break;
                    }
                    std::clog << "upward(task send) Check children  P " << index << " C " << it_cell->index()
                              << " level " << it_cell->csymbolics().level << std::endl
                              << std::flush;
                    ++count;
                }
                std::clog << "upward(task send) nb_send = " << count << std::endl;
                ptr_comm->isend(&count, 1, mpi_int_type, rank - 1, tag_nb);

                msg.set_nb_child_to_send(count);
            }
        }
        if(rank < nb_proc - 1)
        {
            // last_index_child = morton index of the last child cell (at level = level_child)
            auto last_index_child = distrib_child[rank][1] - 1;
            // parent_morton_index = last index of the cell on previous process at current level
            auto parent_morton_index = distrib[rank][1] - 1;
            // first_index_child_next = morton index of the first child cell (at level = level_child) on next mpi process
            auto first_index_child_next = distrib_child[rank + 1][0];
            receive_data = (last_index_child >> dimension) == (first_index_child_next >> dimension);
            if(receive_data)
            {
                // std::cout << "upward receive comm  last_index_child " << last_index_child << " parent "
                //           << (last_index_child >> dimension) << " next child " << first_index_child_next
                //           << " its parent " << (first_index_child_next >> dimension) << std::endl
                //           << std::flush;

                std::clog << std::flush;
                {
                    std::clog << "upward(task receive(" << level << ")) start \n";
                    int count{0};
                    std::vector<value_type> buffer(size);
                    ptr_comm->recv(&count, 1, mpi_int_type, rank + 1, tag_nb);
                    std::clog << "upward(task receive(" << level << "))  " << count << " cell(s)\n" << std::flush;
                    msg.set_nb_child_to_receive(count);
                }
            }
        }
    };
    /// @brief This function constructs the local approximation for all the cells of the tree by applying the operator
    /// m2m
    ///
    /// @param tree   the tree target
    /// @param approximation the approximation to construct the local approximation
    ///
    template<typename Tree, typename Approximation>
    inline auto upward(Tree& tree, Approximation const& approximation) -> void
    {
        auto leaf_level = tree.height() - 1;

        // upper working level is
        const int top_height = tree.box().is_periodic() ? 0 : 2;
        // const int start_duplicated_level = tree.start_duplicated_level();
        //
        // int top = start_duplicated_level < 0 ? top_height : start_duplicated_level - 1;
        int top = top_height;
        for(int level = leaf_level - 1; level >= top /*top_height*/; --level)   // int because top_height could be 0
        {
            // std::cout << "M2M : " << level + 1 << " -> " << level << std::endl << std::flush;
            //
            start_communications(level, tree);
            // std::cout << "  end comm " << std::endl << std::flush;

            omp::pass::upward_level(level, tree, approximation);
            // std::cout << "  end upward_level " << level << std::endl << std::flush;
        }
        // std::cout << "end upward " << std::endl << std::flush;

        //

        // for(int level = start_duplicated_level; level >= top_height; --level)   // int because top_height could be 0
        // {
        //     std::cout << "Level duplicated (seq): " << level << std::endl;
        //     upward_level(level, tree, approximation);
        // }
    }
}   // namespace scalfmm::algorithms::mpi::pass

#endif   // _OPENMP
#endif   // SCALFMM_ALGORITHMS_MPI_UPWARD_HPP
