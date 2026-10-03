// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/mpi/proc_task.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_MPI_PROC_TASK_HPP
#define SCALFMM_ALGORITHMS_MPI_PROC_TASK_HPP

#ifdef _OPENMP

#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/omp/cell_to_leaf.hpp"
#include "scalfmm/algorithms/omp/leaf_to_cell.hpp"
#include "scalfmm/algorithms/omp/priorities.hpp"
#include "scalfmm/algorithms/omp/utils.hpp"

#include "scalfmm/algorithms/mpi/direct.hpp"
#include "scalfmm/algorithms/mpi/downward.hpp"
#include "scalfmm/algorithms/mpi/transfer.hpp"
#include "scalfmm/algorithms/mpi/upward.hpp"
#include "scalfmm/lists/lists.hpp"

#include <cpp_tools/parallel_manager/parallel_manager.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

#include <chrono>
#include <iostream>
#include <omp.h>

namespace scalfmm::algorithms::mpi
{
    namespace impl
    {

        /**
         * @brief  Openmp task algorithm to compute the interactions
         *
         * @tparam S        type of the options
         * @tparam TreeS    type of the tree source
         * @tparam TreeT    type of the tree target
         * @tparam NearField  Near-field type
         * @tparam FarField Far-field type
         *
         * @param s          the option (omp, seq, ...)
         * @param tree_source  the tree of particle sources
         * @param tree_target   the tree of particle sources
         * @param fmmoperators  the fmm operator
         * @param op_in          the type of computation (nearfield, farfield, all) see @see operators_to_proceed
         */
        template<typename... S, typename TreeS, typename TreeT, typename NearField, typename FarField>
        inline auto proc_task(options::settings<S...> s, TreeS& tree_source, TreeT& tree_target,
                              operators::fmm_operators<NearField, FarField> const& fmmoperators,
                              unsigned int op_in = operators_to_proceed::all) -> void
        {
            static_assert(options::support(s, options::_s(options::timit)), "task_dep algo unsupported options!");
            static_assert(std::is_same_v<TreeS, TreeT>, "task_dep algo unsupported source  != target !");
            cpp_tools::timers::timer<std::chrono::milliseconds> time{};
            bool same_tree{false};
            if constexpr(std::is_same_v<TreeS, TreeT>)
            {
                same_tree = (&tree_source == &tree_target);
            }
            //
            auto const& far_field = fmmoperators.far_field();
            auto const& near_field = fmmoperators.near_field();
            auto const& separation_criterion = fmmoperators.near_field().separation_criterion();
            auto const& mutual = fmmoperators.near_field().mutual();
            // const priorities prios(tree.height());
            //  Buffer for m2l optimization : we calculate the inverse fft on the fly and free memory.
            //  get the shape of the buffer used in the m2l pass (get it from the far field)
            //  and allocate it
            if(omp_get_max_task_priority() < omp::priorities::max)
            {
                std::cout << cpp_tools::colors::red
                          << "WARNING the task priorities are not (fully) available. set OMP_MAX_TASK_PRIORITY to "
                          << omp::priorities::max + 1 << cpp_tools::colors::reset << std::endl;
            }
            tree_target.get_parallel_manager().get_communicator().barrier();
            const auto op = tree_target.height() == 2 ? operators_to_proceed::p2p : op_in;

            if constexpr(options::has(s, options::timit))
            {
                time.tic();
            }
            auto buffers{scalfmm::algorithms::omp::impl::init_buffers(far_field.approximation())};
            //
#pragma omp parallel default(none) shared(tree_source, tree_target, far_field, near_field, buffers, op,                \
                                            separation_criterion, mutual, same_tree, std::cout)
            {
#pragma omp single nowait
                {
                    if(tree_target.is_interaction_p2p_lists_built() == false)
                    {
                        list::omp::build_p2p_interaction_list(tree_source, tree_target, separation_criterion, mutual);
                    }
                    if((op & operators_to_proceed::p2p) == operators_to_proceed::p2p)
                    {
                        // std::cout << "pass::direct \n";
                        pass::direct(tree_source, tree_target, near_field);
                    }

                    if(tree_target.is_interaction_m2l_lists_built() == false)
                    {
                        list::omp::build_m2l_interaction_list(tree_source, tree_target, separation_criterion,
                                                              separation_criterion);
                    }

                    if((op & operators_to_proceed::p2m) == operators_to_proceed::p2m)
                    {
                        // std::cout << "pass::leaf_to_cell \n";
                        scalfmm::algorithms::omp::pass::leaf_to_cell(tree_source, far_field);
                    }
                    if((op & operators_to_proceed::m2m) == operators_to_proceed::m2m)
                    {
                        // std::cout << "pass::upward \n";
                        pass::upward(tree_source, far_field.approximation());
                    }
                    // if(same_tree && tree_target.box().is_periodic())
                    // {
                    //     //  timers["field0"].tic();
                    //     pass::build_field_level0(tree_target, far_field.approximation());
                    //     //  timers["field0"].tac();
                    // }
                    if((op & operators_to_proceed::m2l) == operators_to_proceed::m2l)
                    {
                        // std::cout << "pass::transfer   remove_leaf_level \n";
                        pass::transfer(tree_source, tree_target, far_field, buffers,
                                       scalfmm::algorithms::omp::pass::split_m2l::remove_leaf_level);
                    }
                    if((op & operators_to_proceed::l2l) == operators_to_proceed::l2l)
                    {
                        // std::cout << "pass::downward\n";

                        pass::downward(tree_target, far_field.approximation());
                    }

                    if((op & operators_to_proceed::m2l) == operators_to_proceed::m2l)
                    {
                        // std::cout << "pass::transfer   leaf_level \n";

                        pass::transfer(tree_source, tree_target, far_field, buffers,
                                       scalfmm::algorithms::omp::pass::split_m2l::leaf_level);
                    }
                    if((op & operators_to_proceed::l2p) == operators_to_proceed::l2p)
                    {
                        scalfmm::algorithms::omp::pass::cell_to_leaf(tree_target, far_field);
                    }
                }
            }   // end parallel

#ifdef _DEBUG_BLOCK_DATA
            std::clog << "\n";
            std::clog << "  FINAl block (end proc_task)\n";
            int id_group{0};
            auto group_of_leaves = tree_source.vector_of_leaf_groups();
            for(auto pg: group_of_leaves)
            {
                auto const& current_group_symbolics = pg->csymbolics();
                std::clog << "*** Group of leaf index " << ++id_group << " *** index in ["
                          << current_group_symbolics.starting_index << ", " << current_group_symbolics.ending_index
                          << "[";
                std::clog << ", is_mine: " << std::boolalpha << current_group_symbolics.is_mine << "\n";
                std::clog << "    group size:  " << current_group_symbolics.number_of_component_in_group << ", ";
                std::clog << "global index =  " << current_group_symbolics.idx_global << " \n" << std::flush;
                // std::cout << "    index: ";
                // std::clog << "block index " << tt++ << std::endl;
                pg->cstorage().print_block_data(std::clog);
            }
            std::clog << "  ---------------------------------------------------\n";
#endif
            scalfmm::algorithms::omp::impl::delete_buffers(buffers);

            if constexpr(options::has(s, options::timit))
            {
                time.tac();
                std::cout << "Full algo : " << time.cumulated() << " s\n";
            }
        }

        template<typename Tree, typename NearField, typename FarField>
        inline auto proc_task(Tree& tree, operators::fmm_operators<NearField, FarField> const& fmmoperators,
                              unsigned int op = operators_to_proceed::all) -> void
        {
            return proc_task(options::settings<>{}, tree, tree, fmmoperators, op);
        }
        template<typename... S, typename Tree, typename NearField, typename FarField>
        inline auto proc_task(options::settings<S...> s, Tree& tree,
                              operators::fmm_operators<NearField, FarField> const& fmmoperators,
                              unsigned int op = operators_to_proceed::all) -> void
        {
            return proc_task(s, tree, tree, fmmoperators, op);
        }
        template<typename TreeS, typename TreeT, typename NearField, typename FarField>
        inline auto proc_task(TreeS& tree_source, TreeT& tree_target,
                              operators::fmm_operators<NearField, FarField> const& fmmoperators,
                              unsigned int op = operators_to_proceed::all) -> void
        {
            return proc_task(options::settings<>{}, tree_source, tree_target, fmmoperators, op);
        }

    }   // namespace impl

    DECLARE_OPTIONED_CALLEE(proc_task);

}   // namespace scalfmm::algorithms::mpi

#endif   // _OPENMP
#endif   // SCALFMM_ALGORITHMS_MPI_PROC_TASK_HPP
