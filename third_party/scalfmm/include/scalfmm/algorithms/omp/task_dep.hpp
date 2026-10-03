// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/omp/task_dep.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_TASK_DEP_HPP
#define SCALFMM_ALGORITHMS_OMP_TASK_DEP_HPP

#ifdef _OPENMP

#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/omp/cell_to_leaf.hpp"
#include "scalfmm/algorithms/omp/direct.hpp"
#include "scalfmm/algorithms/omp/downward.hpp"
#include "scalfmm/algorithms/omp/leaf_to_cell.hpp"
#include "scalfmm/algorithms/omp/periodic.hpp"
#include "scalfmm/algorithms/omp/transfer.hpp"
#include "scalfmm/algorithms/omp/upward.hpp"
#include "scalfmm/algorithms/omp/utils.hpp"
#include "scalfmm/lists/omp.hpp"

#include <cpp_tools/timers/simple_timer.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>

namespace scalfmm::algorithms::omp
{
    namespace impl
    {
        /**
         * @brief OpenMP task-based algorithm.
	 *
	 * This function launches all the passes of the task-based algorithm.
         *
         * @tparam S        type of the options
         * @tparam SourceTreeType    type of the tree source
         * @tparam TargetTreeType    type of the tree target
         * @tparam NearFieldType  Near-field type
         * @tparam FarFieldType Far-field type
	 *
         * @param s          the option (omp, seq, ...)
         * @param source_tree  the tree of particle sources
         * @param target_tree   the tree of particle sources
         * @param fmmoperators  the fmm operator
         * @param op_in          the type of computation (nearfield, farfield, all) see @see operators_to_proceed
         */
        template<typename... S, typename SourceTreeType, typename TargetTreeType, typename NearFieldType,
                 typename FarFieldType>
        inline auto task_dep(options::settings<S...> s, SourceTreeType& source_tree, TargetTreeType& target_tree,
                             operators::fmm_operators<NearFieldType, FarFieldType> const& fmmoperators,
                             unsigned int op_in = operators_to_proceed::all) -> void
        {
            static_assert(options::support(s, options::_s(options::timit)), "task_dep algo unsupported options!");
            using timer = cpp_tools::timers::timer<std::chrono::milliseconds>;
            timer time{};
            bool same_tree{false};
            if constexpr(std::is_same_v<SourceTreeType, TargetTreeType>)
            {
                same_tree = (&source_tree == &target_tree);
            }

            auto const& near_field = fmmoperators.near_field();
            auto const& far_field = fmmoperators.far_field();
            auto const& approximation = far_field.approximation();
            auto const& near_field_separation_criterion = near_field.separation_criterion();
            auto const& far_field_separation_criterion = far_field.separation_criterion();
            auto const& mutual = near_field.mutual();

            assertm(near_field_separation_criterion == far_field_separation_criterion,
                    "Far field kernel and near field kernel must have the same separation criterion.");

            //  Buffer for m2l optimization : we calculate the inverse fft on the fly and free memory.
            //  get the shape of the buffer used in the m2l pass (get it from the far field)
            //  and allocate it
            // if(omp_get_max_task_priority() < priorities::max)
            // {
            //     std::cout << cpp_tools::colors::red
            //               << "WARNING the task priorities are not (fully) available. set OMP_MAX_TASK_PRIORITY to "
            //               << priorities::max + 1 << cpp_tools::colors::reset << std::endl;
            // }

            const auto op = target_tree.height() == 2 ? op_in & operators_to_proceed::p2p : op_in;

            if constexpr(options::has(s, options::timit))
            {
                time.tic();
            }
            auto buffers{init_buffers(far_field.approximation())};
            //
#pragma omp parallel default(none)                                                                                     \
  shared(source_tree, target_tree, far_field, approximation, near_field, buffers, op, near_field_separation_criterion, \
           far_field_separation_criterion, mutual, same_tree)
            {
#pragma omp single nowait
                {
                    if(target_tree.is_interaction_p2p_lists_built() == false)
                    {
                        list::omp::build_p2p_interaction_list(source_tree, target_tree, near_field_separation_criterion,
                                                              mutual);
                    }
                    if((op & operators_to_proceed::p2p) == operators_to_proceed::p2p)
                    {
                        pass::direct(source_tree, target_tree, near_field);
                    }
                    if(target_tree.is_interaction_m2l_lists_built() == false)
                    {
                        list::omp::build_m2l_interaction_list(source_tree, target_tree, far_field_separation_criterion,
                                                              far_field_separation_criterion);
                    }
                    if((op & operators_to_proceed::p2m) == operators_to_proceed::p2m)
                    {
                        pass::leaf_to_cell(source_tree, far_field);
                    }

                    if((op & operators_to_proceed::m2m) == operators_to_proceed::m2m)
                    {
                        pass::upward(source_tree, approximation);
                    }

                    if(same_tree && target_tree.box().is_periodic())
                    {
                        pass::build_field_level0(target_tree, far_field.approximation());
                    }
                    if((op & operators_to_proceed::m2l) == operators_to_proceed::m2l)
                    {
                        pass::transfer(source_tree, target_tree, far_field, buffers,
                                       pass::split_m2l::remove_leaf_level);
                    }
                    if((op & operators_to_proceed::l2l) == operators_to_proceed::l2l)
                    {
                        pass::downward(target_tree, approximation);
                    }
                    if((op & operators_to_proceed::m2l) == operators_to_proceed::m2l)
                    {
                        pass::transfer(source_tree, target_tree, far_field, buffers, pass::split_m2l::leaf_level);
                    }
                    if((op & operators_to_proceed::l2p) == operators_to_proceed::l2p)
                    {
                        pass::cell_to_leaf(target_tree, far_field);
                    }
                }   // end single
            }   // end parallel

            delete_buffers(buffers);

            if constexpr(options::has(s, options::timit))
            {
                time.tac();
                using duration_type = timer::duration;
                using value_type = double;
                static constexpr value_type unit_multiplier = static_cast<value_type>(duration_type::period::den);
                std::cout << "Full algo : " << time.cumulated() / unit_multiplier << " s \n";
            }
        }

        /**
         * @brief Wrapper to run the OpenMP task-based algorithm both with a source tree and a target tree but
	 * without settings.
         *
         * @tparam SourceTreeType
         * @tparam TargetTreeType
         * @tparam NearFieldType
         * @tparam FarFieldType
	 *
         * @param source_tree
         * @param target_tree
         * @param fmmoperators
         * @param op
	 *
         * @return the task-based algorithm.
         */
        template<typename SourceTreeType, typename TargetTreeType, typename NearFieldType, typename FarFieldType>
        inline auto task_dep(SourceTreeType& source_tree, TargetTreeType& target_tree,
                             operators::fmm_operators<NearFieldType, FarFieldType> const& fmmoperators,
                             unsigned int op = operators_to_proceed::all) -> void
        {
            return task_dep(options::settings<>{}, source_tree, target_tree, fmmoperators, op);
        }

        /**
         * @brief Wrapper to run the OpenMP task-based algorithm both with a source tree and a target tree but
	 * without settings.
         *
         * @tparam TreeType
         * @tparam NearFieldType
         * @tparam FarFieldType
	 *
         * @param tree
         * @param fmmoperators
         * @param op
         *
         * @return the task-based algorithm.
         */
        template<typename TreeType, typename NearFieldType, typename FarFieldType>
        inline auto task_dep(TreeType& tree, operators::fmm_operators<NearFieldType, FarFieldType> const& fmmoperators,
                             unsigned int op = operators_to_proceed::all) -> void
        {
            return task_dep(options::settings<>{}, tree, tree, fmmoperators, op);
        }

        /**
         * @brief Wrapper to run the OpenMP task-based algorithm with a single tree and with settings.
         *
         * @tparam S
         * @tparam TreeType
         * @tparam NearFieldType
         * @tparam FarFieldType
	 *
         * @param s
         * @param tree
         * @param fmmoperators
         * @param op
	 *
         * @return the task-based algorithm.
         */
        template<typename... S, typename TreeType, typename NearFieldType, typename FarFieldType>
        inline auto task_dep(options::settings<S...> s, TreeType& tree,
                             operators::fmm_operators<NearFieldType, FarFieldType> const& fmmoperators,
                             unsigned int op = operators_to_proceed::all) -> void
        {
            return task_dep(s, tree, tree, fmmoperators, op);
        }

    }   // namespace impl

    DECLARE_OPTIONED_CALLEE(task_dep);

}   // namespace scalfmm::algorithms::omp

#endif   // _OPENMP
#endif   // SCALFMM_ALGORITHMS_OMP_TASK_DEP_HPP
