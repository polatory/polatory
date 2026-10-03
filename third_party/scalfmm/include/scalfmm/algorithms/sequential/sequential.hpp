// --------------------------------
// See LICENCE file at project root
// File : algorithm/sequential/sequential.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_SEQUENTIAL_SEQUENTIAL_HPP
#define SCALFMM_ALGORITHMS_SEQUENTIAL_SEQUENTIAL_HPP

#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/sequential/cell_to_leaf.hpp"
#include "scalfmm/algorithms/sequential/direct.hpp"
#include "scalfmm/algorithms/sequential/downward.hpp"
#include "scalfmm/algorithms/sequential/leaf_to_cell.hpp"
#include "scalfmm/algorithms/sequential/periodic.hpp"
#include "scalfmm/algorithms/sequential/transfer.hpp"
#include "scalfmm/algorithms/sequential/upward.hpp"
#include "scalfmm/lists/sequential.hpp"
#include "scalfmm/tools/bench.hpp"
#include "scalfmm/utils/massert.hpp"

#include <cpp_tools/timers/simple_timer.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

namespace scalfmm::algorithms::sequential
{
    namespace impl
    {
        /**
         * @brief Sequential algorithm.
         *
         * This function launches all the passes of the algorithm.
         *
         * @tparam SourceTreeType
         * @tparam TargetTreeType
         * @tparam NearFieldType
         * @tparam FarFieldType
         * @tparam S
         *
         * @param s
         * @param source_tree
         * @param target_tree
         * @param fmmoperators
         * @param op_in
         */
        template<typename SourceTreeType, typename TargetTreeType, typename NearFieldType, typename FarFieldType,
                 typename... S>
        inline auto sequential(options::settings<S...> s, SourceTreeType& source_tree, TargetTreeType& target_tree,
                               operators::fmm_operators<NearFieldType, FarFieldType> const& fmmoperators,
                               unsigned int op_in = operators_to_proceed::all) -> void
        {
            static_assert(options::support(s, options::_s(options::timit)), "sequential algo unsupported options!");

            bool same_tree{false};
            if constexpr(std::is_same_v<SourceTreeType, TargetTreeType>)
            {
                same_tree = (&source_tree == &target_tree);
            }

            // timer for the different passes
            using timer_type = cpp_tools::timers::timer<std::chrono::nanoseconds>;
            std::unordered_map<std::string, timer_type> timers = {
              {"p2m", timer_type()},      {"m2m", timer_type()},      {"m2l", timer_type()},
              {"l2l", timer_type()},      {"l2p", timer_type()},      {"p2p", timer_type()},
              {"m2l-list", timer_type()}, {"p2p-list", timer_type()}, {"field0", timer_type()}};

            auto const& near_field = fmmoperators.near_field();
            auto const& far_field = fmmoperators.far_field();
            auto const& approximation = far_field.approximation();
            auto const& near_field_separation_criterion = near_field.separation_criterion();
            auto const& far_field_separation_criterion = far_field.separation_criterion();
            auto const& mutual = near_field.mutual();

            assertm(near_field_separation_criterion == far_field_separation_criterion,
                    "Far field kernel and near field kernel must have the same separation criterion.");

            // Build M2L interaction list (if needed)
            if(target_tree.is_interaction_m2l_lists_built() == false)
            {
                if constexpr(options::has(s, options::timit))
                {
                    timers["m2l-list"].tic();
                }

                scalfmm::list::sequential::build_m2l_interaction_list(
                  source_tree, target_tree, far_field_separation_criterion, far_field_separation_criterion);

                if constexpr(options::has(s, options::timit))
                {
                    timers["m2l-list"].tac();
                }
            }

            const auto op = target_tree.height() == 2 ? op_in & operators_to_proceed::p2p : op_in;

            // P2M pass
            if((op & operators_to_proceed::p2m) == operators_to_proceed::p2m)
            {
                if constexpr(options::has(s, options::timit))
                {
                    timers["p2m"].tic();
                }
                pass::leaf_to_cell(source_tree, far_field);
                if constexpr(options::has(s, options::timit))
                {
                    timers["p2m"].tac();
                }
            }

            // M2M pass
            if((op & operators_to_proceed::m2m) == operators_to_proceed::m2m)
            {
                if constexpr(options::has(s, options::timit))
                {
                    timers["m2m"].tic();
                }
                pass::upward(source_tree, approximation);
                if constexpr(options::has(s, options::timit))
                {
                    timers["m2m"].tac();
                }
            }

            // Upper levels (for the periodic case)
            if(same_tree && source_tree.box().is_periodic())
            {
                if constexpr(options::has(s, options::timit))
                {
                    timers["field0"].tic();
                }
                pass::build_field_level0(source_tree, target_tree, approximation);
                if constexpr(options::has(s, options::timit))
                {
                    timers["field0"].tac();
                }
            }

            // M2L pass
            if((op & operators_to_proceed::m2l) == operators_to_proceed::m2l)
            {
                if constexpr(options::has(s, options::timit))
                {
                    timers["m2l"].tic();
                }
                pass::transfer(source_tree, target_tree, far_field);
                if constexpr(options::has(s, options::timit))
                {
                    timers["m2l"].tac();
                }
            }

            // L2L pass
            if((op & operators_to_proceed::l2l) == operators_to_proceed::l2l)
            {
                if constexpr(options::has(s, options::timit))
                {
                    timers["l2l"].tic();
                }
                pass::downward(target_tree, approximation);
                if constexpr(options::has(s, options::timit))
                {
                    timers["l2l"].tac();
                }
            }

            // L2P pass
            if((op & operators_to_proceed::l2p) == operators_to_proceed::l2p)
            {
                if constexpr(options::has(s, options::timit))
                {
                    timers["l2p"].tic();
                }
                pass::cell_to_leaf(target_tree, far_field);
                if constexpr(options::has(s, options::timit))
                {
                    timers["l2p"].tac();
                }
            }

            // P2P pass
            if((op & operators_to_proceed::p2p) == operators_to_proceed::p2p)
            {
                if(target_tree.is_interaction_p2p_lists_built() == false)
                {
                    if constexpr(options::has(s, options::timit))
                    {
                        timers["p2p-list"].tic();
                    }
                    scalfmm::list::sequential::build_p2p_interaction_list(source_tree, target_tree,
                                                                          near_field_separation_criterion, mutual);
                    if constexpr(options::has(s, options::timit))
                    {
                        timers["p2p-list"].tac();
                    }
                }

                if constexpr(options::has(s, options::timit))
                {
                    timers["p2p"].tic();
                }
                pass::direct(source_tree, target_tree, near_field);
                if constexpr(options::has(s, options::timit))
                {
                    timers["p2p"].tac();
                }
            }
#ifdef _DEBUG_BLOCK_DATA
            std::clog << "\n  blocks after direct\n";
            int tt{0};
            auto group_of_leaves = tree_target.vector_of_leaf_groups();

            for(auto pg: group_of_leaves)
            {
                std::clog << "block index " << tt++ << std::endl;
                pg->cstorage().print_block_data(std::clog);
            }
            std::clog << "  ---------------------------------------------------\n";
#endif

            // print time of each pass
            if constexpr(options::has(s, options::timit))
            {
                auto [fartime, neartime, overall, ratio] = bench::print(timers);
                // bench::dump_csv( "timings.csv"
                //                , "groupsize,legende,time,type"
                //                , std::to_string(tree.group_of_leaf_size())
                //                , std::string("far")
                //                , std::to_string(fartime)
                //                , std::string("experimental")
                //                );
                /**/
            }
        }

        /**
	* @brief Wrapper to run the sequential algorithm both with a source tree and a target tree
	* but without settings.
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
	* @return the sequential algorithm.
	*/
        template<typename SourceTreeType, typename TargetTreeType, typename NearFieldType, typename FarFieldType>
        inline auto sequential(SourceTreeType& source_tree, TargetTreeType& target_tree,
                               operators::fmm_operators<NearFieldType, FarFieldType> const& fmmoperators,
                               unsigned int op = operators_to_proceed::all) -> void
        {
            return sequential(options::settings<>{}, source_tree, target_tree, fmmoperators, op);
        }

        /**
	* @brief Wrapper to run the sequential algorithm with a single tree and without settings.
	*
	* @tparam TreeType
	* @tparam NearFieldType
	* @tparam FarFieldType
	*
	* @param tree
	* @param fmmoperators
	* @param op
	*
	* @return the sequential algorithm.
	*/
        template<typename TreeType, typename NearFieldType, typename FarFieldType>
        inline auto sequential(TreeType& tree,
                               operators::fmm_operators<NearFieldType, FarFieldType> const& fmmoperators,
                               unsigned int op = operators_to_proceed::all) -> void
        {
            return sequential(options::settings<>{}, tree, tree, fmmoperators, op);
        }

        /**
	* @brief Wrapper to run the sequential algorithm with a single tree and with settings.
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
	* @return the sequential algorithm.
	*/
        template<typename... S, typename TreeType, typename NearFieldType, typename FarFieldType>
        inline auto sequential(options::settings<S...> s, TreeType& tree,
                               operators::fmm_operators<NearFieldType, FarFieldType> const& fmmoperators,
                               unsigned int op = operators_to_proceed::all) -> void
        {
            return sequential(s, tree, tree, fmmoperators, op);
        }

    }   // namespace impl

    DECLARE_OPTIONED_CALLEE(sequential);

}   // namespace scalfmm::algorithms::sequential

#endif   // SCALFMM_TREE_TREE_HPP
