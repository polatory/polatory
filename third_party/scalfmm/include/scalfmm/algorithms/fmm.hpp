// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/fmm.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_FMM_HPP
#define SCALFMM_ALGORITHMS_FMM_HPP
#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/sequential/sequential.hpp"
#include "scalfmm/options/options.hpp"
#ifdef _OPENMP
#include "scalfmm/algorithms/omp/task_dep.hpp"
#endif

namespace scalfmm::algorithms
{
    namespace impl
    {
        /**
	* @brief fmm wrapper with a single tree.
	*
	* @tparam TreeType
	* @tparam NearFieldType
	* @tparam FarFieldType
	*
	* @param tree
        * @param fmmoperators
        * @param op
	*
	* @return
	*/
        template<typename TreeType, typename NearFieldType, typename FarFieldType>
        auto fmm(TreeType& tree, operators::fmm_operators<NearFieldType, FarFieldType> const& fmmoperators,
                 unsigned int op = operators_to_proceed::all) -> void
        {
            return sequential::sequential(tree, fmmoperators, op);
        }

        /**
	* @brief fmm wrapper with a source tree and a traget tree.
	*
	* @tparam SourceTargetTreeTypeype
	* @tparam TargetTreeType
	* @tparam FmmOperatorsType
	*
	* @param source_tree
	* @param target_tree
	* @param fmmoperators
	* @param op
	*
	* @return
	*/
        template<typename SourceTreeType, typename TargetTreeType, typename FmmOperatorsType>
        auto fmm(SourceTreeType& source_tree, TargetTreeType& target_tree, FmmOperatorsType const& fmmoperators,
                 unsigned int op = operators_to_proceed::all) -> void
        {
            return sequential::sequential(source_tree, target_tree, fmmoperators, op);
        }

        /**
	* @brief fmm wrapper with settings and a single tree.
	*
	* @tparam S
	* @tparam TreeType
	* @tparam FmmOperatorsType
	*
	* @param s
	* @param tree
	* @param fmmoperators
	* @param op
	*
	* @return
	*/
        template<typename... S, typename TreeType, typename FmmOperatorsType>
        inline auto fmm(options::settings<S...> s, TreeType& tree, FmmOperatorsType const& fmmoperators,
                        unsigned int op = operators_to_proceed::all) -> void
        {
            static_assert(options::support(s, options::_s(
#ifdef _OPENMP

                                                options::omp, options::omp_timit,
#endif
                                                options::seq, options::seq_timit)),
                          "unsupported fmm algo options!");
            if constexpr(options::has(s, options::seq, options::seq_timit))
            {
                using inner_settings = typename decltype(s)::inner_settings;
                return sequential::sequential[inner_settings{}](tree, fmmoperators, op);
            }
#ifdef _OPENMP
            else if constexpr(options::has(s, options::omp, options::omp_timit))
            {
                using inner_settings = typename decltype(s)::inner_settings;
                return omp::task_dep[inner_settings{}](tree, fmmoperators, op);
            }
#endif
            else if constexpr(options::has(s, options::timit))
            {
                return sequential::sequential[s](tree, fmmoperators, op);
            }
        }

        /**
	* @brief fmm wrapper with a source tree, a target tree and with settings.
	*
	* @tparam S
	* @tparam SourceTree
	* @tparam TargetTree
	* @tparam NearFieldType
	* @tparam FarFieldType
	*
	* @param s
	* @param tree_source
	* @param tree_target
	* @param fmmoperators
	* @param op
	*
	* @return
	*/
        template<typename... S, typename SourceTree, typename TargetTree, typename NearFieldType, typename FarFieldType>
        inline auto fmm(options::settings<S...> s, SourceTree& tree_source, TargetTree& tree_target,
                        operators::fmm_operators<NearFieldType, FarFieldType> const& fmmoperators,
                        unsigned int op = operators_to_proceed::all) -> void
        {
            static_assert(options::support(s, options::_s(
#ifdef _OPENMP
                                                options::omp, options::omp_timit,
#endif
                                                options::seq, options::seq_timit)),
                          "unsupported fmm algo options!");
            if constexpr(options::has(s, options::seq, options::seq_timit))
            {
                using inner_settings = typename decltype(s)::inner_settings;
                return sequential::sequential[inner_settings{}](tree_source, tree_target, fmmoperators, op);
            }
#ifdef _OPENMP
            else if constexpr(options::has(s, options::omp, options::omp_timit))
            {
                using inner_settings = typename decltype(s)::inner_settings;
                return omp::task_dep[inner_settings{}](tree_source, tree_target, fmmoperators, op);
            }
#endif
            else if constexpr(options::has(s, options::timit))
            {
                return sequential::sequential[s](tree_source, tree_target, fmmoperators, op);
            }
        }
    }   // namespace impl

    DECLARE_OPTIONED_CALLEE(fmm);
}   // namespace scalfmm::algorithms

#endif   // SCALFMM_ALGORITHMS_FMM_HPP
