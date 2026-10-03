// --------------------------------
// See LICENCE file at project root
// File : algorithms/omp/utils.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OPENMP_UTILS_HPP
#define SCALFMM_ALGORITHMS_OPENMP_UTILS_HPP

namespace scalfmm::algorithms::omp
{
    namespace impl
    {
        /**
        * @brief
        *
        * @tparam ApproximationType
	*
        * @param approximation
        *
        * @return
	*/
        template<typename ApproximationType>
        auto init_buffers(ApproximationType const& approximation)
          -> std::vector<std::decay_t<decltype(std::declval<ApproximationType>().buffer_initialization())>*>
        {
            using approximation_type = ApproximationType;
            using buffer_type = typename approximation_type::buffer_type;
            using ptr_buffer_type = buffer_type*;

            std::vector<ptr_buffer_type> buffers{};

#pragma omp parallel
            {
#pragma omp single
                {
                    buffers.resize(omp_get_num_threads());
                }
                buffers.at(omp_get_thread_num()) = new buffer_type(approximation.buffer_initialization());
            }

            return buffers;
        }

        /**
        * @brief
        *
        * @tparam BuffersPtrType
	*
        * @param buffers
        *
        * @return
	*/
        template<typename BuffersPtrType>
        auto delete_buffers(std::vector<BuffersPtrType> const& buffers) -> void
        {
#pragma omp parallel
            {
                delete buffers.at(omp_get_thread_num());
            }
        }

    }   // namespace impl

    /**
    * @brief Resets to zero the particles in the tree (task-based version).
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_particles(TreeType& tree)
    {
#pragma omp parallel default(none) shared(tree)
        {
#pragma omp single nowait
            {
                for(auto pg: tree.vector_of_leaf_groups())
                {
#pragma omp task default(none) firstprivate(pg)
                    {
                        // reset the particles in the block
                        pg->storage().reset_particles();
                    }
                }
            }
#pragma omp taskwait
        }
    }

    /**
    * @brief Resets to zero the positions of the particles in the tree (task-based version).
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_positions(TreeType& tree)
    {
#pragma omp parallel shared(tree)
        {
#pragma omp single nowait
            {
                for(auto pg: tree.vector_of_leaf_groups())
                {
#pragma omp task firstprivate(pg)
                    {
                        // reset the inputs in the block
                        pg->storage().reset_positions();
                    }
                }
            }
#pragma omp taskwait
        }
    }

    /**
    * @brief Resets to zero the inputs of the particles in the tree (task-based version).
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_inputs(TreeType& tree)
    {
#pragma omp parallel shared(tree)
        {
#pragma omp single nowait
            {
                for(auto pg: tree.vector_of_leaf_groups())
                {
#pragma omp task firstprivate(pg)
                    {
                        // reset the inputs in the block
                        pg->storage().reset_inputs();
                    }
                }
            }
#pragma omp taskwait
        }
    }

    /**
    * @brief Resets to zero the outputs of the particles in the tree (task-based version).
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_outputs(TreeType& tree)
    {
#pragma omp parallel shared(tree)
        {
#pragma omp single nowait
            {
                for(auto pg: tree.vector_of_leaf_groups())
                {
#pragma omp task firstprivate(pg)
                    {
                        // reset the outputs in the block
                        pg->storage().reset_outputs();
                    }
                }
            }
#pragma omp taskwait
        }
    }

    /**
    * @brief Resets to zero the variables of the particles in the tree (task-based version).
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_variables(TreeType& tree)
    {
#pragma omp parallel shared(tree)
        {
#pragma omp single nowait
            {
                for(auto pg: tree.vector_of_leaf_groups())
                {
#pragma omp task firstprivate(pg)
                    {
                        // reset the inputs in the block
                        pg->storage().reset_variables();
                    }
                }
            }
#pragma omp taskwait
        }
    }

    /**
    * @brief Resets to zero the multipole and the local values in the cells (task-based version).
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_far_field(TreeType& tree) -> void
    {
#pragma omp parallel shared(tree)
        {
#pragma omp single nowait
            {
                auto cell_level_it = tree.cbegin_cells() + (tree.height() - 1);

                int top_level = tree.box().is_periodic() ? 0 : 2;
                for(int level = int(tree.height()) - 1; level >= top_level; --level)
                {
                    auto group_of_cell_begin = std::cbegin(*(cell_level_it));
                    auto group_of_cell_end = std::cend(*(cell_level_it));
                    for(auto it = group_of_cell_begin; it != group_of_cell_end; ++it)
                    {
                        auto const& ptr_group = *it;
#pragma omp task firstprivate(ptr_group)
                        {
                            // auto const& current_group_symbolics = ptr_group->csymbolics();
                            component::for_each(std::begin(*ptr_group), std::end(*ptr_group),
                                                [](auto& cell)
                                                {
                                                    cell.reset_multipoles();
                                                    cell.reset_locals();
                                                });
                        }
                    }
                    --cell_level_it;
                }
            }
#pragma omp taskwait
        }
    }

    /**
    * @brief Resets to zero the multipole values in the cells (task-based version).
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_multipoles(TreeType& tree) -> void
    {
#pragma omp parallel shared(tree)
        {
#pragma omp single nowait
            {
                auto cell_level_it = tree.cbegin_cells() + (tree.height() - 1);

                int top_level = tree.box().is_periodic() ? 0 : 2;
                for(int level = int(tree.height()) - 1; level >= top_level; --level)
                {
                    auto group_of_cell_begin = std::cbegin(*(cell_level_it));
                    auto group_of_cell_end = std::cend(*(cell_level_it));
                    for(auto it = group_of_cell_begin; it != group_of_cell_end; ++it)
                    {
                        auto const& ptr_group = *it;
#pragma omp task firstprivate(ptr_group)
                        {
                            component::for_each(std::begin(*ptr_group), std::end(*ptr_group),
                                                [](auto& cell) { cell.reset_multipoles(); });
                        }
                    }
                    --cell_level_it;
                }
            }
#pragma omp taskwait
        }
    }

    /**
    * @brief Resets to zero the local values in the cells (task-based version).
    *
    * @tparam TreeType
    *
    * @param tree
    */
    template<typename TreeType>
    inline auto reset_locals(TreeType& tree) -> void
    {
#pragma omp parallel shared(tree)
        {
#pragma omp single nowait
            {
                auto cell_level_it = tree.cbegin_cells() + (tree.height() - 1);

                int top_level = tree.box().is_periodic() ? 0 : 2;
                for(int level = int(tree.height()) - 1; level >= top_level; --level)
                {
                    auto group_of_cell_begin = std::cbegin(*(cell_level_it));
                    auto group_of_cell_end = std::cend(*(cell_level_it));
                    for(auto it = group_of_cell_begin; it != group_of_cell_end; ++it)
                    {
                        auto const& ptr_group = *it;
#pragma omp task firstprivate(ptr_group)
                        {
                            component::for_each(std::begin(*ptr_group), std::end(*ptr_group),
                                                [](auto& cell) { cell.reset_locals(); });
                        }
                    }
                    --cell_level_it;
                }
            }
#pragma omp taskwait
        }
    }
}   // namespace scalfmm::algorithms::omp

#endif
