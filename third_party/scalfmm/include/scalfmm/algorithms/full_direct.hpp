// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/full_direct.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_FULL_DIRECT_HPP
#define SCALFMM_ALGORITHMS_FULL_DIRECT_HPP

#include "scalfmm/container/iterator.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/utils/io_helpers.hpp"
#include "scalfmm/utils/static_assert_as_exception.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <omp.h>

using namespace scalfmm::io;

namespace scalfmm::algorithms
{
    /**
    * @brief compute the direct particle interactions for the built-in particle container.
    *
    * Compute the direct interactions between the particles inside the containers.
    *
    *   \f$ out(pt) = \sum_{ps \ne pt }{ matrix\_kernel(pt, ps) input(ps) }  \f$
    *
    * @tparam ParticleContainerType
    * @tparam MatrixKernelType
    *
    * @param[inout] particles the container of particles
    * @param[in] matrix_kernel the matrix kernel
    */
    template<typename ParticleContainerType, typename MatrixKernelType>
    inline auto full_direct(ParticleContainerType& particles, const MatrixKernelType& matrix_kernel) -> void
    {
        using particle_container_type = ParticleContainerType;
        using matrix_kernel_type = MatrixKernelType;
        using particle_type = typename particle_container_type::value_type;
        using value_type = typename particle_type::position_type::value_type;
        using position_type = typename particle_type::position_type;
        static constexpr std::size_t dimension = position_type::dimension;
        using matrix_type = typename matrix_kernel_type::template matrix_type<value_type>;

        static_assert(matrix_kernel_type::km == particle_type::inputs_size,
                      "Different input size between Matrix kernel and container!");
        static_assert(matrix_kernel_type::kn == particle_type::outputs_size,
                      "Different output size between Matrix kernel and container!");

        static constexpr std::size_t inputs_size = matrix_kernel_type::km;
        static constexpr std::size_t outputs_size = matrix_kernel_type::kn;

#pragma omp parallel for default(none) shared(particles, matrix_kernel) schedule(static)
        for(std::size_t idx = 0; idx < particles.size(); ++idx)
        {
            // Get proxy particle position
            auto pt_x = particles.at(idx).position();
            // Get proxy particle outputs
            auto val = particles.at(idx).outputs();

            // val the array of outputs
            matrix_type val_mat{};
            auto compute = [&pt_x, &val, &matrix_kernel, &val_mat, &particles](std::size_t start, std::size_t end)
            {
                for(std::size_t idx_2 = start; idx_2 < end; ++idx_2)
                {
                    auto q = particles.at(idx_2).inputs();
                    auto pt_y = particles.at(idx_2).position();
                    val_mat = matrix_kernel.evaluate(pt_x, pt_y);
                    for(std::size_t j = 0; j < outputs_size; ++j)
                    {
                        for(std::size_t i = 0; i < inputs_size; ++i)
                        {
                            val[j] += val_mat.at(j * inputs_size + i) * q[i];
                        }
                    }
                }
            };

            compute(0, idx);
            compute(idx + 1, particles.size());

            // Treat the case of the self-interaction only if we deal with a smooth kernel
            if constexpr(meta::has_self_interaction_v<matrix_kernel_type, position_type>)
            {
                auto q = particles.at(idx).inputs();
                auto pt_x = particles.at(idx).position();
                val_mat = matrix_kernel.evaluate(pt_x);
                for(std::size_t j = 0; j < outputs_size; ++j)
                {
                    for(std::size_t i = 0; i < inputs_size; ++i)
                    {
                        val[j] += val_mat.at(j * inputs_size + i) * q[i];
                    }
                }
            }
        }
    }

    /**
    * @brief compute the direct particle interactions (when the particle container is a std::vector).
    *
    * @tparam ParticleType
    * @tparam MatrixKernelType
    *
    * @param particles
    * @param matrix_kernel
    */
    template<typename ParticleType, typename MatrixKernelType>
    inline auto full_direct(std::vector<ParticleType>& particles, const MatrixKernelType& matrix_kernel) -> void
    {
        using particle_type = ParticleType;
        using matrix_kernel_type = MatrixKernelType;
        using value_type = typename particle_type::position_type::value_type;
        using matrix_type = typename matrix_kernel_type::template matrix_type<value_type>;
        using position_type = typename particle_type::position_type;
        static constexpr std::size_t dimension = position_type::dimension;

        static_assert(matrix_kernel_type::km == particle_type::inputs_size,
                      "Different input size between Matrix kernel and container!");
        static_assert(matrix_kernel_type::kn == particle_type::outputs_size,
                      "Different output size between Matrix kernel and container!");

        static constexpr std::size_t inputs_size = matrix_kernel_type::km;
        static constexpr std::size_t outputs_size = matrix_kernel_type::kn;

#pragma omp parallel for default(none) shared(particles, matrix_kernel) schedule(static)
        for(std::size_t idx = 0; idx < particles.size(); ++idx)
        {
            // Get proxy particle position
            auto& pt_x = particles.at(idx).position();
            // Get proxy particle outputs
            auto& val = particles.at(idx).outputs();

            // val the array of outputs
            matrix_type val_mat{};
            auto compute = [&pt_x, &val, &matrix_kernel, &val_mat, &particles](std::size_t start, std::size_t end)
            {
                for(std::size_t idx_2 = start; idx_2 < end; ++idx_2)
                {
                    auto q = particles.at(idx_2).inputs();
                    auto pt_y = particles.at(idx_2).position();
                    val_mat = matrix_kernel.evaluate(pt_x, pt_y);
                    for(std::size_t j = 0; j < outputs_size; ++j)
                    {
                        for(std::size_t i = 0; i < inputs_size; ++i)
                        {
                            val[j] += val_mat.at(j * inputs_size + i) * q[i];
                        }
                    }
                }
            };

            compute(0, idx);
            compute(idx + 1, particles.size());

            // Treat the case of the self-interaction only if we deal with a smooth kernel
            if constexpr(meta::has_self_interaction_v<matrix_kernel_type, position_type>)
            {
                auto q = particles.at(idx).inputs();
                auto pt_x = particles.at(idx).position();
                val_mat = matrix_kernel.evaluate(pt_x);
                for(std::size_t j = 0; j < outputs_size; ++j)
                {
                    for(std::size_t i = 0; i < inputs_size; ++i)
                    {
                        val[j] += val_mat.at(j * inputs_size + i) * q[i];
                    }
                }
            }
        }
    }

    /**
    * @brief compute the direct interactions with a built-in source container and a built-in target container.
    *
    * @tparam SourceContainerType
    * @tparam TargetContainerType
    * @tparam MatrixKernelType
    *
    * @param source_particles
    * @param target_particles
    * @param matrix_kernel
    */
    template<typename SourceContainerType, typename TargetContainerType, typename MatrixKernelType>
    inline auto full_direct(SourceContainerType const& source_particles, TargetContainerType& target_particles,
                            const MatrixKernelType& matrix_kernel)
    {
        using source_container_type = SourceContainerType;
        using target_container_type = TargetContainerType;
        using matrix_kernel_type = MatrixKernelType;
        using source_particle_type = typename source_container_type::value_type;
        using target_particle_type = typename target_container_type::value_type;
        using value_type = typename source_particle_type::position_type::value_type;
        using matrix_type = typename matrix_kernel_type::template matrix_type<value_type>;

        static_assert(matrix_kernel_type::km == source_particle_type::inputs_size,
                      "Different input size between Matrix kernel and container!");
        static_assert(matrix_kernel_type::kn == target_particle_type::outputs_size,
                      "Different output size between Matrix kernel and container!");

        static constexpr std::size_t inputs_size = matrix_kernel_type::km;
        static constexpr std::size_t outputs_size = matrix_kernel_type::kn;

        bool same_container{false};
        if constexpr(std::is_same_v<source_container_type, target_container_type>)
        {
            same_container = (&source_particles == &target_particles);
        }
        if(same_container)
        {
            throw std::runtime_error("full_direct: the two containers must be different!");
        }

#pragma omp parallel for default(none) shared(source_particles, target_particles, matrix_kernel) schedule(static)
        for(std::size_t idx = 0; idx < target_particles.size(); ++idx)
        {
            // Get proxy particle position
            auto pt_x = target_particles.at(idx).position();
            // Get proxy particle outputs
            auto val = target_particles.at(idx).outputs();
            matrix_type val_mat{};

            auto compute =
              [&pt_x, &val, &matrix_kernel, &val_mat, &source_particles](std::size_t start, std::size_t end)
            {
                for(std::size_t idx_2 = start; idx_2 < end; ++idx_2)
                {
                    auto q = source_particles.at(idx_2).inputs();

                    auto pt_y = source_particles.at(idx_2).position();

                    val_mat = matrix_kernel.evaluate(pt_x, pt_y);
                    for(std::size_t j = 0; j < outputs_size; ++j)
                    {
                        for(std::size_t i = 0; i < inputs_size; ++i)
                        {
                            val[j] += val_mat.at(j * inputs_size + i) * q[i];
                        }
                    }
                }
            };

            compute(0, source_particles.size());
        }
    }

    /**
    * @brief compute the field due to the particles the source container on the target particles
    *
    * compute the field due to the  particles the source container on the target particles by
    *
    *   \f$ out(pt) = \sum_{ps\in source} { matrix\_kernel(pt, ps) input(ps) }  \f$
    *
    * We don't check if | pt - ps| =0
    *
    * @tparam SourceParticleType
    * @tparam TargetParticleType
    * @tparam MatrixKernelType
    *
    * @param[in] source_particles container of the source particles
    * @param[inout] target_particles container of the target particles
    * @param[in] matrix_kernel matrix kernel
    */
    template<typename SourceParticleType, typename TargetParticleType, typename MatrixKernelType>
    inline auto full_direct(std::vector<SourceParticleType> const& source_particles,
                            std::vector<TargetParticleType>& target_particles, MatrixKernelType const& matrix_kernel)
    {
        using source_particle_type = SourceParticleType;
        using target_particle_type = TargetParticleType;
        using matrix_kernel_type = MatrixKernelType;
        using value_type = typename source_particle_type::position_type::value_type;
        using matrix_type = typename matrix_kernel_type::template matrix_type<value_type>;

        static_assert(matrix_kernel_type::km == source_particle_type::inputs_size,
                      "Different input size between Matrix kernel and source container!");
        static_assert(matrix_kernel_type::kn == target_particle_type::outputs_size,
                      "Different output size between Matrix kernel and target container!");
        static constexpr std::size_t inputs_size = matrix_kernel_type::km;
        static constexpr std::size_t outputs_size = matrix_kernel_type::kn;

        bool same_container{false};
        if constexpr(std::is_same_v<source_particle_type, target_particle_type>)
        {
            same_container = (&source_particles == &target_particles);
        }
        if(same_container)
        {
            throw std::runtime_error("full_direct: the two containers must be different !");
        }

#pragma omp parallel for default(none) shared(source_particles, target_particles, matrix_kernel) schedule(static)
        for(std::size_t idx = 0; idx < target_particles.size(); ++idx)
        {
            auto& p = target_particles[idx];

            // // Get particle position
            auto const& pt_x = p.position();
            // // val is an alias on the array of outputs
            auto& val = p.outputs();
            matrix_type val_mat{};

            auto compute =
              [&pt_x, &val, &matrix_kernel, &val_mat, &source_particles](std::size_t start, std::size_t end)
            {
                for(std::size_t idx_2 = start; idx_2 < end; ++idx_2)
                {
                    auto const& q = source_particles.at(idx_2).inputs();

                    auto const& pt_y = source_particles.at(idx_2).position();

                    val_mat = matrix_kernel.evaluate(pt_x, pt_y);
                    for(std::size_t j = 0; j < outputs_size; ++j)
                    {
                        for(std::size_t i = 0; i < inputs_size; ++i)
                        {
                            val[j] += val_mat.at(j * inputs_size + i) * q[i];
                        }
                    }
                }
            };

            compute(0, source_particles.size());
        }
    }
}   // namespace scalfmm::algorithms

#endif   // SCALFMM_ALGORITHMS_FULL_DIRECT_HPP
