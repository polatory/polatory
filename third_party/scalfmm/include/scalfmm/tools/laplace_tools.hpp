// --------------------------------
// See LICENCE file at project root
// File : algorithm/full_direct.hpp
// --------------------------------
#ifndef SCALFMM_EXAMPLES_POST_PROCESSING_LAPLACE_HPP
#define SCALFMM_EXAMPLES_POST_PROCESSING_LAPLACE_HPP
#include <tuple>

#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tree/group_tree_view.hpp"

#include <cpp_tools/cl_parser/cl_parser.hpp>
#include <cpp_tools/colors/colorized.hpp>

namespace laplace
{
    namespace args
    {
        /**
         * @brief
         *
         */
        struct matrix_kernel
        {
            cpp_tools::cl_parser::str_vec flags = {"--kernel", "-k"};
            std::string description = "Matrix kernels: \n   0) 1/r, 1) grad(1/r), 2) p & grad(1/r) 3) shift(1/r)-> "
                                      "grad 4) shift(1/r)-> p & grad. ";
            using type = int;
            type def = 0;
        };

        /**
         * @brief
         *
         */
        struct post_traitement
        {
            /// Unused type, mandatory per interface specification
            using type = bool;
            /// The parameter is a flag, it doesn't expect a following value
            enum
            {
                flagged
            };
            cpp_tools::cl_parser::str_vec flags = {"--post_traitement", "-pt"};
            std::string description = "Post traitement to obtain Electric field or the weight ";
        };
    }   // namespace args

    /**
     * @brief
     *
     * @tparam Dimension
     * @tparam ContainerType
     * @tparam PointType
     * @tparam ValueType
     * @param filename
     * @param container
     * @param Centre
     * @param width
     */
    template<std::size_t Dimension, typename ContainerType, typename PointType, typename ValueType>
    auto read_data(const std::string& filename, ContainerType*& container, PointType& Centre, ValueType& width) -> void
    {
        //  std::cout << "READ DATA " << std::endl << std::flush;
        using particle_type = typename ContainerType::particle_type;
        bool verbose = true;

        scalfmm::io::FFmaGenericLoader<ValueType, Dimension> loader(filename, verbose);

        const auto number_of_particles = loader.getNumberOfParticles();
        width = loader.getBoxWidth();
        Centre = loader.getBoxCenter();

        auto nb_val_to_red_per_part = loader.get_dimension() + loader.get_number_of_input_per_record();
        ValueType* values_to_read = new ValueType[nb_val_to_red_per_part]{};
        container = new ContainerType(number_of_particles);
        std::cout << "number_of_particles " << number_of_particles << std::endl;
        for(std::size_t idx = 0; idx < number_of_particles; ++idx)
        {
            loader.fillParticle(values_to_read, nb_val_to_red_per_part);
            particle_type p;
            std::size_t ii{0};
            for(auto& e: p.position())
            {
                e = values_to_read[ii++];
            }
            for(auto& e: p.inputs())
            {
                e = values_to_read[ii++];
            }
            for(auto& e: p.outputs())
            {
                e = 0.;
            }
            container->insert_particle(idx, p);
        }
        loader.close();
    }

    /**
     * @brief
     *
     * @tparam MatrixKernelType
     * @tparam LeafType
     * @param mat
     * @param leaf
     * @return auto
     */
    template<class MatrixKernelType, class LeafType>
    auto compute_energy_tuple(MatrixKernelType const& mat, LeafType const& leaf)
    {
        using outputs_type = typename LeafType::particle_type::outputs_type;
        using inputs_type = typename LeafType::particle_type::inputs_type;
        //
        outputs_type energy{};
        for(auto& e: energy)
        {
            e = 0.;
        };
        // std::array<value_type, MatrixKernelType::km>
        inputs_type total_physical_value{};
        for(auto& e: total_physical_value)
        {
            e = 0.;
        };
        for(auto const p_tuple_ref: leaf)
        {
            const auto p = typename LeafType::const_proxy_type(p_tuple_ref);
            // }
            // for(std::size_t i = 0; i < leaf.size(); ++i)
            // {

            // get charge
            const auto q = p.inputs();
            // get forces
            auto out = p.outputs();
            // update
            for(std::size_t j = 0; j < MatrixKernelType::kn; ++j)
            {
                energy[j] += q[j] * out[j];
                std::cout << out[j] << '\n';
            }
            for(std::size_t j = 0; j < MatrixKernelType::km; ++j)
            {
                total_physical_value[j] += q[j];
            }
        }
        return std::make_tuple(energy, total_physical_value);
    }

    /**
     * @brief multiply_force_by_q Compute the force for the simulation F = q \nabla mk
     *
     * @tparam ContainerType
     * @param begin_force  the first position of the force in the outputs
     * @param end_force    the las position of the force in the outputs
     * @param container    the container of particles
     */
    template<class ContainerType>
    auto multiply_force_by_q(const int begin_force, const int end_force, ContainerType& container) -> void
    {
        // const std::size_t nb_part = container.size();
        const auto nb_part = std::distance(std::begin(container), std::end(container));
        for(std::size_t i = 0; i < nb_part; ++i)
        {
            // get charge
            const auto q = container.inputs(i);
            // get forces
            auto out = container.outputs(i);
            // update
            for(int k = begin_force; k < end_force; ++k)
            {
                out[k] *= q[0];
            }
        }
    }

    /**
     * @brief post_traitement specialization function for one_over_r.
     *
     * @tparam ContainerType
     * @param mat the matrix kernel.
     * @param container the container of particles.
     */
    template<class ContainerType>
    auto post_traitement(scalfmm::matrix_kernels::laplace::like_mrhs& mat, ContainerType& container) -> void
    {
        std::cout << "From post_traitement like_mrhs " << std::endl;
        //compute_energy(mat, container);
    }

    /**
     * @brief post_traitement specialization function for val_tgrad_one_over_r
     *
     * Here we compute the forces ont the particles  Q \nabla 1/r
     *   given by inputs[0] * outputs[1-3].
     *
     * @tparam ContainerType
     * @tparam Dimension
     * @param mat         The kernel matrix
     * @param container   The contaier of particles
     */
    template<class ContainerType, std::size_t Dimension>
    auto post_traitement(scalfmm::matrix_kernels::laplace::val_grad_one_over_r<Dimension>& mat,
                         ContainerType& container) -> void
    {
        std::cout << "From post_traitement val_tgrad_one_over_r" << std::endl;
        multiply_force_by_q(1, 4, container);
        // compute_energy(mat, container);
    }

    /**
     * @brief post_traitement specialization function for grad_one_over_r
     *
     * Here we compute the forces ont the particles  Q \nabla 1/r
     *   given by inputs[0] * outputs[0-2].
     *
     * @tparam MatrixKernelType
     * @tparam ContainerType
     * @tparam Dimension
     * @param mat         The kernel matrix
     * @param container   The container of particles
     */
    template<class MatrixKernelType, class ContainerType, std::size_t Dimension>
    auto post_traitement(scalfmm::matrix_kernels::laplace::grad_one_over_r<Dimension>& mat, ContainerType& container)
      -> void
    {
        std::cout << "From post_traitement grad_one_over_r " << std::endl;
        multiply_force_by_q(0, 3, container);
    }

    /**
     * @brief post_traitement specialization function for one_over_r
     *
     * Here we compute and print the energy
     *
     * @tparam ContainerType
     * @param mat         The kernel matrix
     * @param container   The contaier of particles
     */
    template<class ContainerType>
    auto post_traitement(scalfmm::matrix_kernels::laplace::one_over_r& mat, ContainerType& container) -> void
    {
        std::cout << "From post_traitement one_over_r " << std::endl;
        // compute_energy(mat, container);
    }

    /**
     * @brief post_traitement The generic function. Nothing is done.
     *
     * @tparam MatrixKernelType
     * @tparam CellType
     * @tparam LeafType
     * @tparam BoxType
     * @param mat The kernel matrix
     * @param tree The tree of particles
     */
    template<class MatrixKernelType, class CellType, class LeafType, class BoxType>
    void post_traitement(MatrixKernelType& mat, scalfmm::component::group_tree_view<CellType, LeafType, BoxType>* tree)
    {
        std::cout << "Generic nothing to do tree \n";
    }

    /**
     * @brief post_traitement The generic function. Nothing is done.
     *
     * @tparam MatrixKernelType
     * @tparam ContainerType
     * @param mat         The kernel matrix
     * @param container   The contaier of particles
     */
    template<class MatrixKernelType, class ContainerType>
    auto post_traitement(MatrixKernelType& mat, ContainerType& container) -> void
    {
        std::cout << "Generic nothing to do \n";
    }
}   // namespace laplace
#endif
