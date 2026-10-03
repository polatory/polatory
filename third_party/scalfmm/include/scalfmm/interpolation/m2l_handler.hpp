// -----------------------------------
// See LICENCE file at project root
// File : scalfmm/interpolation/m2l_handler.hpp
// -----------------------------------
#ifndef SCALFMM_INTERPOLATION_M2L_HANDLER_HPP
#define SCALFMM_INTERPOLATION_M2L_HANDLER_HPP

#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/builders.hpp"
#include "scalfmm/interpolation/permutations.hpp"
#include "scalfmm/matrix_kernels/mk_common.hpp"
#include "scalfmm/memory/storage.hpp"
#include "scalfmm/meta/const_functions.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/options/options.hpp"
#include "scalfmm/utils/io_helpers.hpp"
#include "scalfmm/utils/low_rank.hpp"
#include "scalfmm/utils/math.hpp"
#include "scalfmm/utils/tensor.hpp"

#include "xsimd/config/xsimd_config.hpp"
#include "xsimd/memory/xsimd_aligned_allocator.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/core/xlayout.hpp"
#include "xtensor/io/xnpy.hpp"
#include "xtensor/views/xslice.hpp"
#include "xtensor/core/xtensor_config.hpp"
#include "xtensor/core/xvectorize.hpp"
#include "xtensor/views/xview.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <any>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

using namespace scalfmm::io;

namespace scalfmm::interpolation
{
    /**
     * @brief
     *
     * @tparam Derived
     */
    template<typename Derived>
    struct interpolator_traits;

    namespace impl
    {
        /**
         * @brief
         *
         * @warning Cell width extension is not yet supported for homogeneous kernels in the latest version of ScalFMM!
         *
         * @tparam Derived
         */
        template<typename Derived>
        struct m2l_handler
        {
          private:
            /**
             * @brief
             *
             */
            struct empty
            {
            };

          public:
            using derived_type = Derived;
            using value_type = typename interpolator_traits<derived_type>::value_type;
            using size_type = std::size_t;

            static constexpr std::size_t dimension = interpolator_traits<derived_type>::dimension;

            using matrix_kernel_type = typename interpolator_traits<derived_type>::matrix_kernel_type;

            using settings = typename interpolator_traits<derived_type>::settings;

            /**
             * @brief
             *
             */
            static constexpr std::size_t kn = matrix_kernel_type::kn;

            /**
             * @brief
             *
             */
            static constexpr std::size_t km = matrix_kernel_type::km;

            /**
             * @brief
             *
             */
            static constexpr auto separation_criterion = matrix_kernel_type::separation_criterion;

            /**
             * @brief
             *
             */
            static constexpr auto homogeneity_tag = matrix_kernel_type::homogeneity_tag;

            /**
             * @brief
             *
             */
            static constexpr auto symmetry_tag = matrix_kernel_type::symmetry_tag;

            /**
             * @brief
             *
             */
            static constexpr auto enable_symmetries = interpolator_traits<derived_type>::enable_symmetries;

            /**
             * @brief
             *
             */
            static constexpr std::size_t max_number_of_cell{7};

            /**
             * @brief
             *
             */
            static constexpr bool symmetry_support{
              (symmetry_tag == matrix_kernels::symmetry::symmetric && (enable_symmetries == true) && (dimension < 4))};

            using scale_factor_type = typename matrix_kernel_type::template vector_type<value_type>;
            using sym_permutations_type = std::conditional_t<symmetry_support, xt::xarray<int>, empty>;
            using k_indices_type = std::conditional_t<symmetry_support, std::vector<std::size_t>, empty>;

            using array_type = xt::xarray<value_type>;
            using array_shape_type = typename array_type::shape_type;

            template<std::size_t d>
            using tensor_type = xt::xtensor<value_type, d>;
            template<std::size_t d>
            using tensor_shape_type = typename tensor_type<d>::shape_type;

            using storage_type = typename interpolator_traits<derived_type>::storage_type;
            using buffer_value_type = typename interpolator_traits<derived_type>::buffer_value_type;
            using buffer_inner_type = typename interpolator_traits<derived_type>::buffer_inner_type;
            using k_tensor_type = std::conditional_t<std::is_same_v<settings, options::low_rank_>,
                                                     std::tuple<xt::xtensor<value_type, 2>, xt::xtensor<value_type, 2>>,
                                                     typename interpolator_traits<derived_type>::k_tensor_type>;
            using interaction_matrix_type = xt::xtensor_fixed<k_tensor_type, xt::xshape<kn, km>>;
            using buffer_shape_type = typename interpolator_traits<derived_type>::buffer_shape_type;
            using buffer_type = typename interpolator_traits<derived_type>::buffer_type;
            using multipoles_inner_type =
              typename memory::storage_traits<typename storage_type::multipoles_storage_type>::inner_type;
            using locals_inner_type =
              typename memory::storage_traits<typename storage_type::locals_storage_type>::inner_type;

            /**
             * @brief Construct a new m2l handler object
             *
             */
            m2l_handler() = delete;

            /**
             * @brief Construct a new m2l handler object
             *
             * @param other
             */
            m2l_handler(m2l_handler const& other) = delete;

            /**
             * @brief Construct a new m2l handler object
             *
             */
            m2l_handler(m2l_handler&&) noexcept = delete;

            /**
             * @brief
             *
             * @return m2l_handler&
             */
            auto operator=(m2l_handler const&) -> m2l_handler& = delete;

            /**
             * @brief
             *
             * @return m2l_handler&
             */
            auto operator=(m2l_handler&&) noexcept -> m2l_handler& = delete;

            /**
             * @brief Destroy the m2l handler object
             *
             */
            ~m2l_handler() = default;

            /**
             * @brief Construct a new m2l handler object
             *
             * @param far_field
             * @param roots
             * @param tree_height
             * @param root_cell_width
             * @param cell_width_extension
             * @param late_init
             */
            m2l_handler(matrix_kernel_type const& far_field, array_type roots, size_type tree_height = 3,
                        value_type root_cell_width = value_type(1.), value_type cell_width_extension = value_type(0.),
                        bool late_init = false)
              : m_far_field(far_field)
              , m_m2l_interactions(math::pow(max_number_of_cell, dimension))
              , m_nnodes(meta::pow(roots.size(), dimension))
              , m_order(roots.size())
              , m_roots(roots)
              , m_epsilon(std::pow(value_type(10.), -value_type(roots.size() - 1)))
              , m_cell_width_extension(cell_width_extension)
            {
                if((cell_width_extension > 0) && (homogeneity_tag == matrix_kernels::homogeneity::homogenous))
                {
                    std::cout << "cell_width_extension: " << cell_width_extension << std::endl;
                    std::cout << "matrix kernel name: " << far_field.name() << std::endl;
                    throw std::runtime_error(
                      "m2lhandler: Cell width extension is not yet supported for homogeneous kernels in the "
                      "latest version of ScalFMM!");
                }

                if(late_init == false)
                {
                    this->initialize(roots.size(), root_cell_width, tree_height);
                }

                if constexpr(symmetry_support)
                {
                    std::tie(m_sym_permutations, m_k_indices) = get_permutations_and_indices<dimension>(
                      roots.size(), meta::pow(roots.size(), dimension), this->m2l_interactions());
                }
            }

            /**
             * @brief
             *
             * @return array_type const&
             */
            [[nodiscard]] inline auto weights() const noexcept -> array_type const& { return m_weights_of_roots; }

            /**
             * @brief
             *
             * @return value_type&
             */
            [[nodiscard]] inline auto epsilon() noexcept -> value_type& { return m_epsilon; }

            /**
             * @brief
             *
             * @return value_type
             */
            [[nodiscard]] inline auto epsilon() const noexcept -> value_type { return m_epsilon; }

            /**
             * @brief member function to get the index of K corresponding to the interaction index in the interaction matrix vector.
             *
             * @param neighbor_idx
             * @return std::size_t
             */
            [[nodiscard]] inline auto symmetry_k_index(std::size_t neighbor_idx) const -> std::size_t
            {
                return m_k_indices.at(neighbor_idx);
            }

            /**
             * @brief
             *
             * @tparam TensorOrViewX
             * @tparam TensorOrViewY
             * @param X
             * @param Y
             * @param n
             * @param m
             * @param thread_id
             * @return std::enable_if_t<
             * !decltype(meta::sig_gen_k_f(std::declval<derived_type>(), X, Y, n, m, thread_id))::value, k_tensor_type>
             */
            template<typename TensorOrViewX, typename TensorOrViewY>
            [[nodiscard]] inline auto generate_matrix_k(TensorOrViewX&& X, TensorOrViewY&& Y, std::size_t n,
                                                        std::size_t m, [[maybe_unused]] size_type thread_id = 0) const
              -> std::enable_if_t<
                !decltype(meta::sig_gen_k_f(std::declval<derived_type>(), X, Y, n, m, thread_id))::value, k_tensor_type>

            {
                if constexpr(std::is_same_v<settings, options::dense_>)
                {
                    auto const& matrix_kernel{this->matrix_kernel()};

                    auto n_d = math::pow(m_order, dimension);

                    k_tensor_type K(std::vector(2, n_d));
                    auto flat_x = xt::flatten(X);
                    auto flat_y = xt::flatten(Y);
                    // TODO : SIMD!

                    for(std::size_t i{0}; i < n_d; ++i)
                    {
                        for(std::size_t j{0}; j < n_d; ++j)
                        {
                            K.at(i, j) = matrix_kernel.evaluate(flat_x.at(i), flat_y.at(j)).at(n * km + m);
                        }
                    }
                    // std::cout << cpp_tools::colors::cyan;
                    // std::cout << K << std::endl;
                    // std::cout << cpp_tools::colors::reset;
                    // xt::dump_npy("interaction_matrix_non_homogenous_with_ext.npy",K);
                    return K;
                }
                else if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    auto const& matrix_kernel{this->matrix_kernel()};
                    return low_rank::generate(matrix_kernel, std::forward<TensorOrViewX>(X),
                                              std::forward<TensorOrViewY>(Y), this->weights(), this->epsilon(), n, m);
                }
                else
                {
                    throw std::runtime_error("Missing generate_matrix_k function!");
                }
            }

            /**
             * @brief
             *
             * @tparam TensorOrViewX
             * @tparam TensorOrViewY
             * @param X
             * @param Y
             * @param n
             * @param m
             * @param thread_id
             * @return std::enable_if_t<
             * decltype(meta::sig_gen_k_f(std::declval<derived_type>(), X, Y, n, m, thread_id))::value, k_tensor_type>
             */
            template<typename TensorOrViewX, typename TensorOrViewY>
            [[nodiscard]] inline auto generate_matrix_k(TensorOrViewX&& X, TensorOrViewY&& Y, std::size_t n,
                                                        std::size_t m, [[maybe_unused]] size_type thread_id = 0) const
              -> std::enable_if_t<
                decltype(meta::sig_gen_k_f(std::declval<derived_type>(), X, Y, n, m, thread_id))::value, k_tensor_type>
            {
                return this->derived_cast().generate_matrix_k_impl(std::forward<TensorOrViewX>(X),
                                                                   std::forward<TensorOrViewX>(Y), n, m, thread_id);
            }

            /**
             * @brief
             *
             * @return std::vector<interaction_matrix_type, XTENSOR_DEFAULT_ALLOCATOR(interaction_matrix_type)>&
             */
            [[nodiscard]] inline auto interactions_matrices() noexcept
              -> std::vector<interaction_matrix_type, XTENSOR_DEFAULT_ALLOCATOR(interaction_matrix_type)>&
            {
                return m_interactions_matrices;
            }

            /**
             * @brief
             *
             * @return std::vector<interaction_matrix_type> const&
             */
            [[nodiscard]] inline auto
            interactions_matrices() const noexcept -> std::vector<interaction_matrix_type> const&
            {
                return m_interactions_matrices;
            }

            /**
             * @brief
             *
             * @return std::vector<interaction_matrix_type> const&
             */
            [[nodiscard]] inline auto
            cinteractions_matrices() const noexcept -> std::vector<interaction_matrix_type> const&
            {
                return m_interactions_matrices;
            }
            /**
             * @brief Compute he memory in bytes used by the interpolator
             *
             * @return the memory used by the interpolator
             */
            [[nodiscard]] inline auto memory_usage() const noexcept -> std::size_t const
            {
                std::size_t memory{0};
                auto size = m_interactions_matrices.size();

                for(auto& mat: m_interactions_matrices)
                {
                    // mat xtensor knxkm
                    // std::cout << cpt++ << " ";
                    for(int i = 0; i < mat.shape()[0]; ++i)
                    {
                        for(int j = 0; j < mat.shape()[1]; ++j)
                        {
                            auto K = mat.at(i, j);
                            // dense case
                            if constexpr(std::is_same_v<settings, options::dense_> or
                                         std::is_same_v<settings, options::fft_>)
                            {
                                memory += K.size() * sizeof(typename k_tensor_type::value_type);
                            }
                            else if constexpr(std::is_same_v<settings, options::low_rank_>)
                            {
                                // Low rang (tuple of two matricies)
                                auto const& A = std::get<0>(K);
                                auto const& B = std::get<1>(K);
                                memory += (A.size() + B.size()) *
                                          sizeof(typename std::tuple_element_t<0, k_tensor_type>::value_type);
                            }
                        }
                    }
                }
                return memory;
            }
            /**
             * @brief
             *
             * @return auto
             */
            [[nodiscard]] inline auto m2l_interactions() const noexcept { return m_m2l_interactions; }

            /**
             * @brief
             *
             * @return sym_permutations_type const&
             */
            [[nodiscard]] inline auto sym_permutations() const -> sym_permutations_type const&
            {
                return m_sym_permutations;
            }

            [[nodiscard]] inline auto matrix_kernel() const noexcept -> matrix_kernel_type const&
            {
                return m_far_field;
            }

            /**
             * @brief
             *
             * @tparam D
             * @return std::enable_if_t<decltype(meta::sig_buffer_init_f(std::declval<D>()))::value, buffer_type>
             */
            template<typename D = derived_type>
            [[nodiscard]] auto buffer_initialization() const
              -> std::enable_if_t<decltype(meta::sig_buffer_init_f(std::declval<D>()))::value, buffer_type>
            {
                return this->derived_cast().buffer_initialization_impl();
            }

            /**
             * @brief Initialize the buffer to aggregate the multipoles and teh locals when the kernel is symmetric
             *
             *  if the kernel is non symmetric we have an empty buffers otherwise the element of the buffer
             *   is a tensor of size (number of multipole associated to the current symmetry, the number of nodes)
             *
             * @tparam D
             * @return std::enable_if_t<!decltype(meta::sig_buffer_init_f(std::declval<D>()))::value, buffer_type>
             */
            template<typename D = derived_type>
            [[nodiscard]] auto buffer_initialization() const
              -> std::enable_if_t<!decltype(meta::sig_buffer_init_f(std::declval<D>()))::value, buffer_type>
            {
                std::vector<std::size_t> shape(2, m_nnodes);
                if constexpr(enable_symmetries)
                {
                    shape[1] = interpolation::largest_number_permutation<dimension>();
                }
                return buffer_type(buffer_shape_type{}, buffer_inner_type(shape, 0.));
            }

            /**
             * @brief Reset the buffer by calling buffer_reset_impl (specialization)
             *
             * @tparam D
             * @param buffers
             */
            template<typename D = derived_type>
            inline auto buffer_reset(buffer_type& buffers) const
              -> std::enable_if_t<decltype(meta::sig_buffer_reset_f(std::declval<D>(), buffers))::value, void>
            {
                this->derived_cast().buffer_reset_impl(buffers);
            }

            /**
             * @brief Reset the buffer, generic function
             *
             * @tparam D
             * @param buffers
             */
            template<typename D = derived_type>
            inline auto buffer_reset(buffer_type& buffers) const
              -> std::enable_if_t<!decltype(meta::sig_buffer_reset_f(std::declval<D>(), buffers))::value, void>
            {
                if constexpr(symmetry_support)
                {
                    for(std::size_t n{0}; n < 2; ++n)
                    {
                        buffers.at(n).fill(buffer_value_type(0.));
                    }
                }
            }

            /**
             * @brief
             *
             * @tparam D
             * @return std::enable_if_t<!decltype(meta::sig_init_k_f(std::declval<D>()))::value, k_tensor_type>
             */
            template<typename D = derived_type>
            inline auto initialize_k() const
              -> std::enable_if_t<!decltype(meta::sig_init_k_f(std::declval<D>()))::value, k_tensor_type>
            {
                if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    return std::make_tuple(xt::xtensor<value_type, 2>{}, xt::xtensor<value_type, 2>{});
                }
                else if constexpr(std::is_same_v<settings, options::dense_>)
                {
                    return k_tensor_type{};
                }
                else
                {
                    throw std::runtime_error("Missing initialize_k function!");
                }
            }

            /**
             * @brief
             *
             * @tparam D
             * @return std::enable_if_t<decltype(meta::sig_init_k_f(std::declval<D>()))::value, k_tensor_type>
             */
            template<typename D = derived_type>
            inline auto initialize_k() const
              -> std::enable_if_t<decltype(meta::sig_init_k_f(std::declval<D>()))::value, k_tensor_type>
            {
                return this->derived_cast().initialize_k_impl();
            }

            /**
             * @brief
             *
             * @tparam Cell
             * @param current_cell
             * @param thread_id
             * @return std::enable_if_t<
             * decltype(meta::sig_preprocess_f(std::declval<derived_type>(), current_cell, thread_id))::value, void>
             */
            template<typename Cell>
            auto apply_multipoles_preprocessing(Cell& current_cell, [[maybe_unused]] size_type thread_id = 0) const
              -> std::enable_if_t<
                decltype(meta::sig_preprocess_f(std::declval<derived_type>(), current_cell, thread_id))::value, void>
            {
                return this->derived_cast().apply_multipoles_preprocessing_impl(current_cell, thread_id);
            }

            /**
             * @brief Default fallback
             *
             * @tparam Cell
             * @param current_cell
             * @param thread_id
             * @return std::enable_if_t<
             * !decltype(meta::sig_preprocess_f(std::declval<derived_type>(), current_cell, thread_id))::value, void>
             */
            template<typename Cell>
            auto apply_multipoles_preprocessing(Cell& current_cell, [[maybe_unused]] size_type thread_id = 0) const
              -> std::enable_if_t<
                !decltype(meta::sig_preprocess_f(std::declval<derived_type>(), current_cell, thread_id))::value, void>
            {
            }

            /**
             * @brief
             *
             * @tparam Cell
             * @param current_cell
             * @param products
             * @param thread_id
             * @return std::enable_if_t<decltype(meta::sig_postprocess_f(std::declval<derived_type>(), current_cell, products,
             * thread_id))::value,
             * void>
             */
            template<typename Cell>
            auto apply_multipoles_postprocessing(Cell& current_cell, [[maybe_unused]] buffer_type const& products,
                                                 [[maybe_unused]] size_type thread_id = 0) const
              -> std::enable_if_t<decltype(meta::sig_postprocess_f(std::declval<derived_type>(), current_cell, products,
                                                                   thread_id))::value,
                                  void>
            {
                return this->derived_cast().apply_multipoles_postprocessing_impl(current_cell, products, thread_id);
            }

            /**
             * @brief
             *
             * @tparam Cell
             * @param current_cell
             * @param products
             * @param thread_id
             * @return std::enable_if_t<!decltype(meta::sig_postprocess_f(std::declval<derived_type>(), current_cell,
             * products, thread_id))::value,
             * void>
             */
            template<typename Cell>
            auto apply_multipoles_postprocessing(Cell& current_cell, [[maybe_unused]] buffer_type const& products,
                                                 [[maybe_unused]] size_type thread_id = 0) const
              -> std::enable_if_t<!decltype(meta::sig_postprocess_f(std::declval<derived_type>(), current_cell,
                                                                    products, thread_id))::value,
                                  void>
            {
            }

            /**
             * @brief  Compute the matrix vector product for the M2L operator
             *
             * The operation is
             *    locals := scale_factor*A*multipoles + beta*locals,   or   y := alpha*A**T*x + beta*y,
             *  if the settings is options::dense_ classical matrix vector product
             *  and if the settings is options::low_rank_ A = UV we perform two matrix vector product
             *
             * @tparam T
             * @tparam T2
             * @param multipoles the multipole values
             * @param locals  the local values
             * @param tmp a working array need for low rank approximation
             * @param knm the kernel matrix
             * @param scale_factor the scaling factor
             * @param acc if true we accumulate (beta=1.0 0 otherwise)
             */
            template<typename T, typename T2>
            inline auto product(T const& multipoles, T& locals, T2& tmp, k_tensor_type const& knm,
                                value_type scale_factor, bool acc) const -> void
            {
                if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    auto const& u = meta::get<0>(knm);
                    auto const& v = meta::get<1>(knm);
                    tensor::blas2_product(multipoles, tmp, v, value_type(1.0), false, true);
                    tensor::blas2_product(tmp, locals, u, scale_factor, acc, false);
                }
                else if constexpr(std::is_same_v<settings, options::dense_>)
                {
                    tensor::blas2_product(multipoles, locals, knm, scale_factor, acc);
                }
                else
                {
                    throw std::runtime_error("No implementation found for m2l product !");
                }
            }

            /**
             * @brief
             *
             * @tparam T
             * @param multipoles
             * @param locals
             * @param knm
             * @param scale_factor
             * @param acc
             */
            template<typename T>
            inline auto product(T const& multipoles, T& locals, k_tensor_type const& knm, value_type scale_factor,
                                bool acc) const -> void
            {
                if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    auto const& u = meta::get<0>(knm);
                    auto const& v = meta::get<1>(knm);
                    T tmp(locals.shape(), value_type(0.));
                    tensor::blas2_product(multipoles, tmp, v, value_type(1.0), false, true);
                    tensor::blas2_product(tmp, locals, u, scale_factor, acc, false);
                }
                else if constexpr(std::is_same_v<settings, options::dense_>)
                {
                    tensor::blas2_product(multipoles, locals, knm, scale_factor, acc);
                }
                else
                {
                    throw std::runtime_error("No implementation found for m2l product !");
                }
            }

            /**
             * @brief  Compute the matrix vector product for the M2L operator
             *
             * The operation is
             *    locals := scale_factor*A*multipoles + beta*locals,   or   y := alpha*A**T*x + beta*y,
             *  if the settings is options::dense_ classical matrix vector product
             *  and if the settings is options::low_rank_ A = UV we perform two matrix vector product
             *
             * @tparam T
             * @tparam T2
             * @param multipoles the multipole values
             * @param locals  the local values
             * @param tmp a working array need for low rank approximation
             * @param knm the kernel matrix
             * @param tmp A temporary matrix used in low rank approximation
             * @param nb_mult the number of multipole to treat (number of column of multipoles)
             * @param scale_factor the scaling factor
             * @param acc if true we accumulate (beta=1.0 0 otherwise)
             */
            template<typename T, typename T1>
            inline auto product_m(T const& multipoles, T& locals, k_tensor_type const& Knm, T1& tmp, int nb_mult,
                                  value_type scale_factor, bool acc) const -> void
            {
                if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    //  Knm = U V^T
                    auto const& U = meta::get<0>(Knm);
                    auto const& V = meta::get<1>(Knm);
                    constexpr bool accumulate = false;

                    constexpr bool transposeV = true;

                    tensor::blas3_product(multipoles, tmp, V, nb_mult, value_type(1.0), accumulate, transposeV);

                    tensor::blas3_product(tmp, locals, U, nb_mult, scale_factor, acc);
                }
                else if constexpr(std::is_same_v<settings, options::dense_>)
                {
                    tensor::blas3_product(multipoles, locals, Knm, nb_mult, scale_factor, acc);
                }
                else
                {
                    throw std::runtime_error("No implementation found for m2l product !");
                }
            }

            /**
             * @brief
             *
             * @tparam Cell
             * @param source_cell
             * @param target_cell
             * @param neighbor_idx
             * @param tree_level
             * @param products
             * @param thread_id
             */
            template<typename Cell>
            auto apply_m2l_single(Cell const& source_cell, Cell& target_cell, std::size_t neighbor_idx,
                                  std::size_t tree_level, [[maybe_unused]] buffer_type& products,
                                  [[maybe_unused]] size_type thread_id = 0) const -> void
            {
                std::size_t level{};
                scale_factor_type scale_factor{};

                if constexpr(homogeneity_tag == matrix_kernels::homogeneity::homogenous)
                {
                    level = 0;
                    // here we scale the target cell width to (homogenous case) match
                    // the [-1,1] unitary cell used to generate the corresponding
                    // interaction matrix.
                    scale_factor = m_far_field.scale_factor((target_cell.width()) / value_type(2.));
                }
                else   // non-homogenous case.
                {
                    // root level is 0 (i.e the simulation box) first level of cell is level 1
                    // we start the indexing of matrixes at 0, hence the -1 on the cell level
                    level = tree_level - 2;
                    scale_factor.fill(1.0);
                }
                if constexpr(symmetry_support)
                {
                    // the kernel is symmetric
                    // Decrease the number of matrix-vector product due to the symmetry
                    // see https://hal.inria.fr/hal-00746089v2
                    const auto neighbor_sym = this->symmetry_k_index(neighbor_idx);
                    const auto number_of_interactions = number_of_matrices_in_orthant<dimension>();

                    auto const& k = m_interactions_matrices.at(level * number_of_interactions + neighbor_sym);
                    auto& permuted_multipoles = products.at(0);
                    auto& permuted_locals = products.at(1);

                    for(std::size_t n = 0; n < kn; ++n)
                    {
                        for(std::size_t m = 0; m < km; ++m)
                        {
                            auto const& multipoles = source_cell.cmultipoles(m);
                            auto& locals = target_cell.locals(n);
                            // the multipole is permuted
                            const auto m_ptr = multipoles.data();
                            const auto l_ptr = locals.data();
                            const auto m_p_ptr = permuted_multipoles.data();
                            const auto l_p_ptr = permuted_locals.data();
                            const auto perm_ptr = &m_sym_permutations.at(neighbor_idx, 0);

                            for(std::size_t i{0}; i < m_nnodes; ++i)
                            {
                                m_p_ptr[perm_ptr[i]] = m_ptr[i];
                            }

                            this->product(permuted_multipoles, permuted_locals, k.at(n, m), scale_factor.at(n), false);

                            // the local expansion is permuted to the original.
                            for(std::size_t i{0}; i < m_nnodes; ++i)
                            {
                                l_ptr[i] += l_p_ptr[perm_ptr[i]];
                            }
                        }
                    }
                }
                else if constexpr(std::is_same_v<settings, options::low_rank_> ||
                                  std::is_same_v<settings, options::dense_>)
                {
                    // non symmetric kernel and low rank or dense approximation of the kernel

                    auto const& k = m_interactions_matrices.at(level * m_m2l_interactions + neighbor_idx);
                    auto const& multipoles = source_cell.cmultipoles();
                    auto& locals = target_cell.locals();   // we generate km*kn products
                    // loop on km
                    for(std::size_t m = 0; m < km; ++m)
                    {
                        // meta loop on kn
                        for(std::size_t n = 0; n < kn; ++n)
                        {
                            this->product(multipoles.at(m), locals.at(n), k.at(n, m), scale_factor.at(n), true);
                        }
                    }
                }
                else
                {
                    // non symmetric kernel and specific product (fft for uniform approximation)
                    //                    auto const& multipoles = source_cell.cmultipoles();
                    //                    auto& locals = target_cell.locals();
                    auto const& k = m_interactions_matrices.at(level * m_m2l_interactions + neighbor_idx);
                    // we generate km*kn products
                    // loop on km
                    for(std::size_t m = 0; m < km; ++m)
                    {
                        // meta loop on kn
                        for(std::size_t n = 0; n < kn; ++n)
                        {
                            this->derived_cast().apply_m2l_impl(source_cell, target_cell, products, k, scale_factor, n,
                                                                m, thread_id);
                        }
                    }
                }
            }

            /**
             * @brief
             *
             * @tparam Cell
             * @param target_cell
             * @param tree_level
             * @param products
             * @param thread_id
             * @return auto
             */
            template<typename Cell>
            auto apply_m2l_loop(Cell& target_cell, std::size_t tree_level, [[maybe_unused]] buffer_type& products,
                                [[maybe_unused]] size_type thread_id = 0) const
            {
                std::size_t level{};
                scale_factor_type scale_factor{};
                //
                if constexpr(homogeneity_tag == matrix_kernels::homogeneity::homogenous)
                {
                    level = 0;
                    // here we scale the target cell width to (homogenous case) match
                    // the [-1,1] unitary cell used to generate the corresponding
                    // interaction matrix.
                    scale_factor = m_far_field.scale_factor((target_cell.width()) / value_type(2.));
                }
                else   // non-homogenous case.
                {
                    // root level is 0 (i.e the simulation box) first level of cell is level 1
                    // we start the indexing of matrixes at 0, hence the -1 on the cell level
                    level = tree_level - 2;
                    scale_factor.fill(1.0);
                }
                auto const& cell_symbolics = target_cell.csymbolics();
                auto const& interaction_positions = cell_symbolics.interaction_positions;
                auto const& interaction_iterators = cell_symbolics.interaction_iterators;
                // #ifdef M2L_NEW_SYM
                // for symmetry support inner_type of the buffer for low_rank approximation otherwise the type of the
                // local approximation
                using local_type = std::conditional_t<std::is_same_v<settings, options::low_rank_>,
                                                      std::conditional_t<symmetry_support, buffer_inner_type,
                                                                         std::decay_t<decltype(target_cell.locals(0))>>,
                                                      meta::empty>;
                // #else
                //                 using local_type = std::decay_t<decltype(target_cell.locals(0))>;
                // #endif
                local_type work;

                if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    // //
                    if constexpr(symmetry_support)
                    {
                        // rank x max_symm
                        work.resize(products.at(1).shape());
                        // work.resize(std::array<std::size_t, 2>{m_nnodes, largest_number_permutation<dimension>()});
                    }
                    else
                    {
                        work.resize(target_cell.locals(0).shape());
                    }
                }
                //
                // Selection product operator depending on symmetry, product optimization
                //
                if constexpr(symmetry_support)
                {
                    // the kernel is symmetric
                    // Decrease the number of matrix-vector product due to the symmetry
                    // see https://hal.inria.fr/hal-00746089v2

                    constexpr auto number_of_symmetries = number_of_matrices_in_orthant<dimension>();

                    std::array<std::array<int, largest_number_permutation<dimension>()>, number_of_symmetries>
                      neighbors_perm;
                    std::array<int, number_of_symmetries> number_of_permutation{};
                    //
                    // access the buffer to store temporary array
                    //  aggregate_multipoles and  aggregate_locals are vectors of
                    //  xt::xarray of size number of permutation
                    auto aggregate_multipoles = products.at(0);
                    auto aggregate_locals = products.at(1);
                    // set cells in neighbors_perm according to their symmetry
                    for(std::size_t index{0}; index < cell_symbolics.existing_neighbors; ++index)
                    {
                        auto const& neighbor_idx = interaction_positions.at(index);
                        const auto neighbor_sym = this->symmetry_k_index(neighbor_idx);
                        //
                        neighbors_perm[neighbor_sym][number_of_permutation[neighbor_sym]] = index;
                        number_of_permutation[neighbor_sym] += 1;
                    }

                    // Loop on the number of rhs (input)
                    for(std::size_t m = 0; m < km; ++m)
                    {
                        // loop on the number of symmetries
                        for(std::size_t idx{0}; idx < number_of_symmetries; ++idx)
                        {
                            if(number_of_permutation[idx] > 0)
                            {
                                auto const& neighs = neighbors_perm[idx];
                                // Fill the temporary array with all multipoles
                                for(int i = 0; i < number_of_permutation[idx]; ++i)
                                {
                                    auto const& index = neighs[i];
                                    auto const& neighbor_idx = interaction_positions.at(index);
                                    // get the permutation associated to the position of the cell
                                    const auto perm_ptr = &m_sym_permutations.at(neighbor_idx, 0);
                                    //
                                    // Gather the multipoles for the permutation
                                    // get the set of multipole for the current symmetry
                                    // get the xt::xarray of aggregated multipole for the current symmetry
                                    // permuted_multipoles is a xarray of size(max cells in symmetry, the grid of
                                    // nodes)
                                    //   the order of the tensor is dimension + 1
                                    // get multipole
                                    auto const& source_cell = *interaction_iterators.at(index);
                                    auto const& multipoles = source_cell.cmultipoles(m);
                                    // get the pointer of the source multipole
                                    const auto m_ptr = multipoles.data();

                                    for(std::size_t j{0}; j < m_nnodes; ++j)
                                    {
                                        aggregate_multipoles(perm_ptr[j], i) = m_ptr[j];
                                        // current_permuted_multipole_ptr[perm_ptr[j]] = m_ptr[j];
                                    }
                                }   // end of the aggregation for the current symmetry idx

                                //////////////////////////////////////////////////////////////
                                // Performed the matrix matrix  product
                                //
                                auto const& k = m_interactions_matrices.at(level * number_of_symmetries + idx);

                                for(std::size_t n = 0; n < kn; ++n)
                                {
                                    auto& locals = target_cell.locals(n);
                                    const auto l_ptr = locals.data();
                                    // Perform matrix-matrix product with aggregate multipoles
                                    this->product_m(aggregate_multipoles, aggregate_locals, k.at(n, m), work,
                                                    number_of_permutation[idx], scale_factor.at(n), false);
                                    //

                                    // the local expansion is permuted to their original.
                                    for(int i = 0; i < number_of_permutation[idx]; ++i)
                                    {
                                        auto const& neighbor_idx = interaction_positions.at(neighs[i]);
                                        // get the permutation associated to the position of the cell
                                        const auto perm_ptr = &m_sym_permutations.at(neighbor_idx, 0);
                                        // add current multipole  to aggregate_multipoles
                                        // auto current_permuted_locals = aggregate_locals.data() + m_nnodes * i;
                                        for(std::size_t j{0}; j < m_nnodes; ++j)
                                        {
                                            l_ptr[j] += aggregate_locals(perm_ptr[j], i);
                                        }
                                    }

                                }   // end kn loop on the output of the matrix kernel
                            }   // end if
                        }   // end loop number of cells in the symmetry
                    }   // end km loop on the input of the matrix kernel

                }   // end constexpr
                else if constexpr(std::is_same_v<settings, options::low_rank_> ||
                                  std::is_same_v<settings, options::dense_>)
                {
                    // non symmetric kernel and low rank or dense approximation of the kernel

                    for(std::size_t index{0}; index < cell_symbolics.existing_neighbors; ++index)
                    {
                        auto const& source_cell = *interaction_iterators.at(index);
                        const auto neighbor_idx = static_cast<std::size_t>(interaction_positions.at(index));

                        auto const& k = m_interactions_matrices.at(level * m_m2l_interactions + neighbor_idx);
                        // we generate km*kn products
                        // loop on km
                        auto const& multipoles = source_cell.cmultipoles();
                        auto& locals = target_cell.locals();

                        for(std::size_t m = 0; m < km; ++m)
                        {
                            // meta loop on kn
                            for(std::size_t n = 0; n < kn; ++n)
                            {
                                this->product(multipoles.at(m), locals.at(n), work, k.at(n, m), scale_factor.at(n),
                                              true);
                            }
                        }
                    }
                }
                else
                {
                    // non symmetric kernel and specific product (fft for uniform approximation)

                    for(std::size_t index{0}; index < cell_symbolics.existing_neighbors; ++index)
                    {
                        auto const& source_cell = *interaction_iterators.at(index);
                        const auto neighbor_idx = static_cast<std::size_t>(interaction_positions.at(index));

                        auto const& k = m_interactions_matrices.at(level * m_m2l_interactions + neighbor_idx);
                        // we generate km*kn products
                        // loop on km
                        for(std::size_t m = 0; m < km; ++m)
                        {
                            // meta loop on kn
                            for(std::size_t n = 0; n < kn; ++n)
                            {
                                this->derived_cast().apply_m2l_impl(source_cell, target_cell, products, k, scale_factor,
                                                                    n, m, thread_id);
                            }
                        }
                    }
                }
            }

          private:
            /**
             * @brief
             *
             * @param order
             * @param width
             * @param tree_height
             */
            inline auto generate_interactions_matrices(size_type order, value_type width,
                                                       std::size_t tree_height) -> void
            {
                std::size_t number_of_level{1};
                std::size_t number_of_interactions{0};
                value_type local_cell_width_extension{0};

                if constexpr(symmetry_support)
                {
                    number_of_interactions = number_of_matrices_in_orthant<dimension>();
                }
                else
                {
                    number_of_interactions = this->m2l_interactions();
                }

                // here width is the root width
                // so first level of cell is of width because we skip level 0 (the root) and (the first level)):
                value_type current_width{width};
                const value_type half{0.5};
                const value_type quarter{0.25};
                // we get the half width to scale the roots

                if constexpr(homogeneity_tag == matrix_kernels::homogeneity::non_homogenous)
                {
                    // (tree_heigh = 4 -> [0-3]) the bottom cells and leaves have the same symbolic level !
                    // but we remove level 0 (the root ie simulation box) and level 1.
                    number_of_level = tree_height - 2;
                    current_width = width * quarter;
                    local_cell_width_extension = m_cell_width_extension;
                }

                value_type half_width{current_width * half};
                // we need to keep the tensorial view
                // let the same view goes down to tensorial of point
                // X and Y are Td tensors storing point.

                // get the ref of the vector
                auto& interactions_matrices{this->interactions_matrices()};

                // resizing and initializing the vector
                // homogenous -> only one level
                // non_homogenous -> we generate interaction matrices for each tree level processed by the m2l
                // operator
                interactions_matrices.resize(
                  number_of_interactions * number_of_level,
                  // xtensor_fixed<xshape<km,kn>>
                  interaction_matrix_type(typename interaction_matrix_type::shape_type{},
                                          // in Chebyshev, K_mn is a matrix of size [nnodes, nnodes]
                                          initialize_k(), xt::layout_type::row_major));
                // loop on levels
                for(std::size_t l{0}; l < number_of_level; ++l)
                {
                    // homogenous -> X is the [-1,1] box reference
                    // non_homogenous -> X is the [-cell_width_at_level/2, cell_width_at_level/2]
                    // X is a multidimensional grid generator returning a grid for X, another one for Y,...
                    // X is a multidimensional grid of points scaled on the size of cell
                    auto X_points = tensor::generate_grid_of_points<dimension>(
                      (half_width + half * local_cell_width_extension) * m_roots);

                    std::vector<std::pair<std::size_t, std::array<int, dimension>>> interactions;
                    std::size_t flat_idx{0};
                    auto collect_interactions = [&interactions, &flat_idx](auto... is)
                    {
                        if(((std::abs(is) > separation_criterion) || ...))
                        {
                            interactions.push_back({flat_idx, {static_cast<int>(is)...}});
                        }
                        ++flat_idx;
                    };

                    if constexpr(symmetry_support)
                    {
                        // we generate only the matrices in the positive cone of symmetries.
                        meta::looper_symmetries<dimension>{}(collect_interactions);
                    }
                    else
                    {
                        // loop range [-3,4[, ie range concept exclude the last value.
                        std::array<int, dimension> starts{};
                        std::array<int, dimension> stops{};
                        starts.fill(-3);
                        stops.fill(4);
                        // here we expand at compile time d loops of the range
                        // the indices of the d loops are input parameters of the lambda collect_interactions
                        meta::looper_range<dimension>{}(collect_interactions, starts, stops);
                    }

                    auto generate_interaction = [order, this, &interactions_matrices, &interactions, &X_points,
                                                 current_width, number_of_interactions,
                                                 l](std::size_t interaction, size_type thread_id)
                    {
                        auto const& [interaction_idx, is] = interactions.at(interaction);
                        // constructing centers to generate Y
                        container::point<value_type, dimension> center;
                        for(std::size_t d = 0; d < dimension; ++d)
                        {
                            center.at(d) = value_type(is.at(d)) * current_width;
                        }
                        xt::xarray<container::point<value_type, dimension>> centers(std::vector(dimension, order));
                        centers.fill(center);
                        // X_points is scaled with the width of cell so, Y_points will be scaled directly
                        xt::xarray<container::point<value_type, dimension>> Y_points = X_points + centers;

                        // we get the ref of interaction matrices to generate
                        auto& nm_fc_tensor = interactions_matrices.at(l * number_of_interactions + interaction_idx);
                        // and we generate each matrix needed for the product (kn*km matrices)
                        for(std::size_t n = 0; n < kn; ++n)
                        {
                            for(std::size_t m = 0; m < km; ++m)
                            {
                                nm_fc_tensor.at(n, m) =
                                  std::move(generate_matrix_k(X_points, Y_points, n, m, thread_id));
                            }
                        }
                    };

                    const auto n_interactions = static_cast<std::int64_t>(interactions.size());
                    if constexpr(std::is_same_v<settings, options::fft_>)
                    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                        for(std::int64_t interaction = 0; interaction < n_interactions; ++interaction)
                        {
#ifdef _OPENMP
                            // The FFT-based interpolators hold one FFT plan per thread.
                            const auto thread_id = static_cast<size_type>(omp_get_thread_num());
#else
                            const size_type thread_id{0};
#endif
                            generate_interaction(static_cast<std::size_t>(interaction), thread_id);
                        }
                    }
                    else
                    {
                        for(std::int64_t interaction = 0; interaction < n_interactions; ++interaction)
                        {
                            generate_interaction(static_cast<std::size_t>(interaction), 0);
                        }
                    }

                    // we divide the widths for the next tree level
                    current_width *= half;
                    half_width *= half;
                }
            }

          protected:
            /**
             * @brief Initialise the interaction matrices K.
             *
             * Homogenous : we construct only K on the points in [-1,1]^d cell, the length is
             * then 2. The matrix is applied on a cell of size width, then we have to scale
             * by cell_width/2.
             * Non-homogenous : The interaction matrices are constructed at the from root level
             * to the bottom of the tree (last cell level ie. same as the leaf level)
             * hence, we don't need the scale_factor of the kernel.
             *
             * @param order : number of therms of the polynomial approximation. order^d is the number of grid points.
             * @param root_cell_width : width of the top root cell needed for the non-homogenous case.
             * @param tree_height : hight of the tree.
             */
            inline auto initialize(size_type order, value_type root_cell_width, std::size_t tree_height) -> void
            {
                if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    m_weights_of_roots = generate_weights(m_order);
                }
                generate_interactions_matrices(
                  order,
                  (homogeneity_tag == matrix_kernels::homogeneity::non_homogenous) ? root_cell_width : value_type(2.),
                  tree_height);
            }

            /**
             * @brief
             *
             * @tparam D
             * @param order
             * @return std::enable_if_t<decltype(meta::sig_gen_w_f(std::declval<D>(), order))::value, array_type>
             */
            template<typename D = derived_type>
            [[nodiscard]] inline auto generate_weights(std::size_t order) const
              -> std::enable_if_t<decltype(meta::sig_gen_w_f(std::declval<D>(), order))::value, array_type>
            {
                return this->derived_cast().generate_weights_impl(order);
            }

            /**
             * @brief
             *
             * @tparam D
             * @param order
             * @return std::enable_if_t<!decltype(meta::sig_gen_w_f(std::declval<D>(), order))::value, array_type>
             */
            template<typename D = derived_type>
            [[nodiscard]] inline auto generate_weights(std::size_t order) const
              -> std::enable_if_t<!decltype(meta::sig_gen_w_f(std::declval<D>(), order))::value, array_type>
            {
                return array_type{};
            }

            /**
             * @brief
             *
             * @return derived_type&
             */
            [[nodiscard]] inline auto derived_cast() & noexcept -> derived_type&
            {
                return *static_cast<derived_type*>(this);
            }

            /**
             * @brief
             *
             * @return derived_type const&
             */
            [[nodiscard]] inline auto derived_cast() const& noexcept -> derived_type const&
            {
                return *static_cast<const derived_type*>(this);
            }

            /**
             * @brief
             *
             * @return derived_type
             */
            [[nodiscard]] inline auto derived_cast() && noexcept -> derived_type
            {
                return *static_cast<derived_type*>(this);
            }

          private:
            std::vector<interaction_matrix_type, XTENSOR_DEFAULT_ALLOCATOR(interaction_matrix_type)>
              m_interactions_matrices{};   ///<  The M2L matricies

            /**
             * @brief
             *
             */
            sym_permutations_type m_sym_permutations{};

            /**
             * @brief
             *
             */
            k_indices_type m_k_indices{};

            /**
             * @brief
             *
             */
            matrix_kernel_type m_far_field{};

            /**
             * @brief the weight associated to the roots of the m_order-1 polynomial
             *
             */
            xt::xarray<value_type> m_weights_of_roots{};

            /**
             * @brief
             *
             */
            const size_type m_m2l_interactions{};

            /**
             * @brief number of modes m_order^dimension
             *
             */
            const size_type m_nnodes{};

            /**
             * @brief number of terms of the expansion (1d)
             *
             */
            const size_type m_order{};

            /**
             * @brief the roots of the m_order-1 polynomial
             *
             */
            const array_type m_roots{};

            /**
             * @brief the accuracy for low-rank approximation (10^(-o)) @todo check 10^(1-o)
             *
             */
            const value_type m_epsilon{};

            /**
             * @brief width of the extension of the cell
             *
             */
            const value_type m_cell_width_extension{};
        };
    }   // namespace impl
}   // namespace scalfmm::interpolation
#endif   // SCALFMM_INTERPOLATION_M2L_HANDLER_HPP
