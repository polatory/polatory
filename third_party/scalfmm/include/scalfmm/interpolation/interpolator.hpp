// --------------------------------
// See LICENCE file at project root
// File : scalfmm/interpolation/interpolator.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_INTERPOLATOR_HPP
#define SCALFMM_INTERPOLATION_INTERPOLATOR_HPP

#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/builders.hpp"
#include "scalfmm/interpolation/mapping.hpp"
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

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/core/xtensor_config.hpp"
#include "xtensor/core/xvectorize.hpp"

#include <algorithm>
#include <any>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
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
         * @tparam Derived
         * @tparam Enable
         */
        template<typename Derived, typename Enable = void>
        class interpolator
        {
            static_assert(std::is_floating_point_v<typename interpolator_traits<Derived>::value_type>,
                          "interpolator ValueType needs to be floating point.");
        };

        /**
         * @brief
         *
         * @tparam Derived
         */
        template<typename Derived>
        struct alignas(XTENSOR_FIXED_ALIGN) interpolator<
          Derived,
          typename std::enable_if_t<std::is_floating_point_v<typename interpolator_traits<Derived>::value_type>>>
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
            using value_type = typename interpolator_traits<Derived>::value_type;
            using size_type = std::size_t;

            /**
             * @brief
             *
             */
            static constexpr std::size_t dimension = interpolator_traits<Derived>::dimension;

            using array_type = xt::xarray<value_type>;
            using array_shape_type = typename array_type::shape_type;

            template<std::size_t d>
            using tensor_type = xt::xtensor<value_type, d>;
            template<std::size_t d>
            using tensor_shape_type = typename tensor_type<d>::shape_type;
            using grid_permutations_type = std::conditional_t<(dimension > 3), empty, xt::xtensor<std::size_t, 2>>;
            using settings = typename interpolator_traits<derived_type>::settings;

            /**
             * @brief Construct a new interpolator object
             *
             */
            interpolator() = delete;

            /**
             * @brief Construct a new interpolator object
             *
             * @param other
             */
            interpolator(interpolator const& other) = delete;

            /**
             * @brief Construct a new interpolator object
             *
             */
            interpolator(interpolator&&) noexcept = delete;

            /**
             * @brief
             *
             * @return interpolator&
             */
            auto operator=(interpolator const&) -> interpolator& = delete;

            /**
             * @brief
             *
             * @return interpolator&
             */
            auto operator=(interpolator&&) noexcept -> interpolator& = delete;

            /**
             * @brief Destroy the interpolator object
             *
             */
            ~interpolator() = default;

            /**
             * @brief constructor of the interpolator
             *
             * @param order number of terms of the 1d expansion ()
             * @param tree_height height of the tree
             * @param root_cell_width width of the simulation box
             * @param cell_width_extension  width of the extension of the cell
             * @param late_init if true the initialization is done later
             */
            interpolator(size_type order, size_type tree_height, value_type root_cell_width,
                         value_type cell_width_extension, bool late_init = false)
              : m_order(order)
              , m_nnodes(meta::pow(order, dimension))
              , m_tree_height(tree_height)
              , m_root_cell_width(root_cell_width)
              , m_cell_width_extension(cell_width_extension)
              , m_use_cell_width_extension(cell_width_extension > 0 ? true : false)
              , m_grid_permutations(this->generate_grid_permutations(order))
            {
                if(late_init == false)
                {
                    this->initialize();
                }
            }

            /**
             * @brief Construct a new interpolator object
             *
             * @param order number of terms of the 1d expansion ()
             * @param tree_height height of the tree
             * @param root_cell_width width of the simulation box
             * @param late_init if true the initialization is done later
             */
            interpolator(size_type order, size_type tree_height, value_type root_cell_width, bool late_init = false)
              : interpolator(order, tree_height, root_cell_width, value_type(0.), late_init)
            {
            }

            /**
             * @brief return the number of 1d points (number of terms in the 1d-expansion)
             *
             * @return std::size_t
             */
            [[nodiscard]] inline auto order() const noexcept -> std::size_t { return m_order; }

            /**
             * @brief return the number of points in de dimension grid
             *
             * @return auto
             */
            [[nodiscard]] inline auto nnodes() const noexcept { return m_nnodes; }

            /**
             * @brief get the cell extension
             *
             * @return value_type
             */
            [[nodiscard]] inline auto cell_width_extension() const noexcept -> value_type
            {
                return m_cell_width_extension;
            }

            /**
             * @brief tell is we use the cell extension
             *
             * @return true
             * @return false
             */
            [[nodiscard]] inline auto use_cell_width_extension() const noexcept -> bool
            {
                return m_use_cell_width_extension;
            }

            /**
             * @brief
             *
             * @return array_type&
             */
            [[nodiscard]] inline auto interpolator_tensor() noexcept -> array_type&
            {
                return m_child_parent_interpolators;
            }

            /**
             * @brief
             *
             * @return array_type const&
             */
            [[nodiscard]] inline auto interpolator_tensor() const noexcept -> array_type const&
            {
                return m_child_parent_interpolators;
            }

            /**
             * @brief
             *
             * @return array_type const&
             */
            [[nodiscard]] inline auto cinterpolator_tensor() const noexcept -> array_type const&
            {
                return m_child_parent_interpolators;
            }

            /**
             * @brief return the roots of the polynomial
             *
             * @return array_type
             */
            [[nodiscard]] inline auto roots() const -> array_type { return this->derived_cast().roots_impl(); }

            /**
             * @brief
             *
             * @tparam ComputationType
             * @param x
             * @param n
             * @return ComputationType
             */
            template<typename ComputationType>
            [[nodiscard]] inline auto polynomials(ComputationType x, std::size_t n) const -> ComputationType
            {
                return this->derived_cast().polynomials_impl(x, n);
            }

            /**
             * @brief
             *
             * @tparam VectorType
             * @tparam ComputationType
             * @tparam Dim
             * @param all_poly
             * @param x
             * @param order
             */
            template<typename VectorType, typename ComputationType, std::size_t Dim>
            inline auto fill_all_polynomials(VectorType& all_poly, container::point<ComputationType, Dim> x,
                                             std::size_t order) const -> void
            {
                this->derived_cast().fill_all_polynomials_impl(all_poly, x, order);
            }

            /**
             * @brief
             *
             * @tparam ComputationType
             * @param x
             * @param n
             * @return auto
             */
            template<typename ComputationType>
            [[nodiscard]] inline auto derivative(ComputationType x, std::size_t n) const
            {
                return this->derived_cast().derivative_impl(x, n);
            }

            /**
             * @brief
             *
             * @return grid_permutations_type const&
             */
            [[nodiscard]] inline auto grid_permutations() const -> grid_permutations_type const&
            {
                return m_grid_permutations;
            }

            /**
             * @brief ompute he memory in bytes used by the interpolator
             *
             * @return the memory used by the interpolator
             */
            [[nodiscard]] inline auto memory_usage() const noexcept -> std::size_t const
            {
                return this->derived_cast().memory_usage();
            }
            [[nodiscard]] inline auto name() const noexcept -> std::string const { return this->derived_cast().name(); }

          private:
            /**
             * @brief This function pre-calculates all the child-parent interpolators required for M2M and L2L passes.
             *
             * This function pre-calculates all the child-parent interpolators required for M2M and L2L passes
             * for all levels that are not leaf levels. If there is no cell width extension, the same set of
             * child-parent interpolators is used for all levels. However, if there is a cell width extension, the
             * child-parent interpolators must be recalculated for each level because the ratio of "child cell width
             * over parent cell width" is not the same.
             *
             * @warning(homogeneity not yet supported when cell_width_extension > 0)
             *
             */
            auto set_child_parent_interpolator() -> void
            {
                using point_type = scalfmm::container::point<value_type, dimension>;

                // Get the number of child cells (depending on to the dimension)
                constexpr auto childs_number = meta::pow(2, dimension);

                // Set number of non-leaf ios that actually need to be computed
                // Maybe always 3 ?
                size_type reduced_tree_height{std::min(size_type{3}, m_tree_height)};   // = 2 + nb of computed nl ios

                // If there is no cell extension, the same set of child-parent interpolator is used for each level
                if(m_cell_width_extension > std::numeric_limits<value_type>::epsilon())
                {
                    reduced_tree_height = m_tree_height;   // compute 1 non-leaf io per level
                }

                // Init non-leaf interpolators
                constexpr value_type half{0.5};
                constexpr value_type two{2.};
                value_type cell_width{m_root_cell_width * half};   // at level 1
                cell_width *= half;                                // at level 2

                // Allocate memory for child-parent interpolator
                // expliquer les differents indices du tenseurs
                m_child_parent_interpolators.resize({reduced_tree_height, childs_number, dimension, m_order * m_order});

                // Generate 1-D ones combinations for relative childs positions
                std::vector<value_type> to_adapt(childs_number * dimension);
                using to_adapt_difference_type =
                  typename std::iterator_traits<typename std::vector<value_type>::iterator>::difference_type;

                for(std::size_t i = 0; i < childs_number; ++i)
                {
                    std::array<value_type, dimension> binary{};
                    auto dec{i};
                    auto count{dimension - 1};

                    while(dec > 0)
                    {
                        binary.at(count) = static_cast<value_type>(dec % 2);
                        dec /= 2;
                        count--;
                    }
                    std::replace_if(binary.begin(), binary.end(),
                                    std::bind(std::equal_to<value_type>(), std::placeholders::_1, value_type(0.)),
                                    value_type(-1.));
                    std::copy(binary.begin(), binary.end(),
                              to_adapt.begin() + static_cast<to_adapt_difference_type>(i * dimension));
                }

                const array_shape_type relative_childs_centers_shape({childs_number, dimension});
                array_type relative_childs_centers(relative_childs_centers_shape);

                // Adapting vector of relative positions
                relative_childs_centers = xt::adapt(to_adapt, relative_childs_centers_shape);

                // Get the 1D interpolation roots (on the reference interval [-1, 1])
                array_type inter_roots(roots());

                // Temporary array to store the interpolation roots in the child cells
                const array_shape_type child_coords_shape({dimension, m_order});
                array_type child_coords(child_coords_shape);

                // Loop on all levels that are not at leaf level
                for(std::size_t level = 2; level < reduced_tree_height; ++level)
                {
                    // Ratio of extended cell widths at the current level (definition: child ext / parent ext)
                    const value_type extended_cell_ratio =
                      (cell_width * half + m_cell_width_extension) / (cell_width + m_cell_width_extension);

                    // Child cell width
                    const value_type childs_width{two * extended_cell_ratio};

                    // Scale factor for the relative child centers (taking into account the extension)
                    const value_type scale_factor = value_type(1.) - extended_cell_ratio;

                    // Loop on the child cells
                    for(std::size_t child_index = 0; child_index < childs_number; ++child_index)
                    {
                        // Set relative child center (taking into account the extension)
                        point_type child_center{}, l_pos{}, g_pos{};
                        for(std::size_t d = 0; d < dimension; ++d)
                        {
                            child_center[d] = relative_childs_centers(child_index, d) * scale_factor;
                        }

                        // Set polynomial roots in the child cell (using a "local to global" mapping)
                        const map_loc_glob<point_type> mapper_lg(child_center, point_type(childs_width));

                        for(std::size_t o = 0; o < m_order; ++o)
                        {
                            l_pos = point_type(inter_roots[o]);
                            mapper_lg(l_pos, g_pos);
                            for(std::size_t d = 0; d < dimension; ++d)
                            {
                                child_coords(d, o) = g_pos[d];
                            }
                        }

                        // Assemble the corresponding interpolator
                        for(std::size_t o1 = 0; o1 < m_order; ++o1)
                        {
                            for(std::size_t o2 = 0; o2 < m_order; ++o2)
                            {
                                for(std::size_t d = 0; d < dimension; ++d)
                                {
                                    // ajouter une variable pour l'acces
                                    m_child_parent_interpolators(level, child_index, d, o1 * m_order + o2) =
                                      polynomials(child_coords(d, o1), o2);
                                }
                            }
                        }
                    }

                    // Update cell width for the next level
                    cell_width *= half;
                }
            }

            /**
             * @brief
             *
             * @param order
             * @return auto
             */
            inline auto generate_grid_permutations(std::size_t order)
            {
                grid_permutations_type perms{};

                if constexpr(dimension == 2)
                {
                    perms.resize({dimension, math::pow(order, dimension)});
                    std::size_t flat{0};
                    for(std::size_t i = 0; i < order; ++i)
                    {
                        for(std::size_t j = 0; j < order; ++j)
                        {
                            perms.at(0, flat) = i * order + j;
                            perms.at(1, flat) = j * order + i;
                            flat++;
                        }
                    }
                }
                else if constexpr(dimension == 3)
                {
                    perms.resize({dimension, math::pow(order, dimension)});
                    std::size_t flat{0};
                    for(std::size_t i = 0; i < order; ++i)
                    {
                        for(std::size_t j = 0; j < order; ++j)
                        {
                            for(std::size_t k = 0; k < order; ++k)
                            {
                                perms.at(0, flat) = i * order * order + j * order + k;
                                perms.at(1, flat) = k * order * order + i * order + j;
                                perms.at(2, flat) = j * order * order + k * order + i;
                                flat++;
                            }
                        }
                    }
                }
                return perms;
            }

          protected:
            /**
             * @brief
             *
             */
            inline auto initialize() -> void { set_child_parent_interpolator(); }

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
            const size_type m_order{};   ///< number of terms of the expansion (1d)

            const size_type m_nnodes{};   ///<  number of modes m_order^dimension

            const size_type m_tree_height{};   ///<  height of the tree

            const value_type m_root_cell_width{};   ///< width of the simulation box

            const value_type m_cell_width_extension{};   ///< width of the extension of the cell

            const bool m_use_cell_width_extension{};   ///< true if we use the cell extension

            array_type m_child_parent_interpolators{};   ///<

            /**
             * @brief
             *
             */
            grid_permutations_type m_grid_permutations{};
        };
    }   // namespace impl
}   // namespace scalfmm::interpolation

#endif   // SCALFMM_INTERPOLATION_UNIFORM_HPP
