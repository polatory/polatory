// --------------------------------
// See LICENCE file at project root
// File : scalfmm/interpolation/permutations.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_PERMUTATIONS_HPP
#define SCALFMM_INTERPOLATION_PERMUTATIONS_HPP

#include "scalfmm/meta/utils.hpp"
#include "scalfmm/utils/io_helpers.hpp"
#include "scalfmm/utils/math.hpp"

#include "xtensor/containers/xarray.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor/generators/xgenerator.hpp"
#include "xtensor/core/xmath.hpp"
#include "xtensor/core/xoperation.hpp"
#include "xtensor/core/xtensor_forward.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace scalfmm::meta
{
    /**
     * @brief The looper symmetries extends loop nests according to the dimension.
     * Indices correspond the symmetries of the positive orthant.
     * A cone in 3D
     * A triangle in 2D
     * A segment in 1D
     *
     * @tparam N the dimension.
     */
    template<std::size_t N>
    struct looper_symmetries
    {
        static_assert(N < 4 && N > 0, "Interpolation with symmetries : The loop nest for the required dimension is not "
                                      "available. Please provide it !");
    };

    /**
     * @brief
     *
     * @tparam
     */
    template<>
    struct looper_symmetries<3>
    {
        template<typename F>
        constexpr inline auto operator()(F&& f)
        {
            for(int i = 2; i <= 3; ++i)
            {
                for(int j = 0; j <= i; ++j)
                {
                    for(int k = 0; k <= j; ++k)
                    {
                        f(i, j, k);
                    }
                }
            }
        }
    };

    /**
     * @brief
     *
     * @tparam
     */
    template<>
    struct looper_symmetries<2>
    {
        template<typename F>
        constexpr inline auto operator()(F&& f)
        {
            for(int i = 2; i <= 3; ++i)
            {
                for(int j = 0; j <= i; ++j)
                {
                    f(i, j);
                }
            }
        }
    };

    /**
     * @brief
     *
     * @tparam
     */
    template<>
    struct looper_symmetries<1>
    {
        template<typename F>
        constexpr inline auto operator()(F&& f)
        {
            for(int i = 2; i <= 3; ++i)
            {
                f(i);
            }
        }
    };
}   // namespace scalfmm::meta

namespace scalfmm::interpolation
{
    namespace impl
    {
        /**
         * @brief Get the orthant index object
         *
         * @tparam Indexes
         * @tparam Is
         * @param s
         * @param is
         * @return constexpr auto
         */
        template<typename... Indexes, std::size_t... Is>
        inline constexpr auto get_orthant_index(std::index_sequence<Is...> s, Indexes... is)
        {
            // here, we check the sign of each indices and shift it if negative
            // resulting to 0 if indices are all positive and 7 (in 3D) if all negative.
            return (... | (std::size_t(std::signbit(static_cast<double>(is))) << Is));
        }
    }   // namespace impl

    /**
     * @brief Get the orthant index object according to the box position.
     *
     * @tparam Indexes the type of the position indices.
     * @param is the position indices.
     * @return the orthant index.
     */
    template<typename... Indexes>
    inline constexpr auto
    get_orthant_index(Indexes... is) -> std::enable_if_t<std::conjunction_v<std::is_signed<Indexes>...>, std::size_t>
    {
        return impl::get_orthant_index(std::index_sequence_for<Indexes...>{}, is...);
    }

    /**
     * @brief Returns the number of interactions matrices K in the positive orthant.
     *
     * @tparam dim the dimension.
     * @param check
     * @return the number of matrices.
     */
    template<std::size_t dim>
    inline constexpr auto number_of_matrices_in_orthant(bool check = true) -> std::size_t
    {
        if constexpr(dim == 1)
        {
            return 2;
        }
        else if constexpr(dim == 2)
        {
            return 7;
        }
        else if constexpr(dim == 3)
        {
            return 16;
        }
        else
        {
            return 1;
            // if(check)
            // {
            //     static_assert(
            //       dim < 4 && dim > 0,
            //       "Interpolation with symmetries : the number of interaction matrices in the orthant is not "
            //       "available. Please provide it !");
            // }
        }
    }

    /**
     * @brief Returns the maximum number of identical permutations in the grid \f$ [-3,3]^d \f$.
     *
     * @tparam dim the dimension.
     * @return the maximum number of same permutation.
     */
    template<std::size_t dim>
    inline constexpr auto largest_number_permutation() -> std::size_t
    {
        if constexpr(dim == 1)
        {
            return 2;
        }
        else if constexpr(dim == 2)
        {
            return 8;
        }
        else if constexpr(dim == 3)
        {
            return 24;
        }
        else
        {
            static_assert(dim < 4 && dim > 0,
                          "Interpolation with symmetries : the number of interaction matrices in the orthant is not "
                          "available. Please provide it !");
        }
    }

    /**
     * @brief Computes the cone index corresponding the position indices and
     * also permutes the position indices to correspond to the indices in the
     * optimized cone.
     *
     * @tparam Integral the type of the position indices of the interaction boxes.
     * @param is the position indices of the interaction boxes.
     * @return a tuple with the cone index and the permuted indices.
     */
    template<typename... Integral>
    inline auto compute_cone_and_permuted_indexe(Integral... is)
    {
        constexpr auto dimension{sizeof...(Integral)};
        constexpr auto number_of_orthant{math::pow(2, dimension)};
        // absolute values of the position indices.
        std::array<std::size_t, dimension> u = {std::size_t(std::abs(is))...};
        // permuted indices
        std::array<std::size_t, dimension> is_permuted{};
        // the array storing the permutation for the position indices
        xt::xarray<int> perms(std::vector<std::size_t>({number_of_orthant, dimension}));
        std::size_t cidx{0};

        if constexpr(dimension == 1)
        {
            // permutation of interaction indices (already absolute value)
            cidx = 0;
            is_permuted.at(0) = u.at(0);
        }
        else if constexpr(dimension == 2)
        {
            // permutation of interaction indices (already absolute value)
            perms.at(0, 0) = 0;
            perms.at(0, 1) = 1;
            perms.at(1, 0) = 1;
            perms.at(1, 1) = 0;

            const std::size_t q0 = (u.at(1) > u.at(0));
            cidx = q0;

            is_permuted.at(0) = u.at(perms.at(cidx, 0));
            is_permuted.at(1) = u.at(perms.at(cidx, 1));
        }
        else if constexpr(dimension == 3)
        {
            // permutation of interaction indices (already absolute value)
            perms.at(0, 0) = 0;
            perms.at(0, 1) = 1;
            perms.at(0, 2) = 2;
            perms.at(4, 0) = 0;
            perms.at(4, 1) = 2;
            perms.at(4, 2) = 1;
            perms.at(6, 0) = 2;
            perms.at(6, 1) = 0;
            perms.at(6, 2) = 1;
            perms.at(1, 0) = 1;
            perms.at(1, 1) = 0;
            perms.at(1, 2) = 2;
            perms.at(3, 0) = 1;
            perms.at(3, 1) = 2;
            perms.at(3, 2) = 0;
            perms.at(7, 0) = 2;
            perms.at(7, 1) = 1;
            perms.at(7, 2) = 0;

            const std::size_t q2 = (u.at(2) > u.at(1)) << 2;
            const std::size_t q1 = (u.at(2) > u.at(0)) << 1;
            const std::size_t q0 = (u.at(1) > u.at(0));
            cidx = (q2 | q1 | q0);

            is_permuted.at(0) = u.at(perms.at(cidx, 0));
            is_permuted.at(1) = u.at(perms.at(cidx, 1));
            is_permuted.at(2) = u.at(perms.at(cidx, 2));
        }
        else
        {
            static_assert(dimension < 4 && dimension > 0,
                          "Interpolation with symmetries : calculation of the cone index is not "
                          "available. Please provide it !");
        }
        return std::make_tuple(cidx, is_permuted);
    }

    /**
     * @brief Returns a tuple holding an xarray of permutations corresponding to all m2l
     * interactions and a std::vector the corresponding K linear index in the interaction
     * matrix vector of Ks.
     *
     * @tparam Dimension : the dimension.
     * @param order : the order of the interpolator.
     * @param nnodes : the size of the tensors.
     * @param m2l_interactions : the number of interactions.
     * @return a tuple holding the xarray of permutations and the std::vector of K indices.
     */
    template<std::size_t Dimension>
    inline auto get_permutations_and_indices(std::size_t order, std::size_t nnodes, std::size_t m2l_interactions)
    {
        using permutations_type = xt::xarray<int>;
        using k_indices_type = std::vector<std::size_t>;

        static constexpr std::size_t dimension = Dimension;

        constexpr auto number_of_orthant{math::pow(2, dimension)};
        std::size_t flat_index{0};

        // array of global permutations
        permutations_type globlal_permutations(
          std::vector<std::size_t>({number_of_orthant * number_of_orthant, nnodes}));
        // the array holding all the permutations for all the interactions
        permutations_type permutations(std::vector<std::size_t>({m2l_interactions, nnodes}));
        // the std::vector holding the index of K in interaction matrix vector.
        k_indices_type k_indices(m2l_interactions);
        // permutations for 8 quadrants
        permutations_type quads(std::vector<std::size_t>({number_of_orthant, nnodes}));
        // permutations for 6 cones in quadrant (+++), 2 and 5 do not exist
        permutations_type cones(std::vector<std::size_t>({number_of_orthant, nnodes}));
        // set quads and cones permutations
        xt::xarray<std::size_t> evn = xt::arange(std::size_t(0), order, 1);
        xt::xarray<std::size_t> odd = xt::cast<std::size_t>(xt::abs(xt::arange(-(int(order) - 1), 1, 1)));
        xt::xarray<std::size_t> ref_k_indexes(std::vector(dimension, std::size_t(4)),
                                              std::size_t(number_of_matrices_in_orthant<dimension>()));

        // here, we build the linear index of K corresponding to the loop in the cone
        auto generate_reference_k_idx = [&ref_k_indexes, &flat_index](auto... is)
        { ref_k_indexes.at(is...) = flat_index++; };
        meta::looper_symmetries<dimension>{}(generate_reference_k_idx);
        // std::cout << "ref_k_indexes " << ref_k_indexes << std::endl;

        // we construct global permutation depending on the dimension
        if constexpr(dimension == 1)
        {
            for(std::size_t i = 0; i < order; ++i)
            {
                quads.at(0, i) = evn.at(i);   // - - -
                quads.at(1, i) = odd.at(i);   // + - -
            }
            for(std::size_t q = 0; q < 2; ++q)
            {
                for(std::size_t n = 0; n < nnodes; ++n)
                {
                    globlal_permutations.at(q * 2, n) = quads.at(q, n);
                }
            }
        }
        else if constexpr(dimension == 2)
        {
            for(std::size_t i = 0; i < order; ++i)
            {
                for(std::size_t j = 0; j < order; ++j)
                {
                    const std::size_t index = i * order + j;

                    // global axis parallel symmetries (8 quads) ///////////      // k j i
                    quads.at(0, index) = evn.at(i) * order + evn.at(j);   // - - -
                    quads.at(2, index) = evn.at(i) * order + odd.at(j);   // - + -
                    quads.at(1, index) = odd.at(i) * order + evn.at(j);   // + - -
                    quads.at(3, index) = odd.at(i) * order + odd.at(j);   // + + -

                    // diagonal symmetries (j>i)(k>i)(k>j) /////////////
                    cones.at(0, index) = evn.at(i) * order + evn.at(j);   // (0) 000
                    cones.at(1, index) = evn.at(j) * order + evn.at(i);   // (1) 001
                }
            }
            // set 48 global permutations (combinations of 4 quadrants and 2 cones respectively)
            for(std::size_t q = 0; q < 4; ++q)
            {
                for(std::size_t c = 0; c < 2; ++c)
                {
                    for(std::size_t n = 0; n < nnodes; ++n)
                    {
                        globlal_permutations.at((q * 4 + c), n) = cones.at(c, quads.at(q, n));
                    }
                }
            }
        }
        else if constexpr(dimension == 3)
        {
            for(std::size_t i = 0; i < order; ++i)
            {
                for(std::size_t j = 0; j < order; ++j)
                {
                    for(std::size_t k = 0; k < order; ++k)
                    {
                        const std::size_t index = i * order * order + j * order + k;

                        // global axis parallel symmetries (8 quads) ///////////      // k j i
                        quads.at(0, index) = evn.at(i) * order * order + evn.at(j) * order + evn.at(k);   // - - -
                        quads.at(4, index) = evn.at(i) * order * order + evn.at(j) * order + odd.at(k);   // - - +
                        quads.at(2, index) = evn.at(i) * order * order + odd.at(j) * order + evn.at(k);   // - + -
                        quads.at(6, index) = evn.at(i) * order * order + odd.at(j) * order + odd.at(k);   // - + +
                        quads.at(1, index) = odd.at(i) * order * order + evn.at(j) * order + evn.at(k);   // + - -
                        quads.at(5, index) = odd.at(i) * order * order + evn.at(j) * order + odd.at(k);   // + - +
                        quads.at(3, index) = odd.at(i) * order * order + odd.at(j) * order + evn.at(k);   // + + -
                        quads.at(7, index) = odd.at(i) * order * order + odd.at(j) * order + odd.at(k);   // + + +

                        // diagonal symmetries (j>i)(k>i)(k>j) /////////////////////
                        cones.at(0, index) = evn.at(i) * order * order + evn.at(j) * order + evn.at(k);   // (0) 000
                        cones.at(1, index) = evn.at(j) * order * order + evn.at(i) * order + evn.at(k);   // (1) 001
                        // cones.at(2) does not exist
                        cones.at(3, index) = evn.at(j) * order * order + evn.at(k) * order + evn.at(i);   // (3) 011
                        cones.at(4, index) = evn.at(i) * order * order + evn.at(k) * order + evn.at(j);   // (4) 100
                        // cones.at(5) does not exist
                        cones.at(6, index) = evn.at(k) * order * order + evn.at(i) * order + evn.at(j);   // (6) 110
                        cones.at(7, index) = evn.at(k) * order * order + evn.at(j) * order + evn.at(i);   // (7) 111
                    }
                }
            }
            // set 48 global permutations (combinations of 8 quadrants and 6 cones respectively)
            for(std::size_t q = 0; q < 8; ++q)
            {
                for(std::size_t c = 0; c < 8; ++c)
                {
                    if(c != 2 && c != 5)
                    {   // cone 2 and 5 do not exist
                        for(std::size_t n = 0; n < nnodes; ++n)
                        {
                            globlal_permutations.at((q * 8 + c), n) = cones.at(c, quads.at(q, n));
                        }
                    }
                }
            }
        }
        else
        {
            static_assert(dimension < 4 && dimension > 0,
                          "Chebyshev interpolation with symmetries : the number of cones is not "
                          "available. Please provide it !");
        }

        // lambda to get back the linear index of K corresponding to the indices in the positive cone
        auto get_k_index = [&ref_k_indexes](std::array<std::size_t, dimension> is_permuted) constexpr
        {
            if constexpr(dimension == 1)
            {
                return ref_k_indexes.at(is_permuted.at(0));
            }
            else if constexpr(dimension == 2)
            {
                return ref_k_indexes.at(is_permuted.at(0), is_permuted.at(1));
            }
            else if constexpr(dimension == 3)
            {
                return ref_k_indexes.at(is_permuted.at(0), is_permuted.at(1), is_permuted.at(2));
            }
        };

        flat_index = 0;
        // the lambda to construct the vector of permutations and the vector of K indices
        auto construct_permutations_and_indices =
          [&permutations, &globlal_permutations, nnodes, &k_indices, &flat_index, &get_k_index](auto... is)
        {
            // for each position of box
            // we get the orthant index,
            std::size_t qidx{get_orthant_index(is...)};
            std::size_t cidx{};
            std::array<std::size_t, dimension> permuted_indices{};
            // then, we get the cone index and the corresponding box position in the positive orthant
            std::tie(cidx, permuted_indices) = compute_cone_and_permuted_indexe(is...);
            // we retrieve the linear K index with the position of the box in the positive orthant
            // and store it at the interaction index
            k_indices.at(flat_index) = get_k_index(permuted_indices);

            for(std::size_t i{0}; i < nnodes; i++)
            {
                // we add the permutation at the corresponding interaction index.
                permutations.at(flat_index, i) = globlal_permutations.at(qidx * number_of_orthant + cidx, i);
            }
            flat_index++;
        };
        std::array<int, dimension> starts{};
        std::array<int, dimension> stops{};
        starts.fill(-3);
        stops.fill(4);
        // here we expand at compile time d loops of the range
        // the indices of the d loops are input parameters of the lambda
        meta::looper_range<dimension>{}(construct_permutations_and_indices, starts, stops);
        // io::print("k_indices: ", k_indices);
        // std::array<int, number_of_matrices_in_orthant<dimension>() + 1> num{};
        // for(auto& e: k_indices)
        // {
        //     num[e]++;
        // }
        // io::print(std::cout, "num: ", num, ", ");
        return std::make_tuple(std::move(permutations), std::move(k_indices));
    }
}   // namespace scalfmm::interpolation
#endif   // SCALFMM_INTERPOLATION_CHEBYSHEV_PERMUTATIONS_HPP
