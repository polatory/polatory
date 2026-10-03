// --------------------------------
// See LICENCE file at project root
// File : scalfmm/utils/generate.hpp
// --------------------------------

#ifndef SCALFMM_UTILS_GENERATE_HPP
#define SCALFMM_UTILS_GENERATE_HPP

#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/math.hpp"

#include <random>

namespace scalfmm::utils
{
    /**
     * @brief
     *
     * @tparam ContainerType
     * @tparam ValueType
     * @param box_width
     * @param tree_height
     * @param decal
     * @return ContainerType
     */
    template<typename ContainerType, typename ValueType>
    auto generate_particle_per_leaf(ValueType const& box_width, int const& tree_height,
                                    ValueType const& decal) -> ContainerType
    {
        using particle_type = typename ContainerType::value_type;
        static constexpr std::size_t dimension{particle_type::dimension};
        static constexpr std::size_t dimpow2 = math::pow(2, dimension);
        using point_type = scalfmm::container::point<ValueType, dimension>;

        // step is the leaf size
        ValueType step{box_width / std::pow(2, (tree_height - 1))};
        auto number_of_values_per_dimension = std::size_t(scalfmm::math::pow(2, (tree_height - 1)));
        std::cout << "Number of value per dimension = " << number_of_values_per_dimension << '\n';
        std::cout << "Step = " << step << '\n';
        // start is the center of the first box
        auto start = -box_width * 0.5 + step * 0.5;
        auto delta = step * 0.25 * decal;   // used to separate source and target
        auto number_of_particles = std::pow(dimpow2, (tree_height - 1));
        std::cout << "number_of_particles = " << number_of_particles << " box_width " << box_width << '\n';

        ContainerType particles(number_of_particles);
        for(std::size_t index{0}; index < number_of_particles; ++index)
        {
            // get coord of the cell in the grid with the morton index
            auto coord = scalfmm::index::get_coordinate_from_morton_index<dimension>(index);

            point_type pos{coord};
            //
            particle_type p;
            int ii{0};
            for(auto& e: p.position())
            {
                e = start + step * pos[ii++] + delta;
            }
            particles[index] = p;
        }
        return particles;
    }

    /**
     * @brief
     *
     * @tparam ParticleType
     * @param nb_particles
     * @param center
     * @param width
     * @param seed
     * @return container::particle_container<ParticleType>
     */
    template<typename ParticleType>
    auto generate_particles(std::size_t nb_particles, typename ParticleType::position_type center,
                            typename ParticleType::position_value_type width,
                            const int seed = 33) -> container::particle_container<ParticleType>
    {
        using particle_type = ParticleType;
        using value_type = typename particle_type::position_value_type;
        using container_type = container::particle_container<particle_type>;

        // init container (returned via RVO)
        container_type container(nb_particles);

        // init random generator
        std::mt19937_64 gen(seed);
        constexpr value_type half{0.5};
        std::uniform_real_distribution<value_type> dist(-width * half, width * half);

        // update each particle
        auto it = std::begin(container);
        for(std::size_t i = 0; i < nb_particles; ++i)
        {
            meta::for_each(*it, [&dist, &gen](auto& v) { v = dist(gen); });
            auto part = particle_type(*it);
            auto& pos = part.position();
            auto& out = part.outputs();
            pos += center;
            meta::repeat([](auto& e) { e = 0.; }, out);
            *it = part.as_tuple();
            ++it;
        }

        return container;
    }

    /**
     * @brief
     *
     * @tparam ParticleType
     * @param nb_particles
     * @param center
     * @param width
     * @param seed
     * @return std::vector<ParticleType>
     */
    template<typename ParticleType>
    auto generate_vector_particles(std::size_t nb_particles, typename ParticleType::position_type center,
                                   typename ParticleType::position_value_type width,
                                   const int seed = 33) -> std::vector<ParticleType>
    {
        using particle_type = ParticleType;
        using value_type = typename particle_type::position_value_type;
        using container_type = std::vector<particle_type>;

        // init container (returned via RVO)
        container_type container(nb_particles);

        // Init random generator
        std::mt19937_64 gen(seed);
        constexpr value_type half{0.5};
        std::uniform_real_distribution<value_type> dist(-width * half, width * half);

        // update each particle
        for(std::size_t i = 0; i < nb_particles; ++i)
        {
            // particle_type p;
            auto& part = container[i];
            auto& position = part.position();
            auto& inputs = part.inputs();
            auto& outputs = part.outputs();

            meta::for_each(position, [&dist, &gen](auto& v) { v = dist(gen); });
            meta::for_each(inputs, [&dist, &gen](auto& v) { v = dist(gen); });
            meta::for_each(outputs, [](auto& v) { v = 0.; });

            position += center;
        }

        return container;
    }

    /**
     * @brief Get the center object
     *
     * @tparam T
     * @tparam dimension
     * @return constexpr auto
     */
    template<typename T, std::size_t dimension>
    constexpr auto get_center()
    {
        if constexpr(dimension == 1)
        {
            return container::point<T, dimension>{0.375};
        }
        if constexpr(dimension == 2)
        {
            return container::point<T, dimension>{0.375, 0.125};
        }
        if constexpr(dimension == 3)
        {
            return container::point<T, dimension>{0.375, 0.125, 0.125};
        }
    }

    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam dimension
     * @tparam pv
     */
    template<typename ValueType, std::size_t dimension, std::size_t pv>
    struct get_particle_type
    {
    };

    /**
     * @brief full direct for unit testing.
     *
     * @tparam ValueType
     * @tparam dimension
     * @tparam physical_values
     * @tparam ContainerIterator
     * @tparam MatrixKernelType
     * @tparam Outputs
     * @param begin
     * @param end
     * @param matrix_kernel
     * @param out
     */
    template<typename ValueType, std::size_t dimension, std::size_t physical_values, typename ContainerIterator,
             typename MatrixKernelType, typename Outputs>
    inline auto full_direct_test(ContainerIterator begin, ContainerIterator end, MatrixKernelType matrix_kernel,
                                 Outputs& out) -> void
    {
        using particle_type = typename utils::get_particle_type<ValueType, dimension, physical_values>::type;

        auto outputs_it = std::begin(out);
        for(auto it_p = begin; it_p < end; ++it_p)
        {
            auto pt_x = particle_type(*it_p).position();
            // 1 is the number of physical values (info coming from the MatrixKernelType
            // get the potential
            for(auto it_p2 = begin; it_p2 < it_p; ++it_p2)
            {
                auto pt_y = particle_type(*it_p2).position();
                auto source = particle_type(*it_p2).attributes();
                meta::for_each(*outputs_it, *outputs_it, source,
                               [&matrix_kernel, &pt_x, &pt_y](auto const& o, auto const& s)
                               { return o + (matrix_kernel.evaluate(pt_x, pt_y) * s); });
            }
            for(auto it_p2 = it_p + 1; it_p2 < end; ++it_p2)
            {
                auto pt_y = particle_type(*it_p2).position();
                auto source = particle_type(*it_p2).attributes();
                meta::for_each(*outputs_it, *outputs_it, source,
                               [&matrix_kernel, &pt_x, &pt_y](auto const& o, auto const& s)
                               { return o + (matrix_kernel.evaluate(pt_x, pt_y) * s); });
            }
            ++outputs_it;
        }
    }

    /**
     * @brief
     *
     * @tparam T
     * @tparam U
     * @tparam F
     * @tparam Is
     * @param t
     * @param u
     * @param f
     * @param s
     * @return true
     * @return false
     */
    template<typename T, typename U, typename F, std::size_t... Is>
    constexpr auto meta_compare_impl(T const& t, U const& u, F&& f, std::index_sequence<Is...> s) -> bool
    {
        return (std::invoke(std::forward<F>(f), meta::get<Is>(t), meta::get<Is>(u)) && ...);
    }

    /**
     * @brief
     *
     * @tparam T
     * @tparam U
     * @tparam F
     * @param t
     * @param u
     * @param f
     * @return true
     * @return false
     */
    template<typename T, typename U, typename F>
    constexpr auto meta_compare(T const& t, U const& u, F&& f = std::equal<>) -> bool
    {
        return meta_compare_impl(t, u, std::forward<F>(f), std::make_index_sequence<meta::tuple_size_v<T>>{});
    }

    /**
     * @brief
     *
     * @tparam T
     * @param x
     * @param y
     * @param ulp
     * @return std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
     */
    template<class T>
    typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y, int ulp)
    {
        // the machine epsilon has to be scaled to the magnitude of the values used
        // and multiplied by the desired precision in ULPs (units in the last place)
        return std::fabs(x - y) <= std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp
               // unless the result is subnormal
               || std::fabs(x - y) < std::numeric_limits<T>::min();
    }
}   // namespace scalfmm::utils

#endif   // SCALFMM_UTILS_GENERATE_HPP
