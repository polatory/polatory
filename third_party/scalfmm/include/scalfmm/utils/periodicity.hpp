// --------------------------------
// See LICENCE file at project root
// File : scalfmm/utils/periodicity.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_PERIODICITY_HPP
#define SCALFMM_UTILS_PERIODICITY_HPP

namespace scalfmm::utils
{

    /**
     * @brief
     *
     * @tparam ContainerType
     * @tparam ValueType
     * @tparam ArrayType
     * @param origin
     * @param width
     * @param pbc
     * @return ContainerType
     */
    template<typename ContainerType, typename ValueType, typename ArrayType>
    auto replicated_distribution_grid_3x3(ContainerType const& origin, ValueType width,
                                          ArrayType const& pbc) -> ContainerType
    {
        using container_type = ContainerType;
        using particle_type = typename container_type::value_type;
        using position_type = typename particle_type::position_type;
        static constexpr std::size_t dimension{particle_type::dimension};

        // number of axes replicated
        std::size_t ones = std::count(std::begin(pbc), std::end(pbc), true);
        // number of times the distribution will be replicated : 3^(axes concerned)
        auto n_replicates = scalfmm::math::pow(3, ones);

        // new container with replicates
        container_type replicated_dist(n_replicates * origin.size());

        // vector storing all translation vectors.
        std::vector<position_type> us{};

        // loop indices for nd loops : [-1,2[
        std::array<int, dimension> starts{};
        std::array<int, dimension> stops{};
        starts.fill(-1);
        stops.fill(2);
        // convert pbc vector to int
        std::array<int, dimension> pbc_integer{};
        std::transform(std::begin(pbc), std::end(pbc), std::begin(pbc_integer), [](auto e) { return int(e); });

        // lambda to generate all translating vectors
        auto finding_us = [&us, &pbc_integer, width](auto... is)
        {
            // building unitary translation vector
            position_type u{ValueType(is)...};
            // building ans of unitary translation vector
            std::array<int, dimension> u_int{std::abs(is)...};
            // mask
            std::array<int, dimension> masked{};

            // applying logical between u_int and pbc_integer
            // this will result in a mask
            std::transform(std::begin(u_int), std::end(u_int), std::begin(pbc_integer), std::begin(masked),
                           [](auto a, auto b) { return a || b; });
            // if this mask is equal to the pbc vector, it means that the current
            // translation vector needs to be selected
            // note that we also select the vector that does not move the original box
            bool eq = std::equal(std::begin(pbc_integer), std::end(pbc_integer), std::begin(masked));
            if(eq)
            {
                // push the unitary translation vector scaled to box width
                us.push_back(u * width);
            }
        };

        // call the finding_us lambda in a nd loop nest.
        scalfmm::meta::looper_range<dimension>{}(finding_us, starts, stops);

        // check if we have the good number of translation vector
        if(us.size() != n_replicates)
        {
            std::cerr << cpp_tools::colors::red << "[error] : number of translating vectors != number of replications."
                      << cpp_tools::colors::reset << '\n';
            std::exit(-1);
        }

        // lambda to replicate and move particles in new container
        // applying a translation vector
        std::size_t jump_replication{0};
        auto applying_us = [&replicated_dist, &origin, &jump_replication](auto u)
        {
            for(std::size_t i{0}; i < origin.size(); ++i)
            {
                auto p = origin.particle(i);
                // applying u
                p.position() = p.position() + u;
                // if u is different from the zeros vector we know it's not
                // the original box so set the flag to 0
                if(u != position_type(0.))
                {
                    auto& is_origin = scalfmm::meta::get<1>(p.variables());
                    is_origin = 0;
                };
                // insert new particle
                replicated_dist.insert_particle((jump_replication * origin.size()) + i, p);
            }
            // jump to next replicate
            ++jump_replication;
        };

        // for each translation vector we generate new particles
        std::for_each(std::begin(us), std::end(us), applying_us);
        // returning the new vector
        return replicated_dist;
    }

    /**
     * @brief
     *
     * @tparam ContainerType
     * @param container
     * @param size_part
     * @param ref
     * @return std::vector<typename ContainerType::particle_type>
     */
    template<typename ContainerType>
    auto extract_particles_from_ref(ContainerType const& container, const int& size_part,
                                    const int& ref) -> std::vector<typename ContainerType::particle_type>
    {
        using particle_type = typename ContainerType::particle_type;

        std::vector<particle_type> extracted_cont(size_part);

        for(std::size_t i{0}; i < container.size(); ++i)
        {
            const auto& p = container.particle(i);
            const auto& box_type = std::get<1>(p.variables());

            if(box_type == ref)
            {
                const auto& idx = std::get<0>(p.variables());
                extracted_cont[idx] = p;
            }
        }

        return extracted_cont;
    }
}   // namespace scalfmm::utils
#endif
