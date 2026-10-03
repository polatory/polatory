#ifndef _INRIA_LINEAR_TREE_GATHER_OCTANT_WEIGHTS_HPP_
#define _INRIA_LINEAR_TREE_GATHER_OCTANT_WEIGHTS_HPP_

#include "node.hpp"

#include "weight_traits.hpp"

#include "inria/algorithm/distributed/mpi.hpp"

namespace inria {
namespace linear_tree {

template<class Integer = std::size_t, class Box, class Position>
Integer get_morton_index(const Position& pos, const Box& box, const std::size_t level) {
    constexpr auto Dim = Position::Dim;
    using dim_t = typename std::remove_const<decltype(Dim)>::type;
    double cell_width = box.width(0) / (1<<level);
    Integer coords[Dim] = {};

    for(dim_t i = 0; i < Dim; ++i) {
        coords[Dim-i-1] = static_cast<Integer>((pos[i] - box.c1()[i]) / cell_width);
        coords[Dim-i-1] <<= Dim-i-1;
    }

    Integer mask = 1;
    Integer idx  = 0;
    auto not_done = [&] {
        if(mask == 0) {
            return false;
        }
        for(dim_t i = 0u; i < Dim; ++i)
            if((mask << i) <= coords[i])
                return true;
        return false;
    };

    while(not_done()) {
        for(dim_t i = 0u; i < Dim; ++i) {
            idx |= (coords[i] & mask);
            mask <<= 1;
            coords[i] <<= Dim-1;
        }
    }

    return idx;
}



template<class Box>
struct morton_index_comp {
    std::size_t l; ///< morton index level
    Box box;       ///< space box
    template<class P1, class P2>
    bool operator()(const P1& p1, const P2& p2) {
        return get_morton_index(p1.position(),box,l)
            < get_morton_index(p2.position(),box,l);
    }
};


template<class OctantForwardIt, class ParticleForwardIt, class Box,
         class T = typename std::iterator_traits<OctantForwardIt>::value_type,
         class P = typename std::iterator_traits<ParticleForwardIt>::value_type
         >
void gather_octants_weight(
    inria::mpi_config conf,
    OctantForwardIt first,
    OctantForwardIt last,
    ParticleForwardIt p_first,
    ParticleForwardIt p_last,
    Box box)
{
    using weight_t = typename meta::weight_traits<P>::weight_t;

    using node::level;
    using node::morton_index;

    std::uint64_t list_size = std::distance(first, last);
    std::uint64_t buffer_size = 0;
    conf.comm.allreduce(&list_size, &buffer_size, 1, MPI_UINT64_T, MPI_MAX);

    std::vector<node::info_t<T>> oct_buffer(buffer_size);
    std::vector<weight_t> weight_buffer(buffer_size);

    MPI_Datatype node_info_datatype = MPI_DATATYPE_NULL;
    auto ni_type_guard = mpi::create_datatype_if_null<node::info_t<T>>(node_info_datatype);

    MPI_Datatype weight_datatype = mpi::get_datatype<weight_t>();
    auto w_type_guard = mpi::create_datatype_if_null<weight_t>(weight_datatype);

    const int proc_count = conf.comm.size();
    const int rank = conf.comm.rank();

    for(int r = 0; r < proc_count; ++r) {
        if(rank == r) {
            oct_buffer.assign(first, last);
            buffer_size = oct_buffer.size();
        }
        conf.comm.bcast(&buffer_size, 1, MPI_UINT64_T, r);
        conf.comm.bcast(oct_buffer.data(), static_cast<int>(buffer_size),
                        node_info_datatype, r);

        weight_buffer.clear();
        weight_buffer.resize(buffer_size);

        int w_idx = 0;
        auto o_first = begin(oct_buffer);
        auto o_last = o_first + buffer_size;

        // Check end of buffer or end of particles
        while(o_first != o_last && p_first != p_last) {
            auto p_idx = get_morton_index(p_first->position(), box,
                                          level(*o_first));
            auto o_idx = morton_index(*o_first);
            if(p_idx > o_idx) {
                ++o_first;
                ++w_idx;
            } else if(p_idx == o_idx) {
                weight_buffer[w_idx] += meta::get_weight(*p_first);
                ++p_first;
            } else {
                throw std::runtime_error("A particle was not affected to an"
                                         " octant, check particle order and"
                                         " octant order.");
            }
        }

        if(rank == r) {
            conf.comm.reduce(MPI_IN_PLACE, weight_buffer.data(),
                             static_cast<int>(buffer_size),
                             weight_datatype, MPI_SUM, r);
            int i = 0;
            for(auto it = first; it != last; ++it, ++i) {
                meta::set_weight(*it, weight_buffer[i]);
            }
        } else {
            for(std::size_t i = 0; i < buffer_size; ++i) {
            }
            conf.comm.reduce(weight_buffer.data(), nullptr,
                             static_cast<int>(buffer_size),
                             weight_datatype, MPI_SUM, r);
        }
    }
}

}} // close namespace inria::linear_tree

#endif /* _INRIA_LINEAR_TREE_GATHER_OCTANT_WEIGHTS_HPP_ */
