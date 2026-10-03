#ifndef _INRIA_COMPLETE_OCTREE_HPP_
#define _INRIA_COMPLETE_OCTREE_HPP_

#include "region.hpp"
#include "linearize.hpp"

#include "inria/algorithm/distributed/mpi.hpp"
#include "inria/algorithm/distributed/unique.hpp"
#include "inria/algorithm/distributed/distribute.hpp"

namespace inria {
namespace linear_tree {



template<class It, class OutRange>
void distributed_regions_to_linear_tree(
    mpi_config conf,
    It first,
    It last,
    OutRange& out_range
    )
{
    using NodeInfo = typename std::iterator_traits<It>::value_type;
    using inria::unique;
    using inria::distribute;
    using node::level;

    const int rank = conf.comm.rank();
    const int proc_count = conf.comm.size();
    const int ok_tag = conf.base_tag + 975;
    const int empty_tag = conf.base_tag + 979;

    auto type_guard = mpi::create_datatype_if_null<NodeInfo>(conf.datatype);

    unique(conf, first, last);
    last = linearize(first, last);
    std::vector<NodeInfo> octant_list;
    distribute(conf, first, last, octant_list);

    const int proc_min = [&]() {
        int m = (! octant_list.empty()) ? rank : proc_count;
        conf.comm.allreduce(MPI_IN_PLACE, &m, 1, MPI_INT, MPI_MIN);
        return m;
    }();

    const int proc_max = [&]() {
        int m = (! octant_list.empty()) ? rank : 0;
        conf.comm.allreduce(MPI_IN_PLACE, &m, 1, MPI_INT, MPI_MAX);
        return m;
    }();

    // Discard empty processes, the distribute call done before ensures that
    // there is not empty process in between
    if(rank < proc_min || rank > proc_max || proc_min == proc_count) {
        return;
    }

    // Extremities have a special case: add the first and last octant if they don't exist
    if(rank == proc_min) {
        const auto& e = octant_list.front();
        auto c_ans = node::first_child(common_ancestor(e, first_descendant(NodeInfo{0,0}, level(e))));
        octant_list.insert(std::begin(octant_list), c_ans);
    }

    if(rank == proc_max) {
        const auto& e = octant_list.back();
        auto c_ans = node::last_child(common_ancestor(e, last_descendant(NodeInfo{0,0}, level(e))));
        octant_list.push_back(c_ans);
    }

    octant_list.erase(std::unique(std::begin(octant_list), std::end(octant_list)), std::end(octant_list));

    NodeInfo new_last;
    mpi::request recv_req = MPI_REQUEST_NULL, send_req = MPI_REQUEST_NULL;
    // Processes other than the last one receive the next process' first element
    // to append it to their octant list
    if(rank < proc_max) {
        recv_req = conf.comm.irecv(&new_last, 1, conf.datatype, rank+1, MPI_ANY_TAG);
    }

    if(rank > proc_min) {
        if(! octant_list.empty()) {
            // Processes other than 0 send their first element to the previous process
            send_req = conf.comm.isend(std::addressof(octant_list.front()), 1, conf.datatype,
                                       rank-1, ok_tag);
        } else if(rank < proc_max) {
            // When the local list is empty, transmit the next process first
            // element to the previous process.
            mpi::status stat;
            recv_req.wait(stat);
            send_req = conf.comm.isend(&new_last, 1, conf.datatype,
                                       rank-1, stat.tag());
        } else {
            // The last process send and undefined value with an error tag
            send_req = conf.comm.isend(&new_last, 1, conf.datatype,
                                       rank-1, empty_tag);
        }
    }


    mpi::status stat;
    recv_req.wait(stat);
    if(stat.tag() == ok_tag) {
        octant_list.insert(std::end(octant_list), new_last);
    } else if(rank != proc_max && stat.tag() != empty_tag) {
        throw std::runtime_error("Received a message with unknown tag: "
                                 + std::to_string(stat.tag()));
    }
    send_req.wait();

    auto ni_it = octant_list.begin();
    auto ni_next = ni_it+1;
    while(ni_next != octant_list.end()) {
        out_range.insert(end(out_range), *ni_it);
        complete_region(*ni_it, *ni_next, out_range);
        ++ni_it;
        ++ni_next;
    }

    if(rank == proc_max) {
        out_range.push_back(*ni_it);
    }
}




}} // close namespace inria::linear_tree



#endif /* _INRIA_COMPLETE_OCTREE_HPP_ */
