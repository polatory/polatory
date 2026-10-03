#ifndef _REMOVE_DUPLICATES_HPP_
#define _REMOVE_DUPLICATES_HPP_

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include "mpi.hpp"
#include "inria/utils.hpp"

namespace inria {

/**
 * \brief Remove duplicates in a distributed sorted range
 *
 * The elements held localy are in the range [first, last[.
 *
 * \note The order the elements are sorted in is not important. Equal elements
 * must be consecutive, possibly accros process boundaries.
 *
 * \param first    Iterator to the range first element.
 * \param last     Iterator past the range last element.
 * \param comp     Function object that returns true when two elements are equal.
 * \param mpi_type MPI datatype corresponding to the elements in the range.
 * \param comm     MPI communicator.
 *
 * \tparam It      Bidirectional iterator
 * \tparam Compare Comparison type. Must expose a call operator such as
 *                 `Compare::operator(T a, T b)` returns true if `a == b` and
 *                 false otherwise.
 * \tparam T       The type of the range elements. T must be trivially copyable.
 *
 * \return The new end of the range. The elements past this new are are in an
 *         unspecified state as per move operator.
 *
 * \warning The duplicate elements are not actually removed. It is the
 * programmer's reponsibility to resize the actual range if needed.
 */
template<class It, class Compare>
It unique(mpi_config conf, It first, It last, Compare comp) {
    using T = typename std::iterator_traits<It>::value_type;
    enum {NONE_TAG = 65, OK_TAG = 94};

    static_assert(std::is_trivially_copyable<T>::value,
                  "The range elements must be trivially copiable");

    // Guess datatype if it is MPI_DATATYPE_NULL
    auto type_guard = mpi::create_datatype_if_null<T>(conf.datatype);

    const int proc_count = conf.comm.size();
    const int rank = conf.comm.rank();
    T prev_last_elem;

    if(proc_count > 1) {
        if(rank == 0) {
            if(last - first == 0) {
                conf.comm.send(nullptr, 0, MPI_BYTE, 1, conf.base_tag + NONE_TAG);
            } else {
                T last_elem = *(last-1);
                conf.comm.send(&last_elem, 1, conf.datatype, rank+1,
                               conf.base_tag + OK_TAG);
            }
        } else {
            mpi::status stat;
            conf.comm.recv(&prev_last_elem, 1, conf.datatype, rank-1, MPI_ANY_TAG, stat);

            if(rank < proc_count -1) {
                if(last - first == 0) {
                    conf.comm.send(&prev_last_elem, 1, conf.datatype, rank+1, stat.raw_status.MPI_TAG);
                } else {
                    T last_elem = *(last-1);
                    conf.comm.send(&last_elem, 1, conf.datatype, rank+1, conf.base_tag + OK_TAG);
                }
            }

            // If the previous process sent an element prev_last_elem, mark it
            // for removal by ovewriting it whith an other element from the
            // range. The duplicates will then be removed using a local unique
            // algorithm.
            //
            // The element copied is the first element different from the
            // received one.
            //
            // This avoid shifting the whole range to remove the previous
            // process element.
            if(stat.raw_status.MPI_TAG != conf.base_tag + NONE_TAG) {
                auto not_prev_last_elem = [&comp, &prev_last_elem](const T& e) {
                    return !comp(e, prev_last_elem);
                };
                auto first_not_prev_last_elem = std::find_if(first, last, not_prev_last_elem);
                if(first_not_prev_last_elem != last) {
                    std::fill(first, first_not_prev_last_elem, *first_not_prev_last_elem);
                } else {
                    // all elements are equal to the previous process last element
                    last = first;
                }
            }
        }
    }

    return std::unique(first, last, comp);
}


template<class It>
It unique(mpi_config conf, It first, It last) {
    using T = typename std::iterator_traits<It>::value_type;
    return unique(conf, first, last, std::equal_to<T>{});
}

/**
 * \brief Erases duplicates in a container and resizes it accordingly.
 */
template<class Container, class Compare>
void unique(mpi_config conf, Container& container, Compare comp) {
    auto new_end = unique(conf, std::begin(container), std::end(container), comp);
    container.erase(new_end, std::end(container));
}

template<class Container>
void unique(mpi_config conf, Container& container) {
    auto new_end = unique(conf, std::begin(container), std::end(container));
    container.erase(new_end, std::end(container));
}

}



#endif /* _REMOVE_DUPLICATES_HPP_ */
