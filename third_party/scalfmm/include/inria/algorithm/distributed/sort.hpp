#ifndef _DISTRIBUTED_SORT_HPP_
#define _DISTRIBUTED_SORT_HPP_

#include <array>
#include <cassert>
#include <iostream>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

#include "impl_sort/binary_search.hpp"
#include "inria/utils.hpp"
#include "mpi.hpp"

#include "inria/algorithm/distributed/distribute.hpp"

/**
 * \brief Distributed sorting algorithm implementation
 * \file
 *
 * The contents of this file implements a distributed sorting algorithm
 * (described in [1]) using MPI.
 *
 * See the description of the sort function for more details
 *
 * [1] David R. Cheng, Viral Shah, John R. Gilbert, Alan Edelman.
 *     "Fast Sorting On a Distributed-Memory Architecture",
 *     Singapore-MPI Alliance, 2005, http://hdl.handle.net/1721.1/7418
 */

namespace inria
{
    namespace details
    {
        namespace sort
        {
            template<class ItStart, class ItEnd = ItStart>
            struct range
            {
                ItStart first;
                ItEnd last;

                ItStart begin() const { return first; }

                ItEnd end() const { return last; }

                auto size() const -> decltype(last - first) { return last - first; }

                ItStart median() { return first + this->size() / 2; }
            };

            /**
             * \brief Computes the size of the global array.
             *
             * \param first Iterator to the beginning of the local range.
             * \param last  Iterator past the end of the local range.
             * \param comm  MPI communicator object.
             *
             * \tparam RandomAccessIterator Random access iterator type.
             *
             * \note This makes a call to the MPI allreduce function and must be called by
             * all the communicator's processes
             */
            template<class RandomAccessIterator>
            std::uint64_t global_range_size(RandomAccessIterator first, RandomAccessIterator last,
                                            mpi::communicator& comm)
            {
                std::uint64_t size = last - first;
                comm.allreduce(MPI_IN_PLACE, &size, 1, MPI_UINT64_T, MPI_SUM);
                return size;
            }

            /**
             * \brief Computes the ideal size of the local section of the global array.
             *
             * The returned size is the final size of the local array after the sort in the
             * best case scenario.
             *
             * \param first Iterator to the beginning of the local range.
             * \param last  Iterator past the end of the local range.
             * \param comm  MPI communicator object.
             *
             * \tparam RandomAccessIterator Random access iterator type.
             *
             * \note This makes a call to the MPI allreduce function and must be called by
             * all the communicator's processes
             */
            template<class RandomAccessIterator>
            std::uint64_t even_distribution_size(RandomAccessIterator first, RandomAccessIterator last,
                                                 mpi::communicator& comm)
            {
                std::uint64_t rank = comm.rank();
                std::uint64_t proc_count = comm.size();
                std::uint64_t size = global_range_size(first, last, comm);
                return size / proc_count + (rank < size % proc_count);
            }

            /**
             * \brief Checks that the data is evenly distributed accross processes
             *
             * \param first Iterator to the beginning of the local range.
             * \param last  Iterator past the end of the local range.
             * \param comm  MPI communicator object.
             *
             * \tparam RandomAccessIterator Random access iterator type.
             *
             * \note This makes a call to the MPI allreduce function and must be called by
             * all the communicator's processes
             */
            template<class RandomAccessIterator>
            bool check_even_distribution(RandomAccessIterator first, RandomAccessIterator last, mpi::communicator& comm)
            {
                char check = (even_distribution_size(first, last, comm) == static_cast<std::uint64_t>(last - first));
                comm.allreduce(MPI_IN_PLACE, &check, 1, MPI_CHAR, MPI_LAND);
                return check;
            }

            /**
             * \brief Compute the pivot ranks in the global array
             *
             * The pivot ranks are the ranks of the first element of each process local
             * range.
             *
             * \param first Iterator to the beginning of the local range.
             * \param last  Iterator past the end of the local range.
             * \param comm  MPI communicator.
             *
             * \tparam RandomAccessIterator Random access iterator type.
             *
             * \note This makes a call to the MPI allreduce function and must be called by
             * all the communicator's processes
             */
            template<class RandomAccessIterator>
            std::vector<std::uint64_t> get_pivot_ranks(RandomAccessIterator first, RandomAccessIterator last,
                                                       mpi::communicator& comm)
            {
                std::uint64_t proc_count = comm.size(), size = last - first;
                comm.allreduce(MPI_IN_PLACE, &size, 1, MPI_UINT64_T, MPI_SUM);
                const std::uint64_t mod = size % proc_count;
                std::vector<std::uint64_t> res(proc_count - 1, size / proc_count);
                for(std::size_t r = 1; r < proc_count - 1; ++r)
                {
                    res[r] += res[r - 1] + (r < mod);
                }
                return res;
            }

            /**
             * \brief Finds the rank interval of the target in a sorted sub-range
             *
             * Finds the ranks where the target could be inserted in the sub-range #range,
             * without breaking the order, relative to the start of the local array starting
             * at #reference_iterator.
             *
             * For example: in the range `[0,2,4,6,8]`, the element 4 has the rank range
             * [2,4[; the element 5 has the range [3,4[.
             *
             * \param range_first     Iterator to the first element of the local array.
             * \param subrange_first  Iterator to the first element of the subrange.
             * \param subrange_last   Iterator to the last element of the subrange.
             * \param target          Element which rank interval to find.
             * \param comp            Comparison function object, returns `true` if the
             *                        first argument is less than the second.
             *
             * \tparam RandomAccessIterator Random access iterator type
             * \tparam T        Type of the target object, must be comparable to the elements in
             *                  the range.
             * \tparam Compare  Comparison functor type. Must expose an operator() with the
             *                  signature `bool operator()(T target, U elem)` which returns
             *                  `true` if `target < elem`.
             *
             * \return A 2 element array containg the rank interval of the target as in
             *         `[begin rank, end rank[`. The interval is empty if the begin and end
             *         ranks are equal.
             */
            template<class RandomAccessIterator, class T, class Compare>
            std::array<std::uint64_t, 2>
            find_local_rank_in_subrange(RandomAccessIterator range_first, RandomAccessIterator subrange_first,
                                        RandomAccessIterator subrange_last, const T& target, Compare comp)
            {
                RandomAccessIterator r_first = std::lower_bound(subrange_first, subrange_last, target, comp);
                RandomAccessIterator r_last = r_first;
                if(r_first != subrange_last && details::equal(*r_first, target, comp))
                {
                    r_last = std::upper_bound(r_first, subrange_last, target, comp);
                }
                return {{static_cast<std::uint64_t>(r_first - range_first),
                         static_cast<std::uint64_t>(r_last - range_first)}};
            }

            /**
             * \brief Find the values of the pivots corresponding to the given ranks
             *
             * This method takes a list of ranks in the globally sorted array and finds the
             * values that correspond.
             *
             * \param first Iterator to the beginning of the local range.
             * \param last  Iterator past the end of the local range.
             * \param pivot_ranks The ranks of the value that are searched.
             * \param comp  Comparison function object, returns `true` if the first argument
             *              is less than the second.
             * \param conf  MPI configuration.
             *
             * \tparam RandomAccessIterator Random access iterator type
             * \tparam T        Type of the objects that are sorted. T must be trivially
             *                  copiable to allow MPI communication
             * \tparam Compare  Comparison functor type. Must expose an operator() with the
             *                  signature `bool operator()(T target, U elem)` which returns
             *                  `true` if `target < elem`.
             *
             * \return A vector of elements of type T found in the global array that have
             *         the given ranks in the same sorted global array.
             *
             * ##### Example:
             * ~~~
             * process:         A     B     C
             * global array: [b,f,h|c,c,g|a,d,i]
             * index:         0 1 2 3 4 5 6 7 8
             *
             * The in-between process pivots have the rank [3,6].
             *
             * The `pivot_selection` function finds the elements that have the ranks [3,6]
             * in the sorted global array without sorting it. It relies on the fact that the
             * local arrays are sorted.
             *
             * Here, the elements returned by the function are [c,g].
             *
             * sorted global array (for reference): [a,b,c|c,d,f|g,h,i]
             *                                       0 1 2 3 4 5 6 7 8
             * ~~~
             */
            template<class RandomAccessIterator,
                     class T = typename std::iterator_traits<RandomAccessIterator>::value_type, class Compare>
            auto pivot_selection(RandomAccessIterator first, RandomAccessIterator last,
                                 std::vector<std::uint64_t> pivot_ranks, Compare comp, const mpi_config& conf)
              -> std::vector<T>
            {
                const int proc_count = conf.comm.size();
                const int rank = conf.comm.rank();
                const int pivot_count = static_cast<int>(pivot_ranks.size());
                std::vector<range<RandomAccessIterator>> active_ranges(pivot_count, {first, last});
                std::vector<T> pivots(pivot_count);
                std::vector<int> pivot_found(pivot_count, false);

                auto medians_buffer = std::unique_ptr<T[]>(new T[proc_count * pivot_count]);
                auto medians = [&pivot_count, &medians_buffer](int proc_idx) {
                    return medians_buffer.get() + pivot_count * proc_idx;
                };

                auto lengths_buffer = std::unique_ptr<std::uint64_t[]>(new uint64_t[proc_count * pivot_count]);
                auto lengths = [&pivot_count, &lengths_buffer](int proc_idx) {
                    return lengths_buffer.get() + pivot_count * proc_idx;
                };

                auto rank_buffer = std::unique_ptr<std::uint64_t[]>(new uint64_t[2 * proc_count * pivot_count]);
                auto rank_interval = [&pivot_count, &rank_buffer](int proc_idx) {
                    return rank_buffer.get() + 2 * pivot_count * proc_idx;
                };

                T* local_medians = medians(rank);
                std::uint64_t* local_lengths = lengths(rank);

                auto w_medians = std::unique_ptr<T[]>(new T[pivot_count]);
                auto local_median_rank_ranges =
                  std::unique_ptr<std::array<std::uint64_t, 2>[]>(new std::array<std::uint64_t, 2>[2 * pivot_count]);
                auto median_rank_ranges =
                  std::unique_ptr<std::array<std::uint64_t, 2>[]>(new std::array<std::uint64_t, 2>[2 * pivot_count]);

                while(std::count(std::begin(pivot_found), std::end(pivot_found), false))
                {
                    // 1. Bcast local medians & interval info
                    for(int i = 0; i < pivot_count; ++i)
                    {
                        // std::cerr << io::container(active_ranges[i]) << '\n';
                        if(active_ranges[i].size())
                            local_medians[i] = *(active_ranges[i].median());
                        local_lengths[i] = active_ranges[i].size();
                    }

                    conf.comm.allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, lengths(0), pivot_count, MPI_UINT64_T);
                    conf.comm.allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, medians(0), pivot_count, conf.datatype);

                    // 2. Find weighted medians.
                    for(int i = 0; i < pivot_count; ++i)
                    {
                        std::uint64_t total_length = 0;
                        for(int pr = 0; pr < proc_count; ++pr)
                        {
                            total_length += lengths(pr)[i];
                        }
                        int w_median_idx = -1;
                        double accu = 0;
                        while((accu < .5) && w_median_idx + 1 < proc_count)
                        {
                            ++w_median_idx;
                            accu += double(lengths(w_median_idx)[i]) / double(total_length);
                        }
                        w_medians[i] = medians(w_median_idx)[i];
                    }

                    // 3. Find first and last idx between which the weighted medians can fit
                    // int the local active ranges
                    for(int i = 0; i < pivot_count; ++i)
                    {
                        local_median_rank_ranges[i] = find_local_rank_in_subrange(
                          first, std::begin(active_ranges[i]), std::end(active_ranges[i]), w_medians[i], comp);
                    }

                    // 4. Find out global rank ranges
                    conf.comm.allgather(local_median_rank_ranges.get(), 2 * pivot_count, MPI_UINT64_T,
                                        rank_buffer.get(), 2 * pivot_count, MPI_UINT64_T);

                    for(int i = 0; i < pivot_count; ++i)
                    {
                        median_rank_ranges[i][0] = 0;
                        median_rank_ranges[i][1] = 0;
                        for(int pr = 0; pr < proc_count; ++pr)
                        {
                            median_rank_ranges[i][0] += rank_interval(pr)[2 * i];
                            median_rank_ranges[i][1] += rank_interval(pr)[2 * i + 1];
                        }
                    }

                    // 5. Check whether pivot_rank is in the computed median_rank_range, if
                    // not loop with a smaller range.
                    for(int i = 0; i < pivot_count; ++i)
                    {
                        if(pivot_found[i])
                        {
                            continue;
                        }

                        if((pivot_ranks[i] >= median_rank_ranges[i][0]) && (pivot_ranks[i] < median_rank_ranges[i][1]))
                        {
                            pivots[i] = w_medians[i];
                            pivot_found[i] = true;
                        }
                        else
                        {
                            int bonus_offset =
                              active_ranges[i].size() && details::equal(w_medians[i], local_medians[i], comp);
                            if(median_rank_ranges[i][0] < pivot_ranks[i])
                            {
                                active_ranges[i].first = first + local_median_rank_ranges[i][0] + bonus_offset;
                            }
                            else if(median_rank_ranges[i][1] > pivot_ranks[i])
                            {
                                active_ranges[i].last = first + local_median_rank_ranges[i][1] - bonus_offset;
                            }
                        }
                    }
                }
                return pivots;
            }

            /**
             * \brief Distribution object using pivots to choose target process
             *
             * This class is to be used with the inria::distribute function. When called, it
             * returns the rank of the process that should hold the given element after the
             * sort.
             *
             * \tparam Comp   The comparison callable type.
             * \tparam Pivots The list of pivots type.
             */
            template<class Comp, class Pivots>
            struct pivot_distribution
            {
                Comp comp;
                Pivots pivots;
                template<class U>
                std::uint64_t operator()(const U& e)
                {
                    auto p_it = std::upper_bound(std::begin(pivots), std::end(pivots), e, comp);
                    std::uint64_t p_idx = p_it - std::begin(pivots);
                    if(p_idx > pivots.size())
                    {
                        throw std::runtime_error("Particle distributed to non existing "
                                                 "process " +
                                                 std::to_string(p_idx));
                    }
                    return p_idx;
                }
            };

            /**
             * \brief Helper function to instanciate a pivot_distribution.
             *
             * \param comp The comparison function object
             * \param pivots The pivots list.
             *
             * \tparam Comp   The comparison callable type.
             * \tparam Pivots The list of pivots type.
             *
             */
            template<class Comp, class Pivots>
            pivot_distribution<Comp, Pivots> make_pivot_distribution(Comp&& comp, Pivots&& pivots)
            {
                return {std::forward<Comp>(comp), std::forward<Pivots>(pivots)};
            }

        }   // namespace sort
    }       // namespace details

    /**
     * \brief Distributed sort.
     *
     * Sorts data accross multiple processes as described in [1]. Processes are
     * numbered between 0 and P-1. There are N pieces of data. The local process
     * holds data in the range \[`first`, `last`\[.
     *
     * Each process holds a local range of elements $v_i$ of the global range
     * $v$. The algorithm sorts the elements in three main steps.
     *
     *   A. Sort the local ranges
     *
     *   B. Find the delimiters between processes and redistribute the data to
     * the right ones
     *      - Find the ranks of the process delimiter values (pivots)
     *        [r_1, r_2, ..., r_(p-1)].
     *      - Find the values corresponding to the ranks [v_1, v_2, ...,
     * v_(p-1)].
     *      - Send the data in the range [v_i, v_(i+1)[ to process i.
     *
     *   C. Sort the received data
     *
     * \param first Iterator to the first element of the local data.
     * \param last  Iterator past the last element of the local data.
     * \param comp  Comparison functor. `comp(a,b)` must return `true` if a < b,
     *              `false` otherwise.
     * \param conf  MPI configuration
     *
     * \tparam Compare   Comparison functor type. Must expose an operator() with
     * the signature `bool operator()(T target, U elem)` which returns `true` if
     * `target < elem`. \tparam T         Type of the data to be sorted. Must be
     * ValueSwappable.
     *
     * \return A newly created/allocated contiguous storage container holding
     * the sorted data for the calling process. The data is balanced accross the
     *         processes regardless of its initial distribution, disallowing an
     *         element instances to be split accross two processes.
     *
     * [1] David R. Cheng, Viral Shah, John R. Gilbert, Alan Edelman.
     *     "Fast Sorting On a Distributed-Memory Architecture",
     *     Singapore-MPI Alliance, 2005, http://hdl.handle.net/1721.1/7418
     */
    template<class It, class Compare, class T = typename std::iterator_traits<It>::value_type>
    std::vector<T> sort(mpi_config conf, It first, It last, Compare comp)
    {
        //    static_assert(std::is_trivially_copyable<T>::value,
        //                  "The elements must be trivially copyable to be sent
        //                  through MPI");
        std::cerr << "Warning " << __FILE__ << " at ine " << __LINE__
                  << "  static_assert(std::is_trivially_copyable<T>::value, is removed " << std::endl;
        // Guess datatype if it is MPI_DATATYPE_NULL
        auto type_guard = mpi::create_datatype_if_null<T>(conf.datatype);

        std::vector<T> data;
        std::uint64_t global_size = details::sort::global_range_size(first, last, conf.comm);
        if(global_size < (std::uint64_t)conf.comm.size())
        {
            // Gather data on process 0
            distribute(conf, first, last, data, [](const T&) { return 0; });
        }
        else
        {
            // Local data sort
            std::sort(first, last, comp);
            // Compute pivots ranks: rank of elements at the start of each local range
            // in the globally sorted array
            auto pivot_ranks = details::sort::get_pivot_ranks(first, last, conf.comm);
            // Find pivots
            auto pivots = details::sort::pivot_selection(first, last, pivot_ranks, comp, conf);
            // Redistribute data
            distribute(conf, first, last, data, details::sort::make_pivot_distribution(comp, std::move(pivots)));
        }
        // Final local sort
        std::sort(begin(data), end(data), comp);
        return data;
    };

    /**
     * \copydoc sort
     */
    template<class It>
    auto sort(mpi_config conf, It first, It last)
      -> decltype(sort(conf, first, last, std::less<typename std::iterator_traits<It>::value_type>{}))
    {
        return sort(conf, first, last, std::less<typename std::iterator_traits<It>::value_type>{});
    }

    template<class ContiguousRange, class Compare>
    void sort(mpi_config conf, ContiguousRange& range, Compare comp)
    {
        auto data = sort(conf, std::begin(range), std::end(range), comp);
        range.assign(begin(data), end(data));
    }

    template<class ContiguousRange, class Compare>
    void sort(mpi_config conf, ContiguousRange& range)
    {
        auto data = sort(conf, std::begin(range), std::end(range));
        range.assign(begin(data), end(data));
    }

}   // namespace inria

#endif /* _DISTRIBUTED_SORT_HPP_ */
