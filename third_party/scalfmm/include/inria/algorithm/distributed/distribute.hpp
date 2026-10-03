#ifndef _DISTRIBUTE_HPP_
#define _DISTRIBUTE_HPP_

/**
 * \brief Distributed distribute algorithm
 * \file
 *
 * \author Quentin Khan
 */

#include "inria/algorithm/distributed/mpi.hpp"
#include "inria/io.hpp"
#include "inria/meta.hpp"
#include "inria/utils.hpp"

#include <iterator>
#include <numeric>
#include <vector>

/**
 * \brief Inria namespace
 *
 */
namespace inria
{
    namespace details
    {
        /**
         * \brief A non owning memory chunk of an array.
         *
         * \tparam T Type of the array's elements
         * \tparam Int Size type
         */
        template<class T, class Int>
        struct chunk_t : std::pair<T*, Int>
        {
            using std::pair<T*, Int>::pair;

            /**
             * \brief Chunk element count
             */
            Int size() const noexcept { return this->second; }

            /**
             * \brief Increase chunk size
             */
            void expand(Int value = 1) { this->second += value; }

            /**
             * \brief Reseat chunk underlying pointer
             */
            void reset(T* addr) { this->first = addr; }

            /**
             * \brief Change span size
             *
             * \param value New chunk size
             */
            void resize(Int value) { this->second = value; }

            /**
             * \brief Get underlying data
             */
            T* data() noexcept { return this->first; }
            /** \copydoc data */
            const T* data() const noexcept { return this->first; }

            /**
             * \brief Get begin iterator
             */
            T* begin() noexcept { return this->first; }
            /** \copydoc begin */
            const T* begin() const noexcept { return this->first; }
            /** \copydoc begin */
            const T* cbegin() const noexcept { return this->first; }

            /**
             * \brief Get end iterator
             */
            T* end() noexcept { return std::next(this->first, this->second); }
            /** \copydoc end */
            const T* end() const noexcept { return std::next(this->first, this->second); }
            /** \copydoc end */
            const T* cend() const noexcept { return std::next(this->first, this->second); }
        };

        /**
         * \brief Implements inter process distribution main steps.
         *
         * Distribution is done in 3 steps:
         *
         *   - Local data initialisation: gather information about the environment, how
         *     distribution is going to be decided, to which process.
         *   - Remote data and communication setup: decide which piece of data send to
         *     which process, asks other processes what data they will send.
         *   - Data communication: send and receive data.
         *
         * \tparam T Type of the objects to send
         * \tparam ForwardIt Local range iterator
         * \tparam Distribution Distribution configuration.
         * \parblock
         * An instance of Distribution must provide an call operator taking a `const T&`
         * and returning the rank of the target process for that object.
         *
         * ~~~{.cpp}
         * int Distribution::operator()(const T&);
         * ~~~
         * \endparblock
         */
        template<class T, class ForwardIt, class Distribution>
        struct distribution_state
        {
            distribution_state() = delete;
            distribution_state(const distribution_state&) = delete;
            distribution_state(distribution_state&&) = delete;
            distribution_state& operator=(const distribution_state&) = delete;
            distribution_state& operator=(distribution_state&&) = delete;

            // MPI configuration

            /// MPI environment configuration
            inria::mpi_config conf;
            /// Datatype management
            inria::mpi::datatype_commit_guard datatype_guard;
            /// MPI communication tag for successful comms
            const int ok_tag;

            /// Distribution object
            Distribution distrib;

            // Communication data

            /**
             * \brief Copy of the range to send
             *
             * One buffer per target process
             */
            std::vector<std::vector<T>> send_buffers;
            /**
             * \brief Reception buffer
             *
             * Only one buffer, the communication sizes are expected to be
             * known. Therefore the reception offsets can be computed and written
             * directly in the final buffer.
             */
            std::vector<T> recv_buffer;

            /// Total reception size, to initialise #recv_buffer
            std::uint64_t recv_size;

            /// View type that holds pointer to data and element count
            using chunk_t = details::chunk_t<T, std::uint64_t>;
            /**
             * \brief Send information buffer
             *
             * Contains the 'send' information for the data to be sent to each
             * process. Each process receives the size of the data
             * communication. #send_buffers is not used directly because MPI needs
             * pointers to existing storage to send data.
             */
            std::vector<chunk_t> send_chunks;
            /**
             * \brief Receive information buffer
             *
             * Holds the size and target pointer to the span of recv_buffer that will
             * receive data from each process.
             */
            std::vector<chunk_t> recv_chunks;

            /**
             * \brief Intialise local data and setup environment
             *
             * On construction, the distribution_state performs the following:
             *
             *   - setup the MPI environment: a datatype is created if the configuration
             *     specifies MPI_DATATYPE_NULL.
             *   - copy the input range to the send buffer
             *   - compute the local range weight, the global range weight, the
             *     cumulated range weight up to local range (excluded) and the target
             *     cumulated weight for all process.
             *
             * The final distribution is computed from the proportions
             * argument. Proportion must expose an operator[rank] that returns the
             * proportion for the process of given rank. The proportion maximum is
             *
             *     sum(0 <= i < nb_proc, propotions[i]);
             *
             * \param conf_ MPI environment configuration
             * \param first Iterator to local range first element
             * \param last  Local range end iterator
             * \param distribution Distribution object
             * \parblock
             *
             * Must provide a call operator returning the rank of the process the given
             * element belongs to.
             *
             * ~~~{.cpp}
             * int Distribution::operator()(const T&);
             * ~~~
             * \endparblock
             */
            distribution_state(inria::mpi_config conf_, ForwardIt first, ForwardIt last, Distribution& distribution)
              : conf(conf_)
              , datatype_guard(mpi::create_datatype_if_null<T>(conf.datatype))
              , ok_tag(conf.base_tag + 153)
              , distrib(distribution)
              , send_buffers(conf.comm.size())
              , send_chunks(conf.comm.size(), {nullptr, 0})
              , recv_chunks(conf.comm.size(), {nullptr, 0})
            {
                setup_send_data(first, last);
                setup_recv_data();
            }

            /**
             * \brief Setup send data
             *
             * Fills the #send_buffers and sets the #send_chunks spans. Each element of
             * the range is passed to the distribution and inserted in the relevant
             * send_buffer.
             *
             * \param first Iterator to local range first element
             * \param last  Local range end iterator
             */
            void setup_send_data(ForwardIt first, ForwardIt last)
            {
                // Copy each element to the corresponding send buffer
                while(first != last)
                {
                    std::uint64_t target_rank = this->distrib(*first);
                    assert(target_rank < this->send_buffers.size());
                    this->send_buffers[target_rank].push_back(*first);
                    ++first;
                }
                // Setup span views
                for(std::size_t i = 0; i < this->send_buffers.size(); ++i)
                {
                    send_chunks[i].resize(this->send_buffers[i].size());
                    send_chunks[i].reset(this->send_buffers[i].data());
                }
            }

            /**
             * \brief Setup send data and prepare reception buffer.
             *
             * All processes communicate to all processes the amount of data they will
             * send. The reception buffer size is computed and sliced to simultaneously
             * receive from several processes at the same time.
             */
            void setup_recv_data()
            {
                using std::begin;
                using std::end;
                // Setup recv data
                //
                // Send/receive future comms metadata to each other process j:
                //    - send the number of elements to send to j
                //    - receive the number of elements to receive
                //
                // Create MPI datatype [****,    ]
                //              uint64_t ^     ^ empty space to reach total size of chunk_t
                MPI_Datatype mpi_chunk_size;
                MPI_Type_create_resized(MPI_UINT64_T, 0, sizeof(chunk_t), &mpi_chunk_size);
                MPI_Type_commit(&mpi_chunk_size);
                // Actual communication round
                conf.comm.alltoall(&(this->send_chunks[0].second), 1, mpi_chunk_size, &(this->recv_chunks[0].second), 1,
                                   mpi_chunk_size);
                MPI_Type_free(&mpi_chunk_size);

                // Compute reception buffer size
                this->recv_size =
                  std::accumulate(begin(this->recv_chunks), end(this->recv_chunks), 0ul,
                                  [](const std::uint64_t& res, const chunk_t& chunk) { return res + chunk.size(); });
                recv_buffer.resize(recv_size);
                // Set the receive chunks start pointer
                recv_chunks[0].first = recv_buffer.data();
                std::partial_sum(begin(recv_chunks), end(recv_chunks), begin(recv_chunks),
                                 [](chunk_t& a, const chunk_t& b) {
                                     return chunk_t{a.end(), b.size()};
                                 });
            }

            /**
             * \brief Send and receive data to all processes individually.
             */
            void do_comms()
            {
                const int proc_count = conf.comm.size();
                const int rank = conf.comm.rank();

                std::vector<mpi::request> requests;
                requests.reserve(2 * proc_count);

                for(int i = 0; i < proc_count; ++i)
                {
                    if(0 == recv_chunks[i].size() || i == rank)
                    {
                        continue;   // Skip empty or local comm
                    }
                    requests.push_back(conf.comm.irecv(recv_chunks[i].data(), static_cast<int>(recv_chunks[i].size()),
                                                       conf.datatype, i, this->ok_tag));
                }
                for(int i = 0; i < proc_count; ++i)
                {
                    if(0 == send_chunks[i].size() || i == rank)
                    {
                        continue;   // Skip empty or local comm
                    }
                    requests.push_back(conf.comm.isend(send_chunks[i].data(), static_cast<int>(send_chunks[i].size()),
                                                       conf.datatype, i, this->ok_tag));
                }
                // Copy the elements that are already on the current process
                std::copy(std::begin(send_chunks[rank]), std::end(send_chunks[rank]), std::begin(recv_chunks[rank]));

                mpi::request::waitall(requests.size(), requests.data());
            }
        };

        /**
         * \brief Helper function to deduce the distribution_state type from arguments.
         *
         * \warning Since the copy constructor of distribution_state is deleted, this
         * method cannot be used to create a distribution state. C++17 lifts this
         * restriction with garanteed copy elision.
         *
         * See documentation of details::distribution_state.
         *
         * \param conf  MPI environment configuration
         * \param first Iterator to local range first element
         * \param last  Local range end iterator
         * \param distrib Distribution object
         *
         * \tparam T Type of the objects to send
         * \tparam ForwardIt Local range iterator
         * \tparam Distribution Distribution configuration.
         */
        template<class ForwardIt, class Distribution, class T = typename std::iterator_traits<ForwardIt>::value_type>
        details::distribution_state<T, ForwardIt, Distribution> create_state(inria::mpi_config conf, ForwardIt first,
                                                                             ForwardIt last, Distribution&& distrib)
        {
            return {conf, first, last, distrib};
        }

    }   // namespace details

    /**
     * \brief Function object returning 1 for any argument
     */
    struct unit_weight
    {
        /**
         * \brief Call operator, returns 1
         *
         * \tparam Ts Parameter types
         */
        template<class... Ts>
        double operator()(const Ts&...)
        {
            return 1;
        }
    };

    /**
     * \brief Indexable object returning 1 for any argument
     */
    struct uniform_proportions
    {
        /**
         * \brief Index operator, returns 1
         *
         * \tparam T Parameter type
         */
        template<class T>
        double operator[](const T&)
        {
            return 1;
        }
    };

    /**
     * \brief Callable object for custom proportional distribution.
     *
     * Implements the `Distribute` concept for the inria::distribute function. Upon
     * creation, this object scans the target range to compute the final ideal
     * weight distribution. Each subsequent call to `operator()` adds the given
     * element weight to an accummulator and returns the element selected target
     * process rank.
     *
     * \tparam Weight Weight function object used to find out elements distribution.
     *
     * \see details::distribution_state
     *
     * ~~~{.cpp}
     * std::vector<T> objs = {...};
     * // For 3 processes, 16.67%, 33.33%, 50%
     * std::array<int, 3> proportions = {100,200,300}
     * proportional_distribution<> distrib(conf, objs, proportions);
     * distribute(conf, objs, distrib);
     * ~~~
     */
    template<class Weight = unit_weight>
    struct proportional_distribution
    {
        /// Weight function object
        Weight weight;
        /// Local range weight
        std::uint64_t local_weight = 0;
        /// Range weight of all processes up to current one (excluded)
        std::uint64_t cumul_weight = 0;
        /// Global range weight
        std::uint64_t total_weight = 0;
        /// Ideal cumulated weight for each process
        std::vector<std::uint64_t> target_weight;

        /// Target rank returned by operator()
        std::uint64_t target_rank = 0;
        /// Number of processes
        int proc_count = 0;

        /**
         * \brief Create a propotional distribution for a distributed range
         *
         * Scans a distributed range to compute the global weight. Succesive calls
         * to operator() yield the process rank of given elements.
         *
         * \param conf  MPI configuration object.
         * \param first Iterator to local rank first element.
         * \param last  Iterator passed local rank last element.
         * \param proportions Array like object that holds proportions for each object.
         * \param w_    Function object to get an element weight
         *
         * \tparam ElemIt Iterator type of the element range
         * \tparam Proportion
         * \parblock
         * Array like type
         *
         * Proportion must specify an operator[] that returns the final proportion
         * of the global range that given process must hold. The sum of the
         * proportions is used.
         * \endparblock
         */
        template<class ElemIt, class Proportion>
        proportional_distribution(mpi_config conf, ElemIt first, ElemIt last, Proportion&& proportions, Weight w_ = {})
          : weight(w_)
          , target_weight(static_cast<std::uint64_t>(conf.comm.size()))
          , proc_count{conf.comm.size()}
        {
            // Compute all the weights
            using T = typename std::iterator_traits<ElemIt>::value_type;
            auto add_weight = [&](std::uint64_t w, const T& e) { return w + get_weight(e); };
            // Compute local weight
            local_weight = std::accumulate(first, last, 0ul, add_weight);
            // Sum weights of previous and current processes
            conf.comm.scan(&local_weight, &cumul_weight, 1, MPI_UINT64_T, MPI_SUM);
            // Last process broadcasts its cumulated weight
            total_weight = cumul_weight;
            conf.comm.bcast(&total_weight, 1, MPI_UINT64_T, proc_count - 1);
            // Correct cumulated weight to not include local range weight
            cumul_weight -= local_weight;

            using proportion_t = typename std::decay<decltype(proportions[0])>::type;
            proportion_t total_distri = 0;
            for(std::uint64_t i = 0; i < target_weight.size(); ++i)
            {
                total_distri += proportions[i];
            }

            proportion_t total_weight_p_t = static_cast<proportion_t>(total_weight);

            target_weight[0] = static_cast<std::uint64_t>(total_weight_p_t * proportions[0] / total_distri);
            for(std::uint64_t i = 1; i < target_weight.size(); ++i)
            {
                target_weight[i] =
                  target_weight[i - 1] + static_cast<std::uint64_t>(total_weight_p_t * proportions[i] / total_distri);
            }

            auto remaining_weight = total_weight - target_weight[proc_count - 1];
            if(remaining_weight > 0)
            {
                auto remaining_div = remaining_weight / proc_count;
                auto remaining_mod = remaining_weight % proc_count;
                for(std::uint64_t i = 0; i < target_weight.size(); ++i)
                {
                    target_weight[i] += (remaining_div * (i + 1)) + std::min(i + 1, remaining_mod);
                }
            }
        }

        /**
         * \brief Create a propotional distribution for a distributed range
         *
         * Scans a distributed range to compute the global weight. Succesive calls
         * to operator() yield the process rank of given elements.
         *
         * \param conf MPI configuration object.
         * \param r    Local range
         * \param p    Array like object that holds proportions for each object.
         * \param w    Function object to get an element weight
         *
         * \tparam ElemIt Iterator type of the element range
         * \tparam Proportion
         * \parblock
         * Array like type
         *
         * Proportion must specify an operator[] that returns the final proportion
         * of the global range that given process must hold. The sum of the
         * proportions is used.
         * \endparblock
         */
        template<class Range, class Proportion>
        proportional_distribution(mpi_config conf, const Range& r, Proportion&& p, Weight w = {})
          : proportional_distribution(conf, std::begin(r), std::end(r), std::forward<Proportion>(p), w)
        {
        }

        /**
         * \brief Compute target process of an element
         *
         * The object's accumulator #cumul_weight is increased by `e`'s weight at
         * each call. The target rank is then chosen according to #cumul_weight.
         *
         * \param e Element of which to compute the target
         *
         * \tparam T Type of `e`
         */
        template<class T>
        std::uint64_t operator()(const T& e)
        {
            const std::uint64_t proc_count_minus_one = proc_count - 1;
            while((target_rank < proc_count_minus_one) && (this->cumul_weight >= target_weight[target_rank]))
            {
                ++target_rank;
            }
            this->cumul_weight += get_weight(e);
            return target_rank;
        }

        /**
         * \brief Helper to call #weight
         *
         * This mainly factorizes the static_cast of `weight(e)`
         *
         * \param e Element to get the weight of
         *
         * \tparam T Type of `e`
         */
        template<class T>
        std::uint64_t get_weight(const T& e)
        {
            return static_cast<std::uint64_t>(weight(e));
        }
    };

    /**
     * \brief Callable object for uniform distribution.
     *
     * Implements the `Distribute` concept for the inria::distribute function.
     *
     * \tparam Weight Weight function object used to find out elements distribution.
     *
     * \see details::proportional_distribution
     * \see details::distribution_state
     *
     * ~~~[cpp]
     * std::vector<T> objs = {...};
     * distribute(conf, objs, uniform_distribution{conf, objs});
     * ~~~
     */
    template<class Weight = unit_weight>
    struct uniform_distribution : proportional_distribution<Weight>
    {
        /**
         * \brief Constructor from range iterators
         *
         * \param conf  MPI configuration object.
         * \param first Iterator to local rank first element.
         * \param last  Iterator passed local rank last element.
         * \param w     Function object to get an element weight
         *
         * \tparam ElemIt Iterator type
         */
        template<class ElemIt>
        uniform_distribution(mpi_config conf, ElemIt first, ElemIt last, Weight w = {})
          : proportional_distribution<Weight>(conf, first, last, uniform_proportions{}, w)
        {
        }

        /**
         * \brief Constructor from range
         *
         * \param conf MPI configuration object.
         * \param r    Local range
         * \param w    Function object to get an element weight
         *
         * \tparam Range Range type
         */
        template<class Range>
        uniform_distribution(mpi_config conf, const Range& r, Weight w = {})
          : proportional_distribution<Weight>(conf, r, uniform_proportions{}, w)
        {
        }
    };

    /**
     * \brief Redistribute global range among processes
     *
     * The global range is shared among the processes and must be
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf  MPI environment
     * \param first Local range begin iterator
     * \param last  Local range end iterator
     * \param out_first Output local range begin iterator
     * \param out_last  Output local range end iterator
     * \param distrib Distribution object
     *
     * \tparam ForwardIt Forward input iterator
     * \tparam ForwardOutputIterator Forward output iterator
     * \tparam Distribution Distribution object type
     * \parblock
     * Must provide a call operator returning the rank of the process the given
     * element belongs to.
     *
     * ~~~{.cpp}
     * int Distribution::operator()(const T&);
     * ~~~
     * \endparblock
     *
     * \return An iterator to the new end of the output range.
     *
     * \exception std::runtime_error is thrown if the output range is too small.
     *
     * Example:
     *
     * ~~~{.cpp}
     * // Distribute data from process 0 to the others
     *
     * // MPI communicator
     * auto world = inria::mpi::communicator::world();
     *
     * // All processes have a large enough buffer
     * int output[100] = {};
     * // Input is an empty vector, no allocation is done
     * std::vector<int> input{};
     *
     * // Process 0 fills input with data
     * if(world.rank() == 0) {
     *     input.resize(100);
     *     std::iota(begin(input), end(input), 0);
     * }
     *
     * // Distribution object, keeps all even elements on 0, all odd elements on 1
     * struct dist {
     *     int operator()(const int& i) {
     *         return i % 2;
     *     }
     * };
     *
     * try {
     *     distribute(world, begin(input), end(input), output, output + 100, dist{});
     * } catch(std::out_of_range& e) {
     *     std::cerr << e.what() << '\n';
     * }
     * ~~~
     */
    template<class ForwardIt, class ForwardOutputIterator, class Distribution,
#ifndef DOXYGEN_DOCUMENTATION
             class T = typename std::iterator_traits<ForwardIt>::value_type,
             class = disable_if_t<is_range<ForwardIt>::value>,
             class = enable_if_t<is_algo_distribution<typename std::remove_reference<Distribution>::type, T>::value>
#endif
             >
    ForwardOutputIterator distribute(inria::mpi_config conf, ForwardIt first, ForwardIt last,
                                     ForwardOutputIterator out_first, ForwardOutputIterator out_last,
                                     Distribution&& distrib)
    {
        using state_t = decltype(details::create_state(conf, first, last, distrib));
        state_t state{conf, first, last, distrib};

        // Once comms are set up, check whether the output range is big enough.
        std::uint64_t output_capacity = std::distance(out_first, out_last);
        char output_large_enough = output_capacity >= state.recv_size;
        char global_large_enough = 1;
        conf.comm.allreduce(&output_large_enough, &global_large_enough, 1, MPI_CHAR, MPI_LAND);
        if(!global_large_enough)
        {
            if(!output_large_enough)
            {
                throw std::runtime_error("Process output range is too small.");
            }
            else
            {
                throw std::runtime_error("Other process output range is too small.");
            }
        }

        state.do_comms();

        return std::copy(std::begin(state.recv_buffer), std::end(state.recv_buffer), out_first);
    }

    /**
     * \brief Evenly redistribute global range among processes
     *
     * The global range is shared among the processes and must be evenly
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf  MPI environment
     * \param first Local range begin iterator
     * \param last  Local range end iterator
     * \param out_first Output local range begin iterator
     * \param out_last  Output local range end iterator
     *
     * \tparam ForwardIt Forward input iterator
     * \tparam OutputIterator Forward output iterator
     *
     * \return An iterator to the new end of the output range.
     *
     * \exception std::runtime_error is thrown if the output range is too small.
     *
     * Example:
     *
     * ~~~{.cpp}
     * // Distribute data from process 0 to the others
     *
     * // MPI communicator
     * auto world = inria::mpi::communicator::world();
     *
     * // All processes have a large enough buffer
     * int output[100] = {};
     * // Input is an empty vector, no allocation is done
     * std::vector<int> input{};
     *
     * // Process 0 fills input with data
     * if(world.rank() == 0) {
     *     input.resize(100);
     *     std::iota(begin(input), end(input), 0);
     * }
     *
     * try {
     *     distribute(world, begin(input), end(input), output, output + 100);
     * } catch(std::out_of_range& e) {
     *     std::cerr << e.what() << '\n';
     * }
     * ~~~

     */
    template<class ForwardIt, class OutputIterator,
#ifndef DOXYGEN_DOCUMENTATION
             class T = typename std::iterator_traits<ForwardIt>::value_type,
             class = disable_if_t<is_range<ForwardIt>::value>
#endif
             >
    OutputIterator distribute(inria::mpi_config conf, ForwardIt first, ForwardIt last, OutputIterator out_first,
                              OutputIterator out_last)
    {
        return distribute(conf, first, last, out_first, out_last, uniform_distribution<>{conf, first, last});
    }

    /**
     * \brief Redistribute global range among processes
     *
     * The global range is shared among the processes and must be
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf MPI environment
     * \param in   Input range
     * \param out_first Output local range begin iterator
     * \param out_last  Output local range end iterator
     * \param distrib Distribution object
     *
     * \tparam InRange Input range
     * \parblock
     * Must satisfy the usual range requirements, tested using inria::is_range.
     * \endparblock
     * \tparam OutputIterator Forward output iterator
     * \tparam Distribution Distribution object type
     * \parblock
     * Must provide a call operator returning the rank of the process the given
     * element belongs to.
     *
     * ~~~{.cpp}
     * int Distribution::operator()(const T&);
     * ~~~
     * \endparblock
     *
     * \return An iterator to the new end of the output range.
     *
     * \exception std::runtime_error is thrown if the output range is too small.
     */
    template<class InRange, class OutputIterator, class Distribution,
#ifndef DOXYGEN_DOCUMENTATION
             class = enable_if_t<is_range<typename std::remove_reference<InRange>::type>::value>,
             class T = range_element_t<typename std::remove_reference<InRange>::type>,
             class = disable_if_t<is_range<OutputIterator>::value>,
             class = enable_if_t<is_algo_distribution<typename std::remove_reference<Distribution>::type, T>::value>
#endif
             >
    OutputIterator distribute(inria::mpi_config conf, InRange&& in, OutputIterator out_first, OutputIterator out_last,
                              Distribution&& distrib)
    {
        using std::begin;
        using std::end;
        return distribute(conf, begin(in), end(in), out_first, out_last, std::forward<Distribution>(distrib));
    }

    /**
     * \brief Evenly redistribute global range among processes
     *
     * The global range is shared among the processes and must be
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf MPI environment
     * \param in   Input range
     * \param out_first Output local range begin iterator
     * \param out_last  Output local range end iterator
     *
     * \tparam InRange Input range
     * \parblock
     * Must satisfy the usual range requirements, tested using inria::is_range.
     * \endparblock
     * \tparam OutputIterator Forward output iterator
     *
     * \return An iterator to the new end of the output range.
     *
     * \exception std::runtime_error is thrown if the output range is too small.
     */
    template<class InRange, class OutputIterator,
#ifndef DOXYGEN_DOCUMENTATION
             class = enable_if_t<is_range<typename std::remove_reference<InRange>::type>::value>,
             class T = range_element_t<typename std::remove_reference<InRange>::type>,
             class = disable_if_t<is_range<OutputIterator>::value>
#endif
             >
    OutputIterator distribute(inria::mpi_config conf, InRange&& in, OutputIterator out_first, OutputIterator out_last)
    {
        return distribute(conf, in, out_first, out_last, uniform_distribution<>{conf, in});
    }

    /**
     * \brief Redistribute global range among processes
     *
     * The global range is shared among the processes and must be
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf  MPI environment
     * \param first Local range begin iterator
     * \param last  Local range end iterator
     * \param out   Output range
     * \param distrib Distribution object
     *
     * \tparam ForwardIt Forward input iterator
     * \tparam AssignableOutRange Output range
     * \parblock
     * Must satisfy the usual range requirements, tested using inria::is_range and
     * the provide an `assign` method that sets the range from an iterator defined
     * range.
     * \endparblock
     * \tparam Distribution Distribution object type
     * \parblock
     * Must provide a call operator returning the rank of the process the given
     * element belongs to.
     *
     * ~~~{.cpp}
     * int Distribution::operator()(const T&);
     * ~~~
     * \endparblock
     */
    template<class ForwardIt, class AssignableOutRange, class Distribution,
#ifndef DOXYGEN_DOCUMENTATION
             class T = typename std::iterator_traits<ForwardIt>::value_type,
             class = enable_if_t<is_range<AssignableOutRange>::value>,
             class = enable_if_t<is_assignable<AssignableOutRange, ForwardIt, ForwardIt>::value>,
             class = enable_if_t<is_algo_distribution<typename std::remove_reference<Distribution>::type, T>::value>
#endif
             >
    void distribute(inria::mpi_config conf, ForwardIt first, ForwardIt last, AssignableOutRange& out,
                    Distribution&& distrib)
    {
        using std::begin;
        using std::end;
        using state_t = decltype(details::create_state(conf, first, last, distrib));
        state_t state{conf, first, last, distrib};
        state.do_comms();
        out.assign(begin(state.recv_buffer), end(state.recv_buffer));
    }

    /**
     * \brief Evenly redistribute global range among processes
     *
     * The global range is shared among the processes and must be
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf  MPI environment
     * \param first Local range begin iterator
     * \param last  Local range end iterator
     * \param out   Output range
     *
     * \tparam ForwardIt Forward input iterator
     * \tparam AssignableOutRange Output range
     * \parblock
     * Must satisfy the usual range requirements, tested using inria::is_range and
     * the provide an `assign` method that sets the range from an iterator defined
     * range.
     * \endparblock
     *
     * Example:
     *
     * ~~~{.cpp}
     * // Distribute data from process 0 to the others
     *
     * // MPI communicator
     * auto world = inria::mpi::communicator::world();
     *
     * // All processes have an assignable output buffer
     * std::vector<int> output{};
     * // Input is an empty vector, no allocation is done
     * std::vector<int> input{};
     *
     * // Process 0 fills input with data
     * if(world.rank() == 0) {
     *     input.resize(100);
     *     std::iota(begin(input), end(input), 0);
     * }
     *
     * distribute(world, begin(input), end(input), output);
     */
    template<class ForwardIt, class AssignableOutRange,
#ifndef DOXYGEN_DOCUMENTATION
             class T = typename std::iterator_traits<ForwardIt>::value_type,
             class = enable_if_t<is_range<AssignableOutRange>::value>,
             class = enable_if_t<is_assignable<AssignableOutRange, ForwardIt, ForwardIt>::value>
#endif
             >
    void distribute(inria::mpi_config conf, ForwardIt first, ForwardIt last, AssignableOutRange& out)
    {
        distribute(conf, first, last, out, uniform_distribution<>{conf, first, last});
    }

    /**
     * \brief Redistribute global range among processes
     *
     * The global range is shared among the processes and must be
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf MPI environment
     * \param in   Input range
     * \param out  Output range
     * \param distrib Distribution object
     *
     * \tparam InRange Input range
     * \parblock
     *
     * Must satisfy the usual range requirements, tested using inria::is_range.
     * \endparblock
     * \tparam AssignableOutRange Output range
     * \parblock
     *
     * Must satisfy the usual range requirements, tested using inria::is_range and
     * the provide an `assign` method that sets the range from an iterator defined
     * range.
     * \endparblock
     * \tparam Distribution Distribution object type
     * \parblock
     *
     * Must provide a call operator returning the rank of the process the given
     * element belongs to.
     *
     * ~~~{.cpp}
     * int Distribution::operator()(const T&);
     * ~~~
     * \endparblock
     */
    template<class InRange, class AssignableOutRange, class Distribution,
#ifndef DOXYGEN_DOCUMENTATION
             class = enable_if_t<is_range<typename std::remove_reference<InRange>::type>::value>,
             class T = range_element_t<typename std::remove_reference<InRange>::type>,
             class = enable_if_t<is_range<AssignableOutRange>::value>,
             class = enable_if_t<is_assignable<AssignableOutRange, T*, T*>::value>,
             class = enable_if_t<is_algo_distribution<typename std::decay<Distribution>::type, T>::value>
#endif
             >
    void distribute(inria::mpi_config conf, InRange&& in, AssignableOutRange& out, Distribution&& distrib)
    {
        using std::begin;
        using std::end;
        distribute(conf, begin(in), end(in), out, distrib);
    }

    /**
     * \brief Redistribute global range among processes
     *
     * The global range is shared among the processes and must be
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf MPI environment
     * \param in   Input range
     * \param out  Output range
     *
     * \tparam InRange Input range
     * \parblock
     *
     * Must satisfy the usual range requirements, tested using inria::is_range.
     * \endparblock
     * \tparam AssignableOutRange Output range
     * \parblock
     *
     * Must satisfy the usual range requirements, tested using inria::is_range and
     * the provide an `assign` method that sets the range from an iterator defined
     * range.
     * \endparblock
     */
    template<class InRange, class AssignableOutRange,
#ifndef DOXYGEN_DOCUMENTATION
             class = enable_if_t<is_range<typename std::remove_reference<InRange>::type>::value>,
             class T = range_element_t<typename std::remove_reference<InRange>::type>,
             class = enable_if_t<is_range<AssignableOutRange>::value>,
             class = enable_if_t<is_assignable<AssignableOutRange, T*, T*>::value>
#endif
             >
    void distribute(inria::mpi_config conf, InRange&& in, AssignableOutRange& out)
    {
        using std::begin;
        using std::end;
        distribute(conf, begin(in), end(in), out, uniform_distribution<>{conf, begin(in), end(in)});
    }

    /**
     * \brief Redistribute global range among processes
     *
     * The global range is shared among the processes and must be
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf  MPI environment
     * \param inout Input and output range
     * \param distrib Distribution object
     *
     * \tparam AssignableInOutRange Input and output range
     * \parblock
     *
     * Must satisfy the usual range requirements, tested using inria::is_range and
     * the provide an `assign` method that sets the range from an iterator defined
     * range.
     * \endparblock
     * \tparam Distribution Distribution object type
     * \parblock
     *
     * Must provide a call operator returning the rank of the process the given
     * element belongs to.
     *
     * ~~~{.cpp}
     * int Distribution::operator()(const T&);
     * ~~~
     * \endparblock
     */
    template<class AssignableInOutRange, class Distribution,
#ifndef DOXYGEN_DOCUMENTATION
             class = enable_if_t<is_range<AssignableInOutRange>::value>,
             class T = range_element_t<AssignableInOutRange>,
             class = enable_if_t<is_assignable<AssignableInOutRange, T*, T*>::value>,
             class = enable_if_t<is_algo_distribution<typename std::remove_reference<Distribution>::type, T>::value>
#endif
             >
    void distribute(inria::mpi_config conf, AssignableInOutRange& inout, Distribution&& distrib)
    {
        distribute(conf, inout, inout, distrib);
    }

    /**
     * \brief Evenly redistribute global range among processes
     *
     * The global range is shared among the processes and must be
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf  MPI environment
     * \param inout Input and output range
     *
     * \tparam AssignableInOutRange Input and output range
     * \parblock
     *
     * Must satisfy the usual range requirements, tested using inria::is_range and
     * the provide an `assign` method that sets the range from an iterator defined
     * range.
     * \endparblock
     */
    template<class AssignableInOutRange,
#ifndef DOXYGEN_DOCUMENTATION
             class = enable_if_t<is_range<AssignableInOutRange>::value>,
             class T = range_element_t<AssignableInOutRange>,
             class = enable_if_t<is_assignable<AssignableInOutRange, T*, T*>::value>
#endif
             >
    void distribute(inria::mpi_config conf, AssignableInOutRange& inout)
    {
        distribute(conf, inout, inout, uniform_distribution<>{conf, begin(inout), end(inout)});
    }

    /**
     * \brief Redistribute global range among processes
     *
     * The global range is shared among the processes and must be
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf  MPI environment
     * \param first Local range begin iterator
     * \param last  Local range end iterator
     * \param out   Output local range
     * \param distrib Distribution object
     *
     * \tparam ForwardIt Forward input iterator
     * \tparam NonAssignableOutRange Output range
     * \parblock
     *
     * Must satisfy the usual range requirements, tested using inria::is_range.
     * \endparblock
     * \tparam Distribution Distribution object type
     * \parblock
     * Must provide a call operator returning the rank of the process the given
     * element belongs to.
     *
     * ~~~{.cpp}
     * int Distribution::operator()(const T&);
     * ~~~
     * \endparblock
     *
     * \return An iterator to the new end of the output range.
     *
     * \exception std::runtime_error is thrown if the output range is too small.
     */
    template<class ForwardIt, class NonAssignableOutRange, class Distribution,
#ifndef DOXYGEN_DOCUMENTATION
             class T = typename std::iterator_traits<ForwardIt>::value_type,
             class = enable_if_t<is_range<NonAssignableOutRange>::value>,
             class = disable_if_t<is_assignable<NonAssignableOutRange, ForwardIt, ForwardIt>::value>,
             class = enable_if_t<is_algo_distribution<typename std::remove_reference<Distribution>::type, T>::value>,
             class AvoidTemplateRedeclarationError = void
#endif
             >
    auto distribute(inria::mpi_config conf, ForwardIt first, ForwardIt last, NonAssignableOutRange& out,
                    Distribution&& distrib) -> decltype(std::begin(out))
    {
        return distribute(conf, first, last, std::begin(out), std::end(out), distrib);
    }

    /**
     * \brief Evenly redistribute global range among processes
     *
     * The global range is shared among the processes and must be
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf  MPI environment
     * \param first Local range begin iterator
     * \param last  Local range end iterator
     * \param out   Output local range
     *
     * \tparam ForwardIt Forward input iterator
     * \tparam NonAssignableOutRange Output range
     * \parblock
     *
     * Must satisfy the usual range requirements, tested using inria::is_range.
     * \endparblock
     *
     * \return An iterator to the new end of the output range.
     *
     * \exception std::runtime_error is thrown if the output range is too small.
     */
    template<class ForwardIt, class NonAssignableOutRange,
#ifndef DOXYGEN_DOCUMENTATION
             class T = typename std::iterator_traits<ForwardIt>::value_type,
             class = enable_if_t<is_range<NonAssignableOutRange>::value>,
             class = disable_if_t<is_assignable<NonAssignableOutRange, ForwardIt, ForwardIt>::value>,
             class AvoidTemplateRedeclarationError = void
#endif
             >
    auto distribute(inria::mpi_config conf, ForwardIt first, ForwardIt last, NonAssignableOutRange& out)
      -> decltype(std::begin(out))
    {
        return distribute(conf, first, last, std::begin(out), std::end(out), uniform_distribution<>{conf, first, last});
    }

    /**
     * \brief Redistribute global range among processes
     *
     * The global range is shared among the processes and must be
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf  MPI environment
     * \param in   Input range
     * \param out   Output local range
     * \param distrib Distribution object
     *
     * \tparam InRange Input range
     * \parblock
     *
     * Must satisfy the usual range requirements, tested using inria::is_range.
     * \endparblock
     * \tparam NonAssignableOutRange Output range
     * \parblock
     *
     * Must satisfy the usual range requirements, tested using inria::is_range.
     * \endparblock
     * \tparam Distribution Distribution object type
     * \parblock
     *
     * Must provide a call operator returning the rank of the process the given
     * element belongs to.
     *
     * ~~~{.cpp}
     * int Distribution::operator()(const T&);
     * ~~~
     * \endparblock
     *
     * \return An iterator to the new end of the output range.
     *
     * \exception std::runtime_error is thrown if the output range is too small.
     */
    template<class InRange, class NonAssignableOutRange, class Distribution,
#ifndef DOXYGEN_DOCUMENTATION
             class = enable_if_t<is_range<typename std::remove_reference<InRange>::type>::value>,
             class T = range_element_t<typename std::remove_reference<InRange>::type>,
             class = enable_if_t<is_range<NonAssignableOutRange>::value>,
             class = disable_if_t<is_assignable<NonAssignableOutRange, T*, T*>::value>,
             class = enable_if_t<is_algo_distribution<typename std::remove_reference<Distribution>::type, T>::value>,
             class AvoidTemplateRedeclarationError = void
#endif
             >
    auto distribute(inria::mpi_config conf, InRange&& in, NonAssignableOutRange& out, Distribution&& distrib)
      -> decltype(std::begin(out))
    {
        return distribute(conf, std::begin(in), std::end(in), std::begin(out), std::end(out), distrib);
    }

    /**
     * \brief Evenly redistribute global range among processes
     *
     * The global range is shared among the processes and must be
     * redistributed. Each process part of the global range is called the local
     * range.
     *
     * The order of the elements is kept. No guarantee is made for elements that
     * compare equal.
     *
     * \note Current implementation keeps local order and if two elements compare
     * equal, the one that was originally on the lowest ranked process is placed
     * first.
     *
     * The elements are copied to the output range. The original local range is kept
     * untouched.
     *
     * \warning It is the developper responsibility to provide an output range large
     * enough to store the results. A check is done using `out_last - out_first`
     *
     * \param conf  MPI environment
     * \param in   Input range
     * \param out   Output local range
     *
     * \tparam InRange Input range
     * \parblock
     *
     * Must satisfy the usual range requirements, tested using inria::is_range.
     * \endparblock
     * \tparam NonAssignableOutRange Output range
     * \parblock
     *
     * Must satisfy the usual range requirements, tested using inria::is_range.
     * \endparblock
     *
     * \return An iterator to the new end of the output range.
     *
     * \exception std::runtime_error is thrown if the output range is too small.
     */
    template<class InRange, class NonAssignableOutRange,
#ifndef DOXYGEN_DOCUMENTATION
             class = enable_if_t<is_range<typename std::remove_reference<InRange>::type>::value>,
             class T = range_element_t<typename std::remove_reference<InRange>::type>,
             class = enable_if_t<is_range<NonAssignableOutRange>::value>,
             class = disable_if_t<is_assignable<NonAssignableOutRange, T*, T*>::value>,
             class AvoidTemplateRedeclarationError = void
#endif
             >
    auto distribute(inria::mpi_config conf, InRange&& in, NonAssignableOutRange& out) -> decltype(std::begin(out))
    {
        using std::begin;
        using std::end;
        return distribute(conf, begin(in), end(in), begin(out), end(out),
                          uniform_distribution<>{conf, begin(in), end(in)});
    }

}   // namespace inria

#endif /* _DISTRIBUTE_HPP_ */
