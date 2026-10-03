// --------------------------------
// See LICENCE file at project root
// File : utils/tic.hpp
// --------------------------------
#ifndef CPP_TOOLS_TIMER_TIMER_HPP
#define CPP_TOOLS_TIMER_TIMER_HPP

#include <chrono>

namespace cpp_tools::timers
{
    template<class Duration = std::chrono::milliseconds>
    class timer
    {
      public:
        using clock = std::chrono::high_resolution_clock;
        using time_point = typename clock::time_point;
        using duration = Duration;
        using rep = typename duration::rep;

        /**\warning Use tic() to restart the timer. */
        inline auto reset() -> void
        {
            start = clock::now();
            end = start;
            cumulate = 0;
        }

        /// Start measuring time.
        inline auto tic() -> void { start = clock::now(); }

        /// Peek at current elapsed time without stopping timer
        [[nodiscard]] inline auto peek() const -> double
        {
            const auto time = clock::now() - start;
            return time.count();
        }

        inline auto tac() -> void
        {
            end = clock::now();
            cumulate += elapsed();
        }

        /// Elapsed time between the last tic() and tac() (in seconds).
        /** \return the time elapsed between tic() & tac() in second. */
        [[nodiscard]] inline auto elapsed() const -> rep
        {
            auto elapsed{std::chrono::duration_cast<duration>(end - start)};
            return elapsed.count();
        }

        [[nodiscard]] inline auto tac_and_elapsed() -> rep
        {
            tac();
            return elapsed();
        }

        /// Cumulated tic() - tac() time spans
        /** \return the time elapsed between ALL tic() & tac() in second. */
        [[nodiscard]] inline auto cumulated() const -> rep { return cumulate; }

        inline auto operator+(timer const& other) const -> timer
        {
            timer res(*this);
            res.cumulate += other.cumulate;
            return res;
        }

        inline auto operator-(timer const& other) const -> timer
        {
            timer res(*this);
            res.cumulate -= other.cumulate;
            return res;
        }

      private:
        time_point start{clock::now()};   ///< start time (tic)
        time_point end{start};            ///< stop time (tac)
        rep cumulate{rep(0.)};            ///< cumulated duration
    };
}   // namespace scalfmm::utils

#endif   // SCALFMM_UTILS_TIMER_HPP
