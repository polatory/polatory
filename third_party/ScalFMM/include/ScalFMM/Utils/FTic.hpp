// See LICENCE file at project root
#ifndef FTIC_HPP
#define FTIC_HPP

#include "FGlobal.hpp"

#define USE_STD_CHRONO

#if defined(USE_STD_CHRONO)
    #include <chrono>
#elif defined(_OPENMP)
    #include <omp.h>
#elif defined(WINDOWS) // We need an os specific function
    #include <time.h>
    #include <windows.h>
#else
    #ifndef POSIX
        #warning Posix used withoug being explicitly defined
    #endif
    #include <time.h>
    #include <sys/time.h>
    #include <unistd.h>
    #include <stdint.h>
#endif


/**
 * \brief Time counter class.
 * \author Berenger Bramas (berenger.bramas@inria.fr)
 *
 * This time counter can be (re)started using tic() and stopped using tac().
 *
 *  - use elapsed() to get the last time interval;
 *  - use cumulated() to get the total running time;
 *  - use reset() to stop and reset the counter.
 *
 * \code
 * FTic timer;
 * timer.tic();
 * //...(1)
 * timer.tac();
 * timer.elapsed();  // time of (1) in s
 * timer.tic();
 * //...(2)
 * timer.tac();
 * timer.elapsed();  // time of (2) in s
 * timer.cumulated() // time of (1) and (2) in s
 * timer.reset()     // reset the object
 * \endcode
 *
 * The special method that uses asm register is based on code by Alexandre DENIS
 * http://dept-info.labri.fr/~denis/Enseignement/2006-SSECPD/timing.h
 */
class FTic {
private:

    double start    = 0;    ///< start time (tic)
    double end      = 0;    ///< stop time (tac)
    double cumulate = 0;    ///< cumulated duration

public:
    /// Constructor
    FTic() {
        this->reset();
    }

    /// Copy constructor
    FTic(const FTic& other) = default;

    /// Move constructor
    FTic(FTic&& other) = default;

    /// Copies an other timer
    FTic& operator=(const FTic& other) {
        start = other.start;
        end = other.end;
        cumulate = other.cumulate;
        return *this;
    }

    /// Adds two timers
    /** The addition is done by keeping :
     *     - the left operand start date
     *     - the left operand end date
     *     - adding the cumulated times
     * \return A new FTic object
     */
    const FTic operator+(const FTic& other) const {
        FTic res(*this);
        res.cumulate += other.cumulate;
        return res;
    }

    /// Resets the timer
    /**\warning Use tic() to restart the timer. */
    void reset() {
        start    = FTic::GetTime();
        end      = start;
        cumulate = 0;
    }

    /// Start measuring time.
    void tic(){
        this->start = FTic::GetTime();
    }

    /// Peek at current elapsed time without stopping timer
    double peek() const {
        return FTic::GetTime() - this->start;;
    }

    /// Stop measuring time and add to cumulated time.
    double tac(){
        this->end = FTic::GetTime();
        auto lapse = this->elapsed();
        cumulate += lapse;
        return lapse;
    }

    /// Elapsed time between the last tic() and tac() (in seconds).
    /** \return the time elapsed between tic() & tac() in second. */
    double elapsed() const {
        return this->end - this->start;
    }

    /// Cumulated tic() - tac() time spans
    /** \return the time elapsed between ALL tic() & tac() in second. */
    double cumulated() const {
        return cumulate;
    }

    /// Combination of tic() and elapsed().
    /**
     * \todo deprecate
     * \return the time elapsed between tic() & tac() in second. */
    double tacAndElapsed() {
        tac();
        return elapsed();
    }

    /// Get system dependent time point.
    /** GetTickCount on windows
     *  gettimeofday on linux or a direct ASM method
     *  \return A system dependent time point.
     */
    static double GetTime(){
#if defined(USE_STD_CHRONO)
        using clock = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<double>;
        return duration(clock::now().time_since_epoch()).count();
#elif defined(_OPENMP)
        return omp_get_wtime();
#elif defined(WINDOWS)
        return static_cast<double>(GetTickCount())/1000.0;
#else // We are in linux/posix
        timeval t;
        gettimeofday(&t, NULL);
        return double(t.tv_sec) + (double(t.tv_usec)/1000000.0);
#endif
    }
};


#endif
