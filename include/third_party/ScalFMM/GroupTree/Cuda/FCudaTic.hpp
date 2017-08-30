#ifndef FCUDATIC_HPP
#define FCUDATIC_HPP

#include "FCudaGlobal.hpp"


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
 * FCudaTic timer;
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
 */
class FCudaTic {
private:

    cudaStream_t stream;
    cudaEvent_t start    = 0;    ///< start time (tic)
    cudaEvent_t end      = 0;    ///< stop time (tac)
    double cumulate = 0;    ///< the cumulate time

public:
    /// Constructor
    explicit FCudaTic(const cudaStream_t inStream = 0)
        : stream(inStream){
        FCudaCheck(cudaEventCreate(&start));
        FCudaCheck(cudaEventCreate(&end));
        tic();
    }

    ~FCudaTic(){
        FCudaCheck( cudaEventDestroy( start ) );
        FCudaCheck( cudaEventDestroy( end ) );
    }

    /// Copy constructor
    FCudaTic(const FCudaTic& other) = delete;
    /// Copy operator
    FCudaTic& operator=(const FCudaTic& other) = delete;


    /// Resets the timer
    /**\warning Use tic() to restart the timer. */
    void reset() {
        cumulate = 0;
    }

    /// Start measuring time.
    void tic(){
        FCudaCheck(cudaEventRecord( start, stream ));
    }

    /// Stop measuring time and add to cumulated time.
    void tac(){
        FCudaCheck(cudaEventRecord( end, stream ));
        FCudaCheck(cudaEventSynchronize( end ));
        cumulate += elapsed();
    }

    /// Elapsed time between the last tic() and tac() (in seconds).
    /** \return the time elapsed between tic() & tac() in second. */
    double elapsed() const{
        float elapsedTime;
        FCudaCheck( cudaEventElapsedTime( &elapsedTime, start, end ) ); // in ms
        return elapsedTime/1000.0;
    }

    /// Cumulated tic() - tac() time spans
    /** \return the time elapsed between ALL tic() & tac() in second. */
    double cumulated() const{
        return cumulate;
    }

    /// Combination of tic() and elapsed().
    /** \return the time elapsed between tic() & tac() in second. */
    double tacAndElapsed() {
        tac();
        return elapsed();
    }
};


#endif // FCUDATIC_HPP

