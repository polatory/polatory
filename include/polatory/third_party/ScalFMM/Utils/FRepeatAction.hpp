#ifndef _FPROGRESSBAR_HPP_
#define _FPROGRESSBAR_HPP_

#include <thread>
#include <chrono>

#include "../Utils/FTic.hpp"

/** \brief Calls a function at time interval until stopped.
 *
 * This object will call at a defined interval any function or lambda that is
 * passed to it. The function can stop the call by returning false.
 *
 * For convenience, the class inherits from FTic to measure time.
 */
class FRepeatAction : public FTic {
    /// Type definition for the function that is run.
    using function = std::function<bool()>;
    /// Sleep time type definition.
    using time_type = std::chrono::milliseconds;

    /// Thread stop condition.
    bool _continue = false;
    
    /// Thread running state (true if running).
    bool _running = false;

    /// Function to call
    function _f;

    /// Time between each call to #_f.
    time_type _sleepTime;

    /// The thread that calls #_f repeateddly.
    std::thread _thread;

    /// #_thread main loop
    static void run( function f, bool& cont, time_type& sleepTime) {
        while( f() && cont ) {
            if(sleepTime == time_type{0}) {
                std::cerr << "Cannot have a null sleep time." << std::endl;
                break;
            }
            std::this_thread::sleep_for(sleepTime);
        }
    }


public:
    /// Deleted default constructor.
    FRepeatAction() = delete;
 
    /// Deleted copy constructor.
    /** A thread cannot be copied. */
    FRepeatAction( const FRepeatAction& other ) = delete;

    /// Deleted move constructor.
    /** Calls in #run prevent moving things around. */
    FRepeatAction( FRepeatAction&& other ) = default;

    /**\brief Constructor.
     * 
     * \param f Callable object or function to call repeatedly.
     * \param sleepTime Time interval between each call.
     * \param start_now If true, start calling f immediately.
     */
    FRepeatAction( function f, int sleepTime, bool start_now = true ) : 
        _f(f),
        _sleepTime(sleepTime) {
        if(start_now)
            start();
    }

    /// Destructor makes a call to #stop.
    ~FRepeatAction() {
        stop();
    }

    /// Stops the repeated action and joins with #_thread.
    void stop() {
        _continue = false;
        FTic::tac();
        if(_thread.joinable())
            _thread.join();
        _running = false;
    }

    /// Starts the repeated action thread.
    void start() {
        if( _running ) {
            std::cerr << __FUNCTION__ << "A thread is already running." << std::endl;
        } else if( _sleepTime == time_type{0} ) {
            std::cerr << "Cannot have a null sleep time." << std::endl;
        } else {
            _running = true;
            _continue = true;
            FTic::tic();
            _thread = std::thread(run, _f, std::ref(_continue), std::ref(_sleepTime));
        }        
    }

    /// Sets the function that is repeatedly called.
    /** \warning The object must be stopped before using this method. */
    void setFunction( function f ) {
        if( _running ) {
            std::cerr << __FUNCTION__ << "A thread is already running." << std::endl;
        } else {
            FTic::reset();
            _f = f;
        }
    }

    /// Sets the time interval between calls to #_f.
    /** \warning The object must be stopped before using this method. */
    void setSleepTime(int sleepTime) {
        if( _running ) {
            std::cerr << __FUNCTION__ << "A thread is already running." << std::endl;
        } else {
            _sleepTime = time_type{sleepTime};
        }
    }

};




#endif
