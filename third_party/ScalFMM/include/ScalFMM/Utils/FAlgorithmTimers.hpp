// See LICENCE file at project root

#ifndef FALGORITHMTIMERS_HPP
#define FALGORITHMTIMERS_HPP

#include <map>
#include <string>
#include <stdexcept>

#include "FTic.hpp"

using FTimerMap = std::map<std::string, FTic>;

/**
 * @brief Collection of timers for FMM operators.
 *
 * This class provide a way for the different algorithms to
 * store the time spent in each operator.
 */
class FAlgorithmTimers{
public:
    static constexpr const char* P2MTimer = "P2M";
    static constexpr const char* M2MTimer = "M2M";
    static constexpr const char* M2LTimer = "M2L";
    static constexpr const char* L2LTimer = "L2L";
    static constexpr const char* L2PTimer = "L2P";
    static constexpr const char* P2PTimer = "P2P";
    static constexpr const char* M2PTimer = "M2P";
    static constexpr const char* P2LTimer = "P2L";
    static constexpr const char* NearTimer = "Near";
    enum {nbTimers = 9};

    /// Timers
    FTimerMap Timers;

    /// Constructor: resets all timers
    FAlgorithmTimers() = default;

    /// Default copy contructor
    FAlgorithmTimers(const FAlgorithmTimers&) = default;
    /// Default move contructor
    FAlgorithmTimers(FAlgorithmTimers&&) = default;

    /// Elapsed time between last FTic::tic() and FTic::tac() for given timer.
    double getTime(std::string TimerName) const{
        double res = 0;
        try {
            res = Timers.at(TimerName).elapsed();
        } catch(std::out_of_range) {
            res = 0;
        }
        return res;
    }

    /// Cumulated time between all FTic::tic() and FTic::tac() for given timer.
    double getCumulatedTime(std::string TimerName) const{
        //assert to verify size
        return Timers.at(TimerName).cumulated();
    }

};

#endif // FALGORITHMTIMERS_HPP
