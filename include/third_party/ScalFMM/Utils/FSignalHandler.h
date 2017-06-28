#ifndef FSIGNALHANDLER_H
#define FSIGNALHANDLER_H

#include "FGlobal.hpp"
#include "FLog.hpp"

#include <functional>
#include <vector>

/** We protect almost every thing if SCALFMM_USE_SIGNALS is undef
  * because is failled to compile with Clang.
  */

// Function in FSignalHandler.cpp
bool f_install_signal_handler();

class FSignalHandler
{
protected:
    bool installed;

    /** All callback to inform */
    std::vector< std::function<void(int)> > callbacks;

    /** Private constructor for singleton */
    FSignalHandler() : installed(false){
        installed = f_install_signal_handler();
    }

    /** A signal has been sent */
    void intercept(const int signalReceived){
#ifdef SCALFMM_USE_SIGNALS
        for(unsigned idx = 0 ; idx < callbacks.size() ; ++idx){
            callbacks[idx](signalReceived);
        }
#endif
    }

    friend void f_sig_handler(int signalReceived);

public:
    /** Singleton */
    static FSignalHandler controler;

    ~FSignalHandler(){
    }

    /** Add a callback to the listeners list */
    void registerCallback(std::function<void(int)> aCallback){
#ifdef SCALFMM_USE_SIGNALS
        callbacks.push_back(aCallback);
#endif
    }

    /** Remove some listeners */
    void unregisterCallback(std::function<void(int)> aCallback){
#ifdef SCALFMM_USE_SIGNALS
        using fptr = void(*)(int);

        unsigned idx = 0 ;
        for(; idx < callbacks.size() && !(callbacks[idx].target<fptr>() == aCallback.target<fptr>()); ++idx){
        }

        unsigned idxCopy = idx;
        idx += 1;

        for(; idx < callbacks.size() ; ++idx){
            if(!(callbacks[idx].target<fptr>() == aCallback.target<fptr>())){
                callbacks[idxCopy++] = callbacks[idx];
            }
        }

        callbacks.resize(callbacks.size() - idxCopy);
#endif
    }

    bool isInstalled(){
        return installed;
    }
};

#endif // FSIGNALHANDLER_H
