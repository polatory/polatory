/**
 * @file
 * This file contains the signal handler core functions.
 * It installs handlers and catch signals.
 * If backtrace is set to on during compilation then
 * the trace is print when a signal is caught.
 */

#include <ScalFMM/Utils/FGlobal.hpp>
#include <ScalFMM/Utils/FSignalHandler.h>


//< Singleton/Controler of the signal system.
FSignalHandler FSignalHandler::controler;

#ifdef SCALFMM_USE_SIGNALS

#include <ScalFMM/Utils/FLog.hpp>
#include <iostream>
#include <execinfo.h>
#include <cstdio>
#include <unistd.h>
#include <csignal>
#include <unistd.h>
#include <iostream>


/**
 * @brief f_sig_handler catch the system signals.
 * @param signalReceived
 */
void f_sig_handler(int signalReceived){
    std::cerr << "[SCALFMM] Signal " << signalReceived << " has been intercepted." << std::endl;
    FSignalHandler::controler.intercept(signalReceived);
    const int maxStackCalls = 40;
    void *stackArray[maxStackCalls];
    const int callsToPrint = backtrace(stackArray, maxStackCalls);
    backtrace_symbols_fd(stackArray, callsToPrint, STDERR_FILENO);
    exit(signalReceived);
}

/**
 * @brief f_install_signal_handler
 * This install several handler for various signals.
 */
bool f_install_signal_handler(){
    // The SIGINT signal is sent to a process by its controlling terminal when a user wishes to interrupt the process
    if( signal(SIGINT, f_sig_handler) == SIG_ERR ){
        FLOG( std::cout << "Signal Handler: Cannot install signal SIGINT\n"; )
    }
    // The SIGTERM signal is sent to a process to request its termination
    if( signal(SIGTERM, f_sig_handler) == SIG_ERR ){
        FLOG( std::cout << "Signal Handler: Cannot install signal SIGTERM\n"; )
    }
    // The SIGFPE signal is sent to a process when it executes an erroneous arithmetic operation, such as division by zero
    if( signal(SIGFPE, f_sig_handler) == SIG_ERR ){
        FLOG( std::cout << "Signal Handler: Cannot install signal SIGFPE\n"; )
    }
    // The SIGABRT signal is sent to a process to tell it to abort, i.e. to terminate
    if( signal(SIGABRT, f_sig_handler) == SIG_ERR ){
        FLOG( std::cout << "Signal Handler: Cannot install signal SIGABRT\n"; )
    }
    // The SIGSEGV signal is sent to a process when it makes an invalid virtual memory reference, or segmentation fault
    if( signal(SIGSEGV, f_sig_handler) == SIG_ERR ){
        FLOG( std::cout << "Signal Handler: Cannot install signal SIGSEGV\n"; )
    }
    return true;
}

#else

void f_sig_handler(int /*signalReceived*/){
}
bool f_install_signal_handler(){
    return false;
}

#endif
