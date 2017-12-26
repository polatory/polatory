// See LICENCE file at project root
#ifndef FASSERT_HPP
#define FASSERT_HPP

#include <iostream>
#include <cassert>

#include "FGlobal.hpp"


/**
 * @brief The FError class
 * It is recommendede to use the macro:
 * FAssertLF( aTest , "some data ", "to ", plot);
 * Assertion are enabled or disabled during the compilation.
 * If disabled, the test instruction is still used (but the return will be optimized out by
 * the compiler).
 */
class FError {
protected:
    /**
     * @brief ErrPrint private method to end print
     */
    static void ErrPrint(){
        std::cerr << '\n';
    }

    /**
     * @brief ErrPrint private methdo to print
     */
    template<typename T, typename... Args>
    static void ErrPrint(const T& toPrint, Args... args){
        std::cerr << toPrint;
        ErrPrint( args... );
    }


public:
    //////////////////////////////////////////////////////////////
    // Should not be called directly
    //////////////////////////////////////////////////////////////

    /** Nothing to print */
    static void Print(){
    }

    /** One or more things to print */
    template<typename T, typename... Args>
    static void Print(const T& toPrint, Args... args){
        std::cerr << "[ERROR] ";
        ErrPrint( toPrint, args... );
    }
};

#ifdef SCALFMM_USE_ASSERT

//////////////////////////////////////////////////////////////
// Sp error activated
//////////////////////////////////////////////////////////////

#define FErrorAssertExit(TEST, ...) \
    if( !(TEST) ){ \
        FError::Print( __VA_ARGS__ ); \
	std::exit(EXIT_FAILURE) ; /*throw std::exception();*/	\
    }


#else

//////////////////////////////////////////////////////////////
// Sp error desactivated
//////////////////////////////////////////////////////////////

#define FErrorAssertExit(TEST, ...) \
    if( !(TEST) ){}


#endif

//////////////////////////////////////////////////////////////
// Shortcut macro
//////////////////////////////////////////////////////////////

#define SPARSETD_ERROR_LINE " At line " , __LINE__ , "."
#define SPARSETD_ERROR_FILE " In file " , __FILE__ , "."

#define FAssert FErrorAssertExit

#define FAssertLF(...) FAssert(__VA_ARGS__, SPARSETD_ERROR_LINE, SPARSETD_ERROR_FILE)

#endif //FASSERT_HPP


