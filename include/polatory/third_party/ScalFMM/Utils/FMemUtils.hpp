// ===================================================================================
// Copyright ScalFmm 2016 INRIA, Olivier Coulaud, Bérenger Bramas,
// Matthias Messner olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the
// FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.
// An extension to the license is given to allow static linking of scalfmm
// inside a proprietary application (no matter its license).
// See the main license file for more details.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info".
// "http://www.gnu.org/licenses".
// ===================================================================================
#ifndef FMEMUTILS_HPP
#define FMEMUTILS_HPP

#include "FGlobal.hpp"

// To get memcpy
#include <cstring>
#include <climits>


/** The memory utils class proposes some methods
  * to copy/set memory with an size bigger than size_t
  * @todo manage the file to remove it if needed
  */
namespace FMemUtils {
    static const FSize MaxSize_t = UINT_MAX; //std::numeric_limits<std::size_t>::max();

    /** memcpy */
    inline void* memcpy(void* const dest, const void* const source, const FSize nbBytes){
        if( nbBytes < MaxSize_t){
            return ::memcpy(dest, source, size_t(nbBytes));
        }
        else{
            char* iterDest          = static_cast<char*>(dest);
            const char* iterSource  = static_cast<const char*>(source);

            for(FSize idx = 0 ; idx < nbBytes - MaxSize_t ; idx += MaxSize_t ){
                ::memcpy(iterDest, iterSource, size_t(MaxSize_t));
                iterDest += MaxSize_t;
                iterSource += MaxSize_t;
            }
            ::memcpy(iterDest, iterSource, size_t(nbBytes%MaxSize_t));

            return dest;
        }
    }

    /** memset */
    inline void* memset(void* const dest, const int val, const FSize nbBytes){
        if( nbBytes < MaxSize_t){
            return ::memset(dest, val, size_t(nbBytes));
        }
        else{
            char* iterDest  = static_cast<char*>(dest);

            for(FSize idx = 0 ; idx < nbBytes - MaxSize_t ; idx += MaxSize_t ){
                ::memset(iterDest, val, size_t(MaxSize_t));
                iterDest += MaxSize_t;
            }
            ::memset(iterDest, val, size_t(nbBytes%MaxSize_t));

            return dest;
        }
    }

    /** copy all value from one vector to the other */
    template <class TypeClass>
    inline void copyall(TypeClass* dest, const TypeClass* source, int nbElements){
        for(; 0 < nbElements ; --nbElements){
            (*dest++) = (*source++);
        }
    }

    /** copy all value from one vector to the other */
    template <class TypeClass>
    inline void addall(TypeClass* dest, const TypeClass* source, int nbElements){
        for(; 0 < nbElements ; --nbElements){
            (*dest++) += (*source++);
        }
    }

    /** copy all value from one vector to the other */
    template <class TypeClass>
    inline void setall(TypeClass* dest, const TypeClass& source, int nbElements){
        for(; 0 < nbElements ; --nbElements){
            (*dest++) = source;
        }
    }
	
	  /** swap values from a and b*/
	  template <class TypeClass>
        inline void swap(TypeClass& a, TypeClass& b){
			TypeClass c(a);
			a=b;
			b=c;
		}

    /** Delete all */
    template <class TypeClass>
    inline void DeleteAllArray(TypeClass*const array[], const int size){
        for(int idx = 0 ; idx < size ; ++idx){
            delete[] array[idx];
        }
    }

    /** Delete all */
    template <class TypeClass>
    inline void DeleteAll(TypeClass*const array[], const int size){
        for(int idx = 0 ; idx < size ; ++idx){
            delete array[idx];
        }
    }
}

#endif // FMEMUTILS_HPP
