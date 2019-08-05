// See LICENCE file at project root
#ifndef FPARAMETERS_H
#define FPARAMETERS_H

#include "FGlobal.hpp"

#include <sstream>
#include <iostream>
#include <cstring>

#include <vector>

#include "../Containers/FVector.hpp"

#include "FAssert.hpp"

/** This file proposes some methods
  * to work with user input parameters.
  */

namespace FParameters{
	/** If it is not found */
	const static int NotFound = -1;
	/**
	 * This function gives a parameter in a standart type
	 * @param str string of chars to be converted from
	 * @param defaultValue the value to be converted to
	 * @return argv[inArg] in the template VariableType form
	 * @warning VariableType need to work with istream >> operator
	 * <code> const int argInt = userParemetersAt<int>(1,-1); </code>
	 */
	template <class VariableType>
    inline const VariableType StrToOther(const char* const str, const VariableType& defaultValue = VariableType(), bool* hasWorked = nullptr){
		std::istringstream iss(str,std::istringstream::in);
        VariableType value = defaultValue;
		iss >> value;
        FAssertLF(iss.eof());
        if(hasWorked) (*hasWorked) = bool(iss.eof());
		if( /*iss.tellg()*/ iss.eof() ) return value;
		return defaultValue;
	}
	
    /** To put a char into lower format
      *
      */
    inline char toLower(const char c){
        return char('A' <= c && c <= 'Z' ? (c - 'A') + 'a' : c);
    }

    /** To know if two char are equals
      *
      */
    inline bool areCharsEquals(const char c1, const char c2, const bool caseSensible = false){
        return (caseSensible && c1 == c2) || (!caseSensible && toLower(c1) == toLower(c2));
    }

    /** To know if two str are equals
      *
      */
    inline bool areStrEquals(const char* const inStr1, const char* const inStr2, const bool caseSensible = false){
        int idxStr = 0;
        while(inStr1[idxStr] != '\0' && inStr2[idxStr] != '\0'){
            if(!areCharsEquals(inStr1[idxStr] ,inStr2[idxStr],caseSensible)){
                return false;
            }
            ++idxStr;
        }
        return inStr1[idxStr] == inStr2[idxStr];
    }

    /** To find a parameters from user format char parameters
      *
      */
    inline int findParameter(const int argc, const char* const * const argv, const char* const inName, const bool caseSensible = false){
        for(int idxArg = 0; idxArg < argc ; ++idxArg){
            if(areStrEquals(inName, argv[idxArg], caseSensible)){
                return idxArg;
            }
        }
        return NotFound;
    }

    /** To know if a parameter exist from user format char parameters
      *
      */
    inline bool existParameter(const int argc, const char* const * const argv, const char* const inName, const bool caseSensible = false){
        return NotFound != findParameter( argc, argv, inName, caseSensible);
    }

    /** To get a value like :
      * getValue(argc,argv, "Toto", 0, false);
      * will return 55 if the command contains : -Toto 55
      * else 0
      */
    template <class VariableType>
    inline const VariableType getValue(const int argc, const char* const * const argv, const char* const inName, const VariableType& defaultValue = VariableType(), const bool caseSensible = false){
        const int position = findParameter(argc,argv,inName,caseSensible);
        FAssertLF(position == NotFound || position != argc - 1);
        if(position == NotFound || position == argc - 1){
            return defaultValue;
        }
        return StrToOther(argv[position+1],defaultValue);
    }

    /** Get a str from argv
      */
    inline const char* getStr(const int argc, const char* const * const argv, const char* const inName, const char* const inDefault, const bool caseSensible = false){
        const int position = findParameter(argc,argv,inName,caseSensible);
        FAssertLF(position == NotFound || position != argc - 1);
        if(position == NotFound || position == argc - 1){
            return inDefault;
        }
        return argv[position+1];
    }


    /** To find a parameters from user format char parameters
      *
      */
    inline int findParameter(const int argc, const char* const * const argv, const std::vector<const char*>& inNames, const bool caseSensible = false){
        for(const char* name : inNames){
            const int res = findParameter(argc, argv, name, caseSensible);
            if(res != NotFound){
                return res;
            }
        }
        return NotFound;
    }

    /** To know if a parameter exist from user format char parameters
      *
      */
    inline bool existParameter(const int argc, const char* const * const argv, const std::vector<const char*>& inNames, const bool caseSensible = false){
        for(const char* name : inNames){
            if(existParameter(argc, argv, name, caseSensible)){
                return true;
            }
        }
        return false;
    }

    /** To get a value like :
      * getValue(argc,argv, "Toto", 0, false);
      * will return 55 if the command contains : -Toto 55
      * else 0
      */
    template <class VariableType>
    inline const VariableType getValue(const int argc, const char* const * const argv, const std::vector<const char*>& inNames, const VariableType& defaultValue = VariableType(), const bool caseSensible = false){
        for(const char* name : inNames){
            const int position = findParameter(argc, argv, name, caseSensible);
            FAssertLF(position == NotFound || position != argc - 1, "Could no find a value for argument: ",name, ". " );
            if(position != NotFound && position != argc - 1){
                return StrToOther(argv[position+1],defaultValue);
            }
        }
        return defaultValue;
    }

    /** Get a str from argv
      */
    inline const char* getStr(const int argc, const char* const * const argv, const std::vector<const char*>& inNames, const char* const inDefault, const bool caseSensible = false){
        for(const char* name : inNames){
            const int position = findParameter(argc, argv, name, caseSensible);
            FAssertLF(position == NotFound || position != argc - 1, "Could no find a value for argument: ",name, ". ");
            if(position != NotFound && position != argc - 1){
                return argv[position+1];
            }
        }
        return inDefault;
    }


    template <class ValueType>
    inline FVector<ValueType> getListOfValues(const int argc, const char* const * const argv, const std::vector<const char*>& inNames, const char separator = ';'){
        const char* valuesStr = getStr( argc, argv, inNames, nullptr);
        if(valuesStr == nullptr){
            return FVector<ValueType>();
        }

        FVector<char> word;
        FVector<ValueType> res;
        int idxCharStart = 0;
        int idxCharEnd = 0;
        while(valuesStr[idxCharEnd] != '\0'){
            if(valuesStr[idxCharEnd] == separator){
                const int lengthWord = idxCharEnd-idxCharStart;
                if(lengthWord){
                    word.clear();
                    word.memocopy(&valuesStr[idxCharStart], lengthWord);
                    word.push('\0');
                    bool hasWorked;
                    const ValueType val = StrToOther(word.data(), ValueType(), &hasWorked);
                    if(hasWorked){
                        res.push(val);
                    }
                }
                idxCharEnd  += 1;
                idxCharStart = idxCharEnd;
            }
            else{
                idxCharEnd  += 1;
            }
        }
        {
            const int lengthWord = idxCharEnd-idxCharStart;
            if(lengthWord){
                word.clear();
                word.memocopy(&valuesStr[idxCharStart], lengthWord);
                word.push('\0');
                bool hasWorked;
                const ValueType val = StrToOther(word.data(), ValueType(), &hasWorked);
                if(hasWorked){
                    res.push(val);
                }
            }
        }

        return res;
    }
}



#endif // FPARAMETERS_H
