#ifndef FENV_HPP
#define FENV_HPP

#include "FGlobal.hpp"

#include <cstdlib>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstring>
#include <array>


/**
 * @brief The FEnv class manages the access and cast to environement variables.
 * It can convert values to native type or test if the values exist in a given array.
 *
 * @code // in this example we ask if the user as set a env variable
 * @code const int choice = FEnv::GetStrInArray("SCALFMM_STUFF", std::array<const char*, 3>{{ "0", "TRUE", "FALSE"}}, 3, 0);
 * @code // choice contain the found value or 0 if the variable does not exist or no value match in the array
 */
class FEnv {
    /**
     * To convert a char array to any native type using Stl.
     */
    template <class VariableType>
    static const VariableType StrToOther(const char* const str, const VariableType& defaultValue = VariableType()){
        std::istringstream iss(str,std::istringstream::in);
        VariableType value;
        iss >> value;
        if( /*iss.tellg()*/ iss.eof() ) return value;
        return defaultValue;
    }

public:
    /**
     * @brief VariableIsDefine
     * @param inVarName
     * @return true if the variable is defined in the environment
     */
    static bool VariableIsDefine(const char inVarName[]){
        return getenv(inVarName) != 0;
    }

    /**
     * This function return a value from the environment.
     * The value is of type VariableType.
     * If the variable does not exist of the cast has failed then defaultValue is returned.
     */
    template <class VariableType>
    static const VariableType GetValue(const char inVarName[], const VariableType defaultValue = VariableType()){
        const char*const value = getenv(inVarName);
        if(!value){
            return defaultValue;
        }
        return StrToOther(value,defaultValue);
    }

    /**
     * This function return a value from the environment.
     * The value is of type VariableType.
     * If the variable does not exist of the cast has failed then defaultValue is returned.
     */
    static bool GetBool(const char inVarName[], const bool defaultValue = false){
        const char*const value = getenv(inVarName);
        if(!value){
            return defaultValue;
        }
        return (strcmp(value,"TRUE") == 0) || (strcmp(value,"true") == 0) || (strcmp(value,"1") == 0);
    }

    /**
     * This function return a str from the environment.
     * If the variable does not exist then defaultValue is returned.
     */
    static const char* GetStr(const char inVarName[], const char* const defaultValue = 0){
        const char*const value = getenv(inVarName);
        if(!value){
            return defaultValue;
        }
        return value;
    }

    /**
     * This function return the found value that match the variable's value.
     */
    template <class VariableType, class ArrayType>
    static int GetValueInArray(const char inVarName[], const ArrayType& possibleValues, const int nbPossibleValues, const int defaultIndex = -1){
        const char*const value = getenv(inVarName);
        if(value){
            for(int idxPossible = 0 ; idxPossible < nbPossibleValues ; ++idxPossible){
                if( StrToOther(value,VariableType()) == possibleValues[idxPossible] ){
                    return idxPossible;
                }
            }
        }
        return defaultIndex;
    }

    /**
     * This function return the found value that match the variable's value.
     */
    template <class ArrayType>
    static int GetStrInArray(const char inVarName[], const ArrayType& possibleValues, const int nbPossibleValues, const int defaultIndex = -1){
        const char*const value = getenv(inVarName);
        if(value){
            for(int idxPossible = 0 ; idxPossible < nbPossibleValues ; ++idxPossible){
                if( strcmp(value,possibleValues[idxPossible]) == 0 ){
                    return idxPossible;
                }
            }
        }
        return defaultIndex;
    }
};

#endif // FENV_HPP

