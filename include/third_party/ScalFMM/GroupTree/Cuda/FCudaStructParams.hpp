#ifndef FCUDASTRUCTPARAMS_HPP
#define FCUDASTRUCTPARAMS_HPP

#include "../StarPUUtils/FStarPUDefaultAlign.hpp"
#include "FCudaGlobal.hpp"

template <class ArrayType, const int Size>
struct alignas(FStarPUDefaultAlign::StructAlign) FCudaParams{
    ArrayType values[Size];
};


#endif // FCUDASTRUCTPARAMS_HPP

