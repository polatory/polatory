#ifndef FCUDACOMPLETECELL_HPP
#define FCUDACOMPLETECELL_HPP

template <class SymboleCellClass, class PoleCellClass, class LocalCellClass>
struct alignas(FStarPUDefaultAlign::StructAlign) FCudaCompositeCell {
    __device__ FCudaCompositeCell()
        : symb(nullptr), up(nullptr), down(nullptr){
    }

    SymboleCellClass* symb;
    PoleCellClass* up;
    LocalCellClass* down;
};

#endif // FCUDACOMPLETECELL_HPP

