// @SCALFMM_PRIVATE
#ifndef FOFFETOF_HPP
#define FOFFETOF_HPP

//#define FOffsetOf(Type,Member) reinterpret_cast<std::size_t>(&((reinterpret_cast<Type*>(0xF00))->Member)) - std::size_t(0xF00)
#define FOffsetOf(Type,Member) offsetof(Type,Member)

#endif // FOFFETOF_HPP

