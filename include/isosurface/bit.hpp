// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

namespace polatory {
namespace isosurface {
namespace bit {

namespace {

int naive_popcnt32(int x)
{
   int count = 0;
   while (x) {
      x &= x - 1;
      count++;
   }
   return count;
}

int naive_bit_scan_forward(int x)
{
   int count = 0;
   while (!(x & 0x1)) {
      x >>= 1;
      count++;
   }
   return count;
}

int count(unsigned int bit_set)
{
#if defined(_MSC_VER)
   return naive_popcnt32(bit_set);
#elif defined(__INTEL_COMPILER)
   return _popcnt32(bit_set);
#else
   return naive_popcnt32(bit_set);
#endif
}

int peek(unsigned int bit_set)
{
   if (bit_set == 0) return -1;

#if defined(_MSC_VER)
   return naive_bit_scan_forward(bit_set);
#elif defined(__INTEL_COMPILER)
   return _bit_scan_forward(bit_set);
#else
   return naive_bit_scan_forward(bit_set);
#endif
}

int pop(unsigned int& bit_set)
{
   if (bit_set == 0) return -1;

   int bit_idx = peek(bit_set);
   unsigned int bit = 1 << bit_idx;
   bit_set ^= bit;

   return bit_idx;
}

int pop(unsigned short& bit_set)
{
   if (bit_set == 0) return -1;

   int bit_idx = peek(bit_set);
   unsigned short bit = 1 << bit_idx;
   bit_set ^= bit;

   return bit_idx;
}

} // namespace

} // namespace bit
} // namespace isosurface
} // namespace polatory
