#pragma once

#ifdef __cpp_lib_simd
#include <experimental/simd>
namespace simd = std::experimental;
#else
// Fallback if std::experimental::simd is unavailable
#include <algorithm>
#endif

#include <vector>

// SIMD-friendly min/max swap (for example)
inline void simd_minmax_swap(int& a, int& b) {
#ifdef __cpp_lib_simd
    auto va = simd::native<int, simd::native_abi>::broadcast(a);
    auto vb = simd::native<int, simd::native_abi>::broadcast(b);
    va = simd::min(va, vb);
    vb = simd::max(va, vb);
    a = va[0];
    b = vb[0];
#else
    if (a > b) std::swap(a, b);
#endif
}

// Placeholder for vectorized copy (can expand later)
inline void simd_copy(const std::vector<int>& src, std::vector<int>& dst) {
#ifdef __cpp_lib_simd
    // Implement SIMD copy using simd::load/store
#else
    std::copy(src.begin(), src.end(), dst.begin());
#endif
}
