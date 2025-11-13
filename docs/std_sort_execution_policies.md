# std::sort Execution Policies

## Overview

To ensure fair comparison between sequential and parallel implementations, we explicitly control the execution policy of `std::sort`.

## Implementation

### Sequential Versions (naive & optimized)

```cpp
#include <execution>

// Explicitly sequential - prevents any parallelization
std::sort(std::execution::seq, vec.begin(), vec.end());
```

**Guarantees:**
- ✅ No parallelization (even with OpenMP enabled)
- ✅ No multi-threading
- ✅ Pure sequential execution
- ✅ Fair baseline for comparison

### Parallel Version (parallel_cpu)

```cpp
#include <execution>

// Explicitly parallel - enables multi-threading
std::sort(std::execution::par, vec.begin(), vec.end());
```

**Guarantees:**
- ✅ Uses parallel execution when available
- ✅ Multi-threaded sorting
- ✅ Leverages OpenMP/TBB backend

## CMakeLists.txt Configuration

Sequential executables explicitly disable OpenMP to prevent compiler optimizations that might introduce parallelism:

```cmake
# Sequential builds - no OpenMP
target_compile_options(sequential_naive_${exec_name} PRIVATE -fno-openmp)
target_compile_options(sequential_optimized_${exec_name} PRIVATE -fno-openmp)

# Parallel builds - with OpenMP
target_link_libraries(parallel_cpu_${exec_name} OpenMP::OpenMP_CXX)
```

## Execution Policies (C++17)

| Policy | Description | Thread Safety |
|--------|-------------|---------------|
| `std::execution::seq` | Sequential execution | Single-threaded |
| `std::execution::par` | Parallel execution | Multi-threaded |
| `std::execution::par_unseq` | Parallel + vectorized | Multi-threaded + SIMD |

## Why This Matters

Without explicit execution policies:
- ❌ Compiler might auto-parallelize `std::sort` with aggressive optimization flags
- ❌ OpenMP linkage could enable parallel execution by default
- ❌ Unfair comparison (sequential version might secretly use parallelism)
- ❌ Inconsistent benchmarks across different compilers

With explicit policies:
- ✅ Guaranteed sequential execution for baseline
- ✅ Guaranteed parallel execution for comparison
- ✅ Fair and reproducible benchmarks
- ✅ Clear performance attribution

## References

- [C++17 Parallel Algorithms](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t)
- [std::sort](https://en.cppreference.com/w/cpp/algorithm/sort)
