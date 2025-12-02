// Parallel Radix Sort in C++ using OpenMP
// LSD (Least Significant Digit) radix sort with parallel optimizations:
// - Base-256 (byte-level) for fewer passes
// - Skip passes where all elements have same byte (huge win for sorted/few_unique)
// - Early termination based on max value
// - Parallel histogram computation
// - Parallel scatter with thread-local offsets
// - SIMD memory operations

#include <vector>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#include "main_template.hpp"

using namespace std;

// Tuning parameters
const int RADIX_BITS = 8;
const int RADIX_SIZE = 1 << RADIX_BITS;      // 256 buckets
const int RADIX_MASK = RADIX_SIZE - 1;       // 0xFF
const int PARALLEL_THRESHOLD = 65536;

// SIMD-accelerated memory copy
inline void simd_memcpy(int* dst, const int* src, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), data);
    }
    for (; i < count; i++) {
        dst[i] = src[i];
    }
}

// Sequential radix sort with skip optimization for small arrays
void sequential_radix_sort(int* arr, int n) {
    if (n <= 1) return;
    
    // Find max value to determine necessary passes
    int max_val = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    if (max_val == 0) return;
    
    int num_passes = 0;
    int temp_max = max_val;
    while (temp_max > 0) {
        num_passes++;
        temp_max >>= RADIX_BITS;
    }
    
    int* temp = new int[n];
    int* src = arr;
    int* dst = temp;
    
    for (int pass = 0; pass < num_passes; pass++) {
        int shift = pass * RADIX_BITS;
        int count[RADIX_SIZE] = {0};
        
        // Count
        for (int i = 0; i < n; i++) {
            count[(src[i] >> shift) & RADIX_MASK]++;
        }
        
        // Check if skip possible (all same byte value)
        int non_zero = 0;
        for (int i = 0; i < RADIX_SIZE && non_zero < 2; i++) {
            if (count[i] > 0) non_zero++;
        }
        if (non_zero == 1) continue;  // Skip this pass
        
        // Prefix sum
        int total = 0;
        for (int i = 0; i < RADIX_SIZE; i++) {
            int old = count[i];
            count[i] = total;
            total += old;
        }
        
        // Scatter
        for (int i = 0; i < n; i++) {
            int byte_val = (src[i] >> shift) & RADIX_MASK;
            dst[count[byte_val]++] = src[i];
        }
        
        swap(src, dst);
    }
    
    if (src != arr) {
        memcpy(arr, src, n * sizeof(int));
    }
    
    delete[] temp;
}

// Main parallel radix sort with skip optimization
void radix_sort_parallel(int* arr, int n) {
    if (n <= 1) return;
    
    if (n < PARALLEL_THRESHOLD) {
        sequential_radix_sort(arr, n);
        return;
    }
    
    int num_threads = omp_get_max_threads();
    
    // Parallel max finding
    int max_val = 0;
    #pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    
    if (max_val == 0) return;
    
    // Determine number of passes needed
    int num_passes = 0;
    int temp_max = max_val;
    while (temp_max > 0) {
        num_passes++;
        temp_max >>= RADIX_BITS;
    }
    
    int* temp = new int[n];
    int* src = arr;
    int* dst = temp;
    
    // Allocate thread-local histogram storage once
    int** local_counts = new int*[num_threads];
    int** thread_offsets = new int*[num_threads];
    int* thread_starts = new int[num_threads + 1];
    
    for (int t = 0; t < num_threads; t++) {
        local_counts[t] = new int[RADIX_SIZE];
        thread_offsets[t] = new int[RADIX_SIZE];
    }
    
    // Precompute thread chunk boundaries (must be consistent!)
    int chunk_size = (n + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        thread_starts[t] = t * chunk_size;
        if (thread_starts[t] > n) thread_starts[t] = n;
    }
    thread_starts[num_threads] = n;
    
    for (int pass = 0; pass < num_passes; pass++) {
        int shift = pass * RADIX_BITS;
        int global_count[RADIX_SIZE] = {0};
        
        // Reset local counts
        for (int t = 0; t < num_threads; t++) {
            memset(local_counts[t], 0, RADIX_SIZE * sizeof(int));
        }
        
        // Parallel histogram - each thread counts its own chunk
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int* my_count = local_counts[tid];
            int start = thread_starts[tid];
            int end = thread_starts[tid + 1];
            
            for (int i = start; i < end; i++) {
                my_count[(src[i] >> shift) & RADIX_MASK]++;
            }
        }
        
        // Merge histograms and check for skip
        int non_zero_buckets = 0;
        for (int i = 0; i < RADIX_SIZE; i++) {
            for (int t = 0; t < num_threads; t++) {
                global_count[i] += local_counts[t][i];
            }
            if (global_count[i] > 0) non_zero_buckets++;
        }
        
        // Skip pass if all elements have same byte value
        if (non_zero_buckets == 1) {
            continue;
        }
        
        // Prefix sum for global offsets
        int offsets[RADIX_SIZE];
        int total = 0;
        for (int i = 0; i < RADIX_SIZE; i++) {
            offsets[i] = total;
            total += global_count[i];
        }
        
        // Compute per-thread offsets based on their local counts
        // Thread t's offset for bucket i = global_offset[i] + sum of local_counts[0..t-1][i]
        for (int i = 0; i < RADIX_SIZE; i++) {
            int running = offsets[i];
            for (int t = 0; t < num_threads; t++) {
                thread_offsets[t][i] = running;
                running += local_counts[t][i];
            }
        }
        
        // Parallel scatter - each thread scatters its own chunk
        // Each thread needs its own copy of offsets since we increment them
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int start = thread_starts[tid];
            int end = thread_starts[tid + 1];
            
            // Make a local copy of this thread's offsets
            int my_offsets[RADIX_SIZE];
            memcpy(my_offsets, thread_offsets[tid], RADIX_SIZE * sizeof(int));
            
            for (int i = start; i < end; i++) {
                int byte_val = (src[i] >> shift) & RADIX_MASK;
                dst[my_offsets[byte_val]++] = src[i];
            }
        }
        
        swap(src, dst);
    }
    
    // Cleanup thread storage
    for (int t = 0; t < num_threads; t++) {
        delete[] local_counts[t];
        delete[] thread_offsets[t];
    }
    delete[] local_counts;
    delete[] thread_offsets;
    delete[] thread_starts;
    
    // Copy result back if needed
    if (src != arr) {
        simd_memcpy(arr, src, n);
    }
    
    delete[] temp;
}

// Wrapper for template compatibility
void radix_sort_parallel_wrapper(std::vector<int>& vec) {
    if (vec.empty()) return;
    radix_sort_parallel(vec.data(), static_cast<int>(vec.size()));
}

// Entry point (uses common benchmarking template)
int main(int argc, char* argv[]) {
    return run_sort("radix_sort", "parallel_cpu", radix_sort_parallel_wrapper, argc, argv);
}
