// Parallel Radix Sort in C++ using OpenMP
// LSD (Least Significant Digit) radix sort with parallel optimizations:
// - Base-256 (byte-level) for fewer passes
// - Parallel histogram computation
// - Parallel prefix sum
// - Parallel scatter with thread-local buffers
// - Cache-friendly memory access patterns
// - AVX2 optimizations for memory operations

#include <vector>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#include "main_template.hpp"

using namespace std;

// Tuning parameters
const int RADIX_BITS = 8;                    // Process 8 bits at a time (base-256)
const int RADIX_SIZE = 1 << RADIX_BITS;      // 256 buckets
const int RADIX_MASK = RADIX_SIZE - 1;       // 0xFF
const int PARALLEL_THRESHOLD = 65536;        // Minimum size for parallelization
const int CACHE_LINE_INTS = 16;              // 64 bytes / 4 bytes per int

// SIMD-accelerated memory copy
inline void simd_memcpy(int* dst, const int* src, size_t count) {
    size_t i = 0;
    
    // AVX2: Copy 8 integers at a time
    for (; i + 8 <= count; i += 8) {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), data);
    }
    
    // Handle remaining elements
    for (; i < count; i++) {
        dst[i] = src[i];
    }
}

// Parallel histogram computation
void parallel_histogram(const int* arr, int n, int shift, int* global_count, int num_threads) {
    // Thread-local histograms to avoid false sharing
    int** local_counts = new int*[num_threads];
    for (int t = 0; t < num_threads; t++) {
        local_counts[t] = new int[RADIX_SIZE]();
    }
    
    // Parallel counting
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int* my_count = local_counts[tid];
        
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            int byte_val = (arr[i] >> shift) & RADIX_MASK;
            my_count[byte_val]++;
        }
    }
    
    // Merge local histograms into global histogram
    memset(global_count, 0, RADIX_SIZE * sizeof(int));
    for (int t = 0; t < num_threads; t++) {
        for (int i = 0; i < RADIX_SIZE; i++) {
            global_count[i] += local_counts[t][i];
        }
        delete[] local_counts[t];
    }
    delete[] local_counts;
}

// Parallel prefix sum (exclusive scan)
void parallel_prefix_sum(int* count, int* offsets) {
    int total = 0;
    for (int i = 0; i < RADIX_SIZE; i++) {
        offsets[i] = total;
        total += count[i];
    }
}

// Sequential radix sort for small arrays
void sequential_radix_sort(int* arr, int n) {
    if (n <= 1) return;
    
    int* temp = new int[n];
    int* src = arr;
    int* dst = temp;
    
    for (int shift = 0; shift < 32; shift += RADIX_BITS) {
        int count[RADIX_SIZE] = {0};
        
        // Count
        for (int i = 0; i < n; i++) {
            count[(src[i] >> shift) & RADIX_MASK]++;
        }
        
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

// Main parallel radix sort
void radix_sort_parallel(int* arr, int n) {
    if (n <= 1) return;
    
    // Use sequential version for small arrays
    if (n < PARALLEL_THRESHOLD) {
        sequential_radix_sort(arr, n);
        return;
    }
    
    int num_threads = omp_get_max_threads();
    
    // Allocate buffers
    int* temp = new int[n];
    int* src = arr;
    int* dst = temp;
    
    // Process each byte (4 passes for 32-bit integers)
    for (int shift = 0; shift < 32; shift += RADIX_BITS) {
        int count[RADIX_SIZE] = {0};
        int offsets[RADIX_SIZE];
        
        // Step 1: Parallel histogram computation
        parallel_histogram(src, n, shift, count, num_threads);
        
        // Step 2: Compute prefix sums (offsets)
        parallel_prefix_sum(count, offsets);
        
        // Step 3: Compute per-thread offsets for parallel scatter
        // Each thread needs to know where to write its elements
        int** thread_offsets = new int*[num_threads];
        int** thread_counts = new int*[num_threads];
        
        for (int t = 0; t < num_threads; t++) {
            thread_offsets[t] = new int[RADIX_SIZE];
            thread_counts[t] = new int[RADIX_SIZE]();
        }
        
        // First, count per-thread contributions
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int* my_count = thread_counts[tid];
            
            int chunk_size = (n + num_threads - 1) / num_threads;
            int start = tid * chunk_size;
            int end = min(start + chunk_size, n);
            
            for (int i = start; i < end; i++) {
                int byte_val = (src[i] >> shift) & RADIX_MASK;
                my_count[byte_val]++;
            }
        }
        
        // Compute thread-specific offsets
        for (int i = 0; i < RADIX_SIZE; i++) {
            int running = offsets[i];
            for (int t = 0; t < num_threads; t++) {
                thread_offsets[t][i] = running;
                running += thread_counts[t][i];
            }
        }
        
        // Step 4: Parallel scatter
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int* my_offsets = thread_offsets[tid];
            
            int chunk_size = (n + num_threads - 1) / num_threads;
            int start = tid * chunk_size;
            int end = min(start + chunk_size, n);
            
            for (int i = start; i < end; i++) {
                int byte_val = (src[i] >> shift) & RADIX_MASK;
                dst[my_offsets[byte_val]++] = src[i];
            }
        }
        
        // Cleanup
        for (int t = 0; t < num_threads; t++) {
            delete[] thread_offsets[t];
            delete[] thread_counts[t];
        }
        delete[] thread_offsets;
        delete[] thread_counts;
        
        // Swap buffers for next pass
        swap(src, dst);
    }
    
    // After 4 passes, copy result back if needed
    if (src != arr) {
        simd_memcpy(arr, src, n);
    }
    
    delete[] temp;
}

// Wrapper for template compatibility
void radix_sort_parallel_wrapper(std::vector<int>& vec) {
    if (vec.empty()) return;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            radix_sort_parallel(vec.data(), static_cast<int>(vec.size()));
        }
    }
}

// Entry point (uses common benchmarking template)
int main(int argc, char* argv[]) {
    return run_sort("radix_sort", "parallel_cpu", radix_sort_parallel_wrapper, argc, argv);
}
