// Optimized Radix Sort in C++
// LSD (Least Significant Digit) radix sort with optimizations:
// - Base-256 (byte-level) instead of base-10 for fewer passes
// - Skip passes where all elements have the same byte value
// - Early termination when remaining bytes are all zero
// - Preallocated buffers to avoid repeated allocations
// - Cache-friendly memory access patterns with prefetching

#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include "main_template.hpp"

using namespace std;

const int RADIX_BITS = 8;                    // Process 8 bits at a time (base-256)
const int RADIX_SIZE = 1 << RADIX_BITS;      // 256 buckets
const int RADIX_MASK = RADIX_SIZE - 1;       // 0xFF

// Optimized radix sort with skip optimization
// Skips passes where all elements have the same byte value (common in sorted/few_unique)
void radix_sort_opt(int arr[], int n) {
    if (n <= 1) return;

    // Find max to determine how many passes we actually need
    int max_val = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    
    // If max is 0, array is all zeros - already sorted
    if (max_val == 0) return;
    
    // Determine number of passes needed based on max value
    int num_passes = 0;
    int temp_max = max_val;
    while (temp_max > 0) {
        num_passes++;
        temp_max >>= RADIX_BITS;
    }
    
    int* temp = new int[n];
    int* src = arr;
    int* dst = temp;
    
    int passes_done = 0;
    
    // Process only the necessary bytes
    for (int pass = 0; pass < num_passes; pass++) {
        int shift = pass * RADIX_BITS;
        
        // Histogram for counting occurrences
        int count[RADIX_SIZE] = {0};
        
        // Count occurrences with prefetching
        for (int i = 0; i < n; i++) {
            // Prefetch ahead for better cache utilization
            if (i + 16 < n) {
                __builtin_prefetch(&src[i + 16], 0, 0);
            }
            int byte_val = (src[i] >> shift) & RADIX_MASK;
            count[byte_val]++;
        }
        
        // Check if this pass can be skipped (all elements have same byte value)
        int non_zero_buckets = 0;
        int last_non_zero = -1;
        for (int i = 0; i < RADIX_SIZE; i++) {
            if (count[i] > 0) {
                non_zero_buckets++;
                last_non_zero = i;
            }
        }
        
        // If all elements fall into one bucket, skip this pass
        if (non_zero_buckets == 1) {
            continue;  // No need to scatter, array order unchanged for this byte
        }
        
        // Prefix sum
        int total = 0;
        for (int i = 0; i < RADIX_SIZE; i++) {
            int old_count = count[i];
            count[i] = total;
            total += old_count;
        }
        
        // Scatter with prefetching
        for (int i = 0; i < n; i++) {
            if (i + 16 < n) {
                __builtin_prefetch(&src[i + 16], 0, 0);
            }
            int byte_val = (src[i] >> shift) & RADIX_MASK;
            dst[count[byte_val]++] = src[i];
        }
        
        // Swap src and dst for next pass
        std::swap(src, dst);
        passes_done++;
    }
    
    // If odd number of actual passes done, result is in temp
    // Copy back to arr if needed
    if (src != arr) {
        memcpy(arr, src, n * sizeof(int));
    }
    
    delete[] temp;
}

// Wrapper for template compatibility
void radix_sort_wrapper(std::vector<int>& vec) {
    if (vec.empty()) return;
    radix_sort_opt(vec.data(), static_cast<int>(vec.size()));
}

// Entry point (uses common benchmarking template)
int main(int argc, char* argv[]) {
    return run_sort("radix_sort", "sequential_optimized", radix_sort_wrapper, argc, argv);
}
