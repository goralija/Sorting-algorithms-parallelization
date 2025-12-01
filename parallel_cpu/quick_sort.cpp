// parallel_cpu/quick_sort_avx_omp.cpp
// Parallel + AVX optimized version of quicksort
// Uses OpenMP tasks + AVX vectorization for partitioning
// Compile with -O3 -march=native -fopenmp and AVX2 support

#include <vector>
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include "main_template.hpp"

using namespace std;

// Tuning parameters
const int INSERTION_SORT_THRESHOLD = 32;
const int AVX_PARTITION_THRESHOLD = 512;
const int CACHE_LINE_SIZE = 64;
const int MAX_RECURSION_DEPTH_FACTOR = 2;

// Dynamic task threshold based on available cores
inline int get_task_threshold(int total_size) {
    int cores = omp_get_max_threads();
    return max(1024, total_size / (cores * 4));
}

// Cache-friendly block size (L1 cache optimized)
inline int get_cache_block_size() {
    return 8192; // ~32KB / 4 bytes per int
}

// Optimized insertion sort with sentinels
inline void insertion_sort(int* arr, int n) {
    if (n <= 1) return;
    
    // Find minimum and place at beginning as sentinel
    int min_idx = 0;
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[min_idx]) min_idx = i;
    }
    if (min_idx != 0) swap(arr[0], arr[min_idx]);
    
    // Unguarded insertion sort (no bounds checking)
    for (int i = 2; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Heap sort fallback for worst-case scenarios
inline void heap_sort(int* arr, int n) {
    std::make_heap(arr, arr + n);
    std::sort_heap(arr, arr + n);
}

// Median-of-three pivot selection with swapping
inline int select_pivot(int arr[], int low, int high) {
    int mid = low + ((high - low) >> 1);
    
    // Sort low, mid, high to get median
    if (arr[mid] < arr[low]) swap(arr[mid], arr[low]);
    if (arr[high] < arr[low]) swap(arr[high], arr[low]);
    if (arr[high] < arr[mid]) swap(arr[high], arr[mid]);
    
    return mid; // Return index of median
}

// AVX-accelerated forward scan
inline int avx_forward_scan(int arr[], int start, int end, int pivot) {
    __m256i pivot_vec = _mm256_set1_epi32(pivot);
    int i = start;
    
    // Process in blocks of 8
    while (i + 7 <= end) {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<__m256i*>(arr + i));
        __m256i cmp = _mm256_cmpgt_epi32(pivot_vec, data);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        
        // If any element >= pivot, find the exact position
        if (mask != 0xFF) {
            for (int k = 0; k < 8; k++) {
                if (arr[i + k] >= pivot) {
                    return i + k;
                }
            }
        }
        i += 8;
    }
    
    // Handle remaining elements
    while (i <= end && arr[i] < pivot) i++;
    return i;
}

// AVX-accelerated backward scan  
inline int avx_backward_scan(int arr[], int start, int end, int pivot) {
    __m256i pivot_vec = _mm256_set1_epi32(pivot);
    int j = end;
    
    // Process in blocks of 8
    while (j - 7 >= start) {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<__m256i*>(arr + j - 7));
        __m256i cmp = _mm256_cmpgt_epi32(data, pivot_vec);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        
        // If any element <= pivot, find the exact position
        if (mask != 0xFF) {
            for (int k = 7; k >= 0; k--) {
                if (arr[j - k] <= pivot) {
                    return j - k;
                }
            }
        }
        j -= 8;
    }
    
    // Handle remaining elements
    while (j >= start && arr[j] > pivot) j--;
    return j;
}

// Hybrid AVX-scalar partition
inline int hybrid_partition(int arr[], int low, int high) {
    // Select pivot using median-of-three
    int pivot_idx = select_pivot(arr, low, high);
    int pivot = arr[pivot_idx];
    
    // Move pivot to temporary position for Hoare partition
    swap(arr[pivot_idx], arr[low]);
    
    int i = low;
    int j = high;
    
    // Use AVX for large partitions, scalar for small ones
    if (high - low + 1 >= AVX_PARTITION_THRESHOLD) {
        i = avx_forward_scan(arr, low + 1, high, pivot);
        j = avx_backward_scan(arr, low + 1, high, pivot);
    } else {
        // Scalar scans
        i = low + 1;
        j = high;
        while (true) {
            while (i <= high && arr[i] < pivot) i++;
            while (j > low && arr[j] > pivot) j--;
            if (i >= j) break;
            swap(arr[i], arr[j]);
            i++;
            j--;
        }
    }
    
    // Restore pivot to correct position
    swap(arr[low], arr[j]);
    return j;
}

// Cache-aware sequential quicksort with tail recursion
void cache_aware_quick_sort(int arr[], int low, int high, int depth) {
    const int cache_block_size = get_cache_block_size();
    const int max_depth = MAX_RECURSION_DEPTH_FACTOR * (int)log2(high - low + 1);
    
    // Switch to heap sort if recursion depth too high
    if (depth > max_depth) {
        heap_sort(arr + low, high - low + 1);
        return;
    }
    
    while (high - low > INSERTION_SORT_THRESHOLD) {
        // For very large blocks, use cache-aware processing
        if (high - low > cache_block_size) {
            int pi = hybrid_partition(arr, low, high);
            
            // Sort smaller partition recursively, larger iteratively
            if (pi - low < high - pi) {
                cache_aware_quick_sort(arr, low, pi - 1, depth + 1);
                low = pi + 1;
            } else {
                cache_aware_quick_sort(arr, pi + 1, high, depth + 1);
                high = pi - 1;
            }
        } else {
            // Standard optimized quicksort for cache-sized blocks
            int pi = hybrid_partition(arr, low, high);
            
            // Always recurse on smaller partition first for better stack usage
            if (pi - low < high - pi) {
                cache_aware_quick_sort(arr, low, pi - 1, depth + 1);
                low = pi + 1;
            } else {
                cache_aware_quick_sort(arr, pi + 1, high, depth + 1);
                high = pi - 1;
            }
        }
    }
    
    // Final insertion sort for small segments
    if (high > low) {
        insertion_sort(arr + low, high - low + 1);
    }
}

// Check if two memory ranges might cause false sharing
inline bool might_have_false_sharing(int low1, int high1, int low2, int high2) {
    int block1_start = low1 / (CACHE_LINE_SIZE / sizeof(int));
    int block1_end = high1 / (CACHE_LINE_SIZE / sizeof(int));
    int block2_start = low2 / (CACHE_LINE_SIZE / sizeof(int));
    int block2_end = high2 / (CACHE_LINE_SIZE / sizeof(int));
    
    return (block1_end >= block2_start) && (block2_end >= block1_start);
}

// Main parallel quicksort with OpenMP tasks
void quick_sort_parallel(int arr[], int low, int high, int depth, int task_threshold) {
    int size = high - low + 1;
    
    // Base case: use sequential sort for small arrays
    if (size <= INSERTION_SORT_THRESHOLD) {
        insertion_sort(arr + low, size);
        return;
    }
    
    // Check recursion depth and switch to heap sort if too deep
    const int max_depth = MAX_RECURSION_DEPTH_FACTOR * (int)log2(size);
    if (depth > max_depth) {
        heap_sort(arr + low, size);
        return;
    }
    
    // Partition the array
    int pi = hybrid_partition(arr, low, high);
    
    // Calculate partition sizes
    int left_size = pi - low;
    int right_size = high - pi;
    
    // Determine if we should use parallel tasks
    bool use_parallel = (size >= task_threshold);
    
    if (use_parallel) {
        // Check for potential false sharing
        bool potential_false_sharing = might_have_false_sharing(low, pi, pi + 1, high);
        
        if (potential_false_sharing) {
            // If false sharing might occur, use untied tasks for better work stealing
            if (left_size < right_size) {
                #pragma omp task untied firstprivate(arr, low, pi, depth, task_threshold) \
                    if(left_size > task_threshold/2)
                quick_sort_parallel(arr, low, pi - 1, depth + 1, task_threshold);
                
                #pragma omp task untied firstprivate(arr, pi, high, depth, task_threshold) \
                    if(right_size > task_threshold/2)
                quick_sort_parallel(arr, pi + 1, high, depth + 1, task_threshold);
            } else {
                #pragma omp task untied firstprivate(arr, pi, high, depth, task_threshold) \
                    if(right_size > task_threshold/2)
                quick_sort_parallel(arr, pi + 1, high, depth + 1, task_threshold);
                
                #pragma omp task untied firstprivate(arr, low, pi, depth, task_threshold) \
                    if(left_size > task_threshold/2)
                quick_sort_parallel(arr, low, pi - 1, depth + 1, task_threshold);
            }
        } else {
            // No false sharing concern, use regular tasks
            #pragma omp task firstprivate(arr, low, pi, depth, task_threshold) \
                if(left_size > task_threshold/2)
            quick_sort_parallel(arr, low, pi - 1, depth + 1, task_threshold);
            
            #pragma omp task firstprivate(arr, pi, high, depth, task_threshold) \
                if(right_size > task_threshold/2)
            quick_sort_parallel(arr, pi + 1, high, depth + 1, task_threshold);
        }
        
        #pragma omp taskwait
        
    } else {
        // Sequential recursion with cache awareness
        cache_aware_quick_sort(arr, low, high, depth);
    }
}

// Wrapper function with proper OpenMP initialization
void quick_sort_parallel_wrapper(std::vector<int>& vec) {
    if (vec.empty()) return;
    
    int n = vec.size();
    int task_threshold = get_task_threshold(n);
    
    // Prefetch first cache line for better performance
    _mm_prefetch(reinterpret_cast<const char*>(vec.data()), _MM_HINT_T0);
    
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            quick_sort_parallel(vec.data(), 0, n - 1, 0, task_threshold);
        }
    }
}

int main(int argc, char* argv[]) {
    return run_sort("quick_sort", "parallel_opt", quick_sort_parallel_wrapper, argc, argv);
}