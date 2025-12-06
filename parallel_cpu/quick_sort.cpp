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
const int MAX_RECURSION_DEPTH_FACTOR = 3;

// Dynamic task threshold based on available cores
inline int get_task_threshold(int total_size) {
    int cores = omp_get_max_threads();
    // Much higher threshold to ensure parallelism only when beneficial
    return 65536;
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

// Fast Hoare partition (same as sequential optimized)
inline int partition_hoare(int arr[], int low, int high) {
    // Inline median-of-three
    int mid = low + ((high - low) >> 1);
    if (arr[mid] < arr[low]) swap(arr[mid], arr[low]);
    if (arr[high] < arr[low]) swap(arr[high], arr[low]);
    if (arr[high] < arr[mid]) swap(arr[high], arr[mid]);
    
    int pivot = arr[mid];
    int i = low - 1;
    int j = high + 1;
    
    while (true) {
        do { i++; } while (arr[i] < pivot);
        do { j--; } while (arr[j] > pivot);
        
        if (i >= j) return j;
        
        swap(arr[i], arr[j]);
    }
}

// 3-way partitioning (Dutch National Flag) for handling duplicates efficiently
// Returns a pair: [lt, gt] where arr[low..lt-1] < pivot, arr[lt..gt] == pivot, arr[gt+1..high] > pivot
inline void partition_3way(int arr[], int low, int high, int& lt, int& gt) {
    if (low >= high) {
        lt = low;
        gt = high;
        return;
    }
    
    // Select pivot using median-of-three
    int pivot_idx = select_pivot(arr, low, high);
    int pivot = arr[pivot_idx];
    
    // 3-way partitioning
    int i = low;
    lt = low;
    gt = high;
    
    while (i <= gt) {
        if (arr[i] < pivot) {
            swap(arr[lt], arr[i]);
            lt++;
            i++;
        } else if (arr[i] > pivot) {
            swap(arr[i], arr[gt]);
            gt--;
        } else {
            i++;
        }
    }
}

// Simple 2-way partition for compatibility
inline int hybrid_partition(int arr[], int low, int high) {
    int lt, gt;
    partition_3way(arr, low, high, lt, gt);
    // Return middle of equal range for 2-way compatibility
    return (lt + gt) / 2;
}

// Cache-aware sequential quicksort with tail recursion
void cache_aware_quick_sort(int arr[], int low, int high, int depth) {
    const int cache_block_size = get_cache_block_size();
    const int max_depth = MAX_RECURSION_DEPTH_FACTOR * (int)log2(high - low + 1);
    
    // Switch to insertion sort if recursion depth too high
    if (depth > max_depth) {
        insertion_sort(arr + low, high - low + 1);
        return;
    }
    
    while (low < high && high - low > INSERTION_SORT_THRESHOLD) {
        // Use fast Hoare partition (same as sequential optimized)
        int pi = partition_hoare(arr, low, high);
        
        // Hoare returns last index of left partition
        int left_size = pi - low + 1;
        int right_size = high - pi;
        
        if (left_size < right_size) {
            if (left_size > 1) {
                cache_aware_quick_sort(arr, low, pi, depth + 1);
            }
            low = pi + 1;
        } else {
            if (right_size > 1) {
                cache_aware_quick_sort(arr, pi + 1, high, depth + 1);
            }
            high = pi;
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
    
    // Check recursion depth and switch to insertion sort if too deep
    const int max_depth = MAX_RECURSION_DEPTH_FACTOR * (int)log2(size);
    if (depth > max_depth) {
        insertion_sort(arr + low, size);
        return;
    }
    
    // Use fast Hoare partition (same as sequential optimized)
    int pi = partition_hoare(arr, low, high);
    
    // Calculate partition sizes (Hoare: pi is last of left partition)
    int left_size = pi - low + 1;
    int right_size = high - pi;
    
    // Early exit if partition very unbalanced - might be few_unique, use 3-way
    if (left_size < size / 16 || right_size < size / 16) {
        // Likely many duplicates, use 3-way partition
        int lt, gt;
        partition_3way(arr, low, high, lt, gt);
        left_size = lt - low;
        right_size = high - gt;
        
        if (left_size == 0 && right_size == 0) return;
        
        // Recurse on non-equal parts only
        if (left_size > 1) {
            cache_aware_quick_sort(arr, low, lt - 1, depth + 1);
        }
        if (right_size > 1) {
            cache_aware_quick_sort(arr, gt + 1, high, depth + 1);
        }
        return;
    }
    
    // Determine if we should use parallel tasks
    bool use_parallel = (size >= task_threshold);
    
    if (use_parallel) {
        // Check for potential false sharing
        bool potential_false_sharing = might_have_false_sharing(low, pi, pi + 1, high);
        
        if (potential_false_sharing) {
            // If false sharing might occur, use untied tasks for better work stealing
            if (left_size < right_size) {
                #pragma omp task untied firstprivate(arr, low, pi, depth, task_threshold) \
                    if(left_size > INSERTION_SORT_THRESHOLD)
                quick_sort_parallel(arr, low, pi, depth + 1, task_threshold);
                
                #pragma omp task untied firstprivate(arr, pi, high, depth, task_threshold) \
                    if(right_size > INSERTION_SORT_THRESHOLD)
                quick_sort_parallel(arr, pi + 1, high, depth + 1, task_threshold);
            } else {
                #pragma omp task untied firstprivate(arr, pi, high, depth, task_threshold) \
                    if(right_size > INSERTION_SORT_THRESHOLD)
                quick_sort_parallel(arr, pi + 1, high, depth + 1, task_threshold);
                
                #pragma omp task untied firstprivate(arr, low, pi, depth, task_threshold) \
                    if(left_size > INSERTION_SORT_THRESHOLD)
                quick_sort_parallel(arr, low, pi, depth + 1, task_threshold);
            }
        } else {
            // No false sharing concern, use regular tasks
            #pragma omp task firstprivate(arr, low, pi, depth, task_threshold) \
                if(left_size > INSERTION_SORT_THRESHOLD)
            quick_sort_parallel(arr, low, pi, depth + 1, task_threshold);
            
            #pragma omp task firstprivate(arr, pi, high, depth, task_threshold) \
                if(right_size > INSERTION_SORT_THRESHOLD)
            quick_sort_parallel(arr, pi + 1, high, depth + 1, task_threshold);
        }
        
        #pragma omp taskwait
        
    } else {
        // Sequential recursion - handle both partitions
        if (left_size > 1) {
            cache_aware_quick_sort(arr, low, pi, depth + 1);
        }
        if (right_size > 1) {
            cache_aware_quick_sort(arr, pi + 1, high, depth + 1);
        }
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