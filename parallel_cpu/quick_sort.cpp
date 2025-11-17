// parallel_cpu/quick_sort_avx_omp.cpp
// Parallel + AVX optimized version of quicksort
// Uses OpenMP tasks + AVX vectorization for partitioning
// Compile with -O3 -march=native -fopenmp and AVX2 support

#include <vector>
#include <algorithm>
#include <cstring>
#include <immintrin.h>  // AVX intrinsics
#include <omp.h>
#include "main_template.hpp"

const int INSERTION_SORT_THRESHOLD = 16;
const int TASK_THRESHOLD = 1 << 16;      // ~65536
const int AVX_PARTITION_THRESHOLD = 1024; // Use AVX for larger partitions

// Your optimized insertion sort (unchanged)
inline void insertion_sort(int* arr, int n) {
    int min_idx = 0;
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[min_idx]) min_idx = i;
    }
    if (min_idx != 0) std::swap(arr[0], arr[min_idx]);
    
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

// AVX-vectorized partition using SIMD comparisons
inline int avx_partition(int arr[], int low, int high) {
    // Median-of-three pivot selection
    int mid = low + ((high - low) >> 1);
    if (arr[mid] < arr[low]) std::swap(arr[mid], arr[low]);
    if (arr[high] < arr[low]) std::swap(arr[high], arr[low]);
    if (arr[high] < arr[mid]) std::swap(arr[high], arr[mid]);
    
    int pivot = arr[mid];
    int i = low - 1;
    int j = high + 1;
    
    // Use AVX for vectorized comparison when partition is large enough
    if (high - low + 1 >= AVX_PARTITION_THRESHOLD) {
        __m256i pivot_vec = _mm256_set1_epi32(pivot);
        int* left_ptr = arr + low;
        int* right_ptr = arr + high - 7; // Start from high-7 for AVX
        
        while (left_ptr < right_ptr) {
            // Load 8 elements from left
            __m256i left_data = _mm256_loadu_si256(reinterpret_cast<__m256i*>(left_ptr));
            __m256i left_cmp = _mm256_cmpgt_epi32(pivot_vec, left_data);
            int left_mask = _mm256_movemask_ps(_mm256_castsi256_ps(left_cmp));
            
            // Load 8 elements from right  
            __m256i right_data = _mm256_loadu_si256(reinterpret_cast<__m256i*>(right_ptr));
            __m256i right_cmp = _mm256_cmpgt_epi32(right_data, pivot_vec);
            int right_mask = _mm256_movemask_ps(_mm256_castsi256_ps(right_cmp));
            
            // Find elements to swap
            if (left_mask != 0 && right_mask != 0) {
                // Simple case: swap first mismatched elements
                for (int k = 0; k < 8; k++) {
                    if (!(left_mask & (1 << k)) && (right_mask & (1 << k))) {
                        std::swap(left_ptr[k], right_ptr[k]);
                        break;
                    }
                }
            }
            
            left_ptr += 8;
            right_ptr -= 8;
        }
    }
    
    // Fall back to scalar Hoare partition for final steps
    while (true) {
        do { i++; } while (arr[i] < pivot);
        do { j--; } while (arr[j] > pivot);
        
        if (i >= j) return j;
        std::swap(arr[i], arr[j]);
    }
}

// Scalar partition for smaller arrays (your original optimized version)
inline int scalar_partition(int arr[], int low, int high) {
    int mid = low + ((high - low) >> 1);
    if (arr[mid] < arr[low]) std::swap(arr[mid], arr[low]);
    if (arr[high] < arr[low]) std::swap(arr[high], arr[low]);
    if (arr[high] < arr[mid]) std::swap(arr[high], arr[mid]);
    
    int pivot = arr[mid];
    int i = low - 1;
    int j = high + 1;
    
    while (true) {
        do { i++; } while (arr[i] < pivot);
        do { j--; } while (arr[j] > pivot);
        
        if (i >= j) return j;
        std::swap(arr[i], arr[j]);
    }
}

// Main parallel quicksort with OpenMP tasks
void quick_sort_parallel(int arr[], int low, int high) {
    int size = high - low + 1;
    
    // Base case: use insertion sort for small arrays
    if (size <= INSERTION_SORT_THRESHOLD) {
        insertion_sort(arr + low, size);
        return;
    }
    
    // Choose partition method based on size
    int pi;
    if (size >= AVX_PARTITION_THRESHOLD) {
        pi = avx_partition(arr, low, high);
    } else {
        pi = scalar_partition(arr, low, high);
    }
    
    // Calculate partition sizes
    int left_size = pi - low + 1;
    int right_size = high - pi;
    
    // Use OpenMP tasks for parallel recursion
    if (size >= TASK_THRESHOLD) {
        #pragma omp task shared(arr) firstprivate(low, pi) if(left_size > TASK_THRESHOLD/2)
        quick_sort_parallel(arr, low, pi);
        
        #pragma omp task shared(arr) firstprivate(pi, high) if(right_size > TASK_THRESHOLD/2)  
        quick_sort_parallel(arr, pi + 1, high);
        
        #pragma omp taskwait
    } else {
        // Sequential recursion for smaller partitions
        quick_sort_parallel(arr, low, pi);
        quick_sort_parallel(arr, pi + 1, high);
    }
}

// Wrapper function with OpenMP parallel region
void quick_sort_parallel_wrapper(std::vector<int>& vec) {
    if (vec.empty()) return;
    
    #pragma omp parallel
    {
        #pragma omp single nowait
        quick_sort_parallel(vec.data(), 0, static_cast<int>(vec.size()) - 1);
    }
}

int main(int argc, char* argv[]) {
    return run_sort("quick_sort_avx_omp", "parallel_avx", quick_sort_parallel_wrapper, argc, argv);
}