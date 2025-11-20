// Optimized Quick Sort in C++
#include <iostream>
#include <vector>
#include <algorithm> // for std::swap
#include "main_template.hpp"

using namespace std;

const int INSERTION_SORT_THRESHOLD = 16; // Keep small for best performance

// Optimized insertion sort with sentinels (no bounds checking in inner loop)
inline void insertion_sort(int* arr, int n) {
    // Place smallest element at position 0 as sentinel
    int min_idx = 0;
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[min_idx]) min_idx = i;
    }
    if (min_idx != 0) swap(arr[0], arr[min_idx]);
    
    // Now we can use unguarded insertion (no j >= 0 check needed)
    for (int i = 2; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        // No bounds check needed - sentinel guarantees we'll stop
        while (arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Inline median-of-three (no function call, directly in partition)
// Fast Hoare partition with integrated median-of-three
inline int partition(int arr[], int low, int high) {
    // Inline median-of-three
    int mid = low + ((high - low) >> 1);
    if (arr[mid] < arr[low]) swap(arr[mid], arr[low]);
    if (arr[high] < arr[low]) swap(arr[high], arr[low]);
    if (arr[high] < arr[mid]) swap(arr[high], arr[mid]);
    
    int pivot = arr[mid];
    int i = low - 1;
    int j = high + 1;
    
    // Hoare partition with fast loop
    while (true) {
        do { i++; } while (arr[i] < pivot);
        do { j--; } while (arr[j] > pivot);
        
        if (i >= j) return j;
        
        swap(arr[i], arr[j]);
    }
}

// Optimized quicksort with tail recursion elimination
void quick_sort(int arr[], int low, int high) {
    while (low < high) {
        int size = high - low + 1;
        
        // Use insertion sort for small partitions
        if (size <= INSERTION_SORT_THRESHOLD) {
            insertion_sort(arr + low, size);
            return;
        }
        
        int pi = partition(arr, low, high);
        
        // Calculate partition sizes once
        int left_size = pi - low + 1;
        int right_size = high - pi;
        
        // Recursively sort the smaller partition first (O(log n) stack depth)
        // Then use tail recursion for the larger one
        if (left_size < right_size) {
            quick_sort(arr, low, pi);
            low = pi + 1;
        } else {
            quick_sort(arr, pi + 1, high);
            high = pi;
        }
    }
}

// Wrapper for benchmarking template 
void quick_sort_optimized_wrapper(std::vector<int>& vec) {
    if (!vec.empty())
        quick_sort(vec.data(), 0, static_cast<int>(vec.size()) - 1);
}

int main(int argc, char* argv[]) {
    return run_sort("quick_sort_optimized", "sequential", quick_sort_optimized_wrapper, argc, argv);
}
