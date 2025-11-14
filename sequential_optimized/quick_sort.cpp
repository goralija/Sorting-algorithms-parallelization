// Optimized Quick Sort in C++
#include <iostream>
#include <vector>
#include <algorithm> // for std::swap
#include "main_template.hpp"

using namespace std;

const int INSERTION_SORT_THRESHOLD = 16; // Keep small for best performance

// Simple, fast insertion sort
inline void insertion_sort(int arr[], int low, int high) {
    for (int i = low + 1; i <= high; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= low && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Simple median-of-three pivot selection
inline int median_of_three(int arr[], int low, int high) {
    int mid = low + ((high - low) >> 1);
    
    if (arr[mid] < arr[low])
        swap(arr[mid], arr[low]);
    if (arr[high] < arr[low])
        swap(arr[high], arr[low]);
    if (arr[high] < arr[mid])
        swap(arr[high], arr[mid]);
    
    return mid;
}

// Fast Hoare partition scheme
inline int partition(int arr[], int low, int high) {
    int pivot_idx = median_of_three(arr, low, high);
    int pivot = arr[pivot_idx];
    
    int i = low - 1;
    int j = high + 1;
    
    while (true) {
        do { i++; } while (arr[i] < pivot);
        do { j--; } while (arr[j] > pivot);
        
        if (i >= j)
            return j;
        
        swap(arr[i], arr[j]);
    }
}

// Optimized quicksort with tail recursion elimination
void quick_sort(int arr[], int low, int high) {
    while (low < high) {
        // Use insertion sort for small partitions
        if (high - low < INSERTION_SORT_THRESHOLD) {
            insertion_sort(arr, low, high);
            return;
        }
        
        int pi = partition(arr, low, high);
        
        // Recursively sort the smaller partition first (limits stack depth)
        // Then use tail recursion for the larger one
        if (pi - low < high - pi) {
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
