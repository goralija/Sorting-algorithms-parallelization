// Quick sort in C++
#include <iostream>
#include <vector>
#include "main_template.hpp"

using namespace std;

// Partition function (always uses last element as pivot - worst case on sorted data)
int partition(int arr[], int low, int high) {
    int pivot = arr[high];   // Always pick last element (no median-of-three)
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            // Inline swap (no function call optimization)
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    // Inline swap for pivot
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    
    return (i + 1);
}

// Basic recursive quick sort (no tail recursion elimination, no insertion sort fallback)
void quick_sort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        
        // Always recurse both sides (no optimization for smaller partition first)
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

// Wrapper for template compatibility
void quick_sort_wrapper(std::vector<int>& vec) {
    quick_sort(vec.data(), 0, static_cast<int>(vec.size()) - 1);
}

// Entry point (uses common benchmarking template)
int main(int argc, char* argv[]) {
    return run_sort("quick_sort", "sequential", quick_sort_wrapper, argc, argv);
}
