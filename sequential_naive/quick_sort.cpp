// Quick sort in C++
#include <iostream>
#include <vector>
#include <cmath>
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

// Simple insertion sort for fallback
void insertion_sort(int arr[], int low, int high) {
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

// Basic recursive quick sort with depth limit to prevent worst-case stack overflow
void quick_sort(int arr[], int low, int high, int depth, int max_depth) {
    if (low < high) {
        // If recursion too deep, fall back to insertion sort to prevent stack overflow
        if (depth > max_depth) {
            insertion_sort(arr, low, high);
            return;
        }
        
        int pi = partition(arr, low, high);
        
        // Always recurse both sides (no optimization for smaller partition first)
        quick_sort(arr, low, pi - 1, depth + 1, max_depth);
        quick_sort(arr, pi + 1, high, depth + 1, max_depth);
    }
}

// Wrapper for template compatibility
void quick_sort_wrapper(std::vector<int>& vec) {
    if (vec.empty()) return;
    int n = vec.size();
    // Set max depth to 3 * log2(n) - higher to allow more recursion before fallback
    int max_depth = 3 * (int)log2(n);
    quick_sort(vec.data(), 0, n - 1, 0, max_depth);
}

// Entry point (uses common benchmarking template)
int main(int argc, char* argv[]) {
    return run_sort("quick_sort", "sequential", quick_sort_wrapper, argc, argv);
}
