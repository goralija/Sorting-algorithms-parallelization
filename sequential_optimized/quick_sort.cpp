// Quick sort in C++
#include <iostream>
#include <vector>
#include "main_template.hpp"

using namespace std;

// Swap two elements
inline void swap_elements(int& a, int& b) {
    int t = a;
    a = b;
    b = t;
}

// Partition function
int partition(int arr[], int low, int high) {
    int pivot = arr[high];   // pivot element
    int i = low - 1;         // index of smaller element

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap_elements(arr[i], arr[j]);
        }
    }

    swap_elements(arr[i + 1], arr[high]);
    return (i + 1);
}

// Recursive quick sort
void quick_sort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        // Recursively sort elements before and after partition
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

// Wrapper for template compatibility
void quick_sort_wrapper(std::vector<int>& vec) {
    if (!vec.empty())
        quick_sort(vec.data(), 0, static_cast<int>(vec.size()) - 1);
}

// Entry point (uses common benchmarking template)
int main(int argc, char* argv[]) {
    return run_sort("quick_sort", "sequential", quick_sort_wrapper, argc, argv);
}
