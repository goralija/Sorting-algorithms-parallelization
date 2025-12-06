// Naive Radix Sort in C++
// Basic LSD (Least Significant Digit) radix sort implementation
// No optimizations - simple and straightforward

#include <iostream>
#include <vector>
#include "main_template.hpp"

using namespace std;

// A utility function to get maximum value in arr[]
int getMax(int arr[], int n) {
    int mx = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > mx) {
            mx = arr[i];
        }
    }
    return mx;
}

// A function to do counting sort of arr[] according to
// the digit represented by exp (10^i)
void countSort(int arr[], int n, int exp) {
    // Output array - using VLA (simple but not optimal)
    int* output = new int[n];
    int count[10] = {0};

    // Store count of occurrences in count[]
    for (int i = 0; i < n; i++) {
        count[(arr[i] / exp) % 10]++;
    }

    // Change count[i] so that count[i] now contains
    // actual position of this digit in output[]
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    // Build the output array (traverse from end for stability)
    for (int i = n - 1; i >= 0; i--) {
        int digit = (arr[i] / exp) % 10;
        output[count[digit] - 1] = arr[i];
        count[digit]--;
    }

    // Copy the output array to arr[]
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }

    delete[] output;
}

// The main function that sorts arr[] of size n using Radix Sort
void radix_sort(int arr[], int n) {
    if (n <= 1) return;

    // Find the maximum number to know number of digits
    int m = getMax(arr, n);

    // Do counting sort for every digit
    // exp is 10^i where i is current digit number
    for (int exp = 1; m / exp > 0; exp *= 10) {
        countSort(arr, n, exp);
    }
}

// Wrapper for template compatibility
void radix_sort_wrapper(std::vector<int>& vec) {
    if (vec.empty()) return;
    radix_sort(vec.data(), static_cast<int>(vec.size()));
}

// Entry point (uses common benchmarking template)
int main(int argc, char* argv[]) {
    return run_sort("radix_sort", "sequential", radix_sort_wrapper, argc, argv);
}
