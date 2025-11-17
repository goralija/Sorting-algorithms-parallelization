// Bitonic Sort in C++
#include <iostream>
#include <vector>
#include "main_template.hpp"

using namespace std;

// Compare and swap
void compAndSwap(int *a, int i, int j, bool asc) {
    if (asc) {
        if (a[i] > a[j])
            std::swap(a[i], a[j]);
    } else {
        if (a[i] < a[j])
            std::swap(a[i], a[j]);
    }
}

// Merge faza bitoničke sekvence
void bitonicMerge(int *a, int low, int cnt, bool asc) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            compAndSwap(a, i, i + k, asc);
        }
        bitonicMerge(a, low, k, asc);
        bitonicMerge(a, low + k, k, asc);
    }
}

// Rekurzivna izgradnja bitoničke sekvence
void bitonicSort(int *a, int low, int cnt, bool asc) {
    if (cnt > 1) {
        int k = cnt / 2;
        // lijeva polovina u rastućem
        bitonicSort(a, low, k, true);
        // desna polovina u opadajućem
        bitonicSort(a, low + k, k, false);
        // spoji u pravom smjeru (asc/opadajuće)
        bitonicMerge(a, low, cnt, asc);
    }
}

// Wrapper za run_sort (kao merge_sort_wrapper)
void bitonic_sort_wrapper(std::vector<int>& vec) {
    bitonicSort(vec.data(), 0, vec.size(), true); // default ascending
}

// Main – identična struktura kao kod merge sorta
int main(int argc, char* argv[]) {
    //return run_sort("bitonic_sort", "sequential", bitonic_sort_wrapper, argc, argv);
}
