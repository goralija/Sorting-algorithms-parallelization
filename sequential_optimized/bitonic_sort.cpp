#include <algorithm> // for std::swap
#include <vector>
#include <iostream>
#include "main_template.hpp"
#include <cstring> 

using namespace std;

// insertion sort za male sekvence
void insertionSort(int *a, int low, int cnt) {
    for (int i = low + 1; i < low + cnt; i++) {
        int key = a[i];
        int j = i;

        // pronađi poziciju gdje ide key
        while (j > low && a[j - 1] > key) j--;

        // pomakni sve elemente desno od j do i-1
        if (j != i) {
            memmove(a + j + 1, a + j, (i - j) * sizeof(int));
            a[j] = key;
        }
    }
}

// compare & swap
inline void compAndSwap(int *a, int i, int j, bool asc) {
    int ai = a[i], aj = a[j];
    if (asc) {
        int swap = ai > aj;
        int tmp = swap ? ai : aj;
        a[i] = swap ? aj : ai;
        a[j] = tmp;
    } else {
        int swap = ai < aj;
        int tmp = swap ? aj : ai;
        a[i] = swap ? ai : aj;
        a[j] = tmp;
    }
}


// iterativna verzija bitonic merge-a (bez rekurzije)
void bitonicMergeIter(int *a, int low, int cnt, bool asc) {
    for (int size = cnt; size > 1; size >>= 1) {
        int half = size >> 1;
        for (int i = low; i < low + cnt; i += size) {
            for (int j = i; j < i + half; j++) {
                compAndSwap(a, j, j + half, asc);
            }
        }
    }
}

// optimizovana verzija bitonic sorta sa insertion sort pragom
void bitonicSortOptimized(int *a, int low, int cnt, bool asc) {
    const int INSERTION_THRESHOLD = 32;
    
    if (cnt <= INSERTION_THRESHOLD) {
        insertionSort(a, low, cnt);
        if (!asc) {
            std::reverse(a + low, a + low + cnt);
        }
        return;
    }

    int k = cnt / 2;
    bitonicSortOptimized(a, low, k, true);
    bitonicSortOptimized(a, low + k, k, false);
    bitonicMergeIter(a, low, cnt, asc);
}

// wrapper – ista forma kao bitonic_sort_wrapper u tvom kodu
void bitonic_sort_wrapper(std::vector<int>& vec) {
    // ovdje pretpostavljamo da je vec.size() = 2^k, nema provjere
    bitonicSortOptimized(vec.data(), 0, vec.size(), true);
}

// main – kao i kod merge sorta
int main(int argc, char* argv[]) {
    //return run_sort("bitonic_sort_optimized", "sequential", bitonic_sort_wrapper, argc, argv);
}
