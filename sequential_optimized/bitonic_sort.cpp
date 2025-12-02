#include <algorithm>
#include <vector>
#include <cstring>
#include <immintrin.h>
#include "main_template.hpp"

using namespace std;

// ---------------------------
// Tunable parameters
// ---------------------------
const int INSERTION_THRESHOLD = 32;

// ---------------------------
// Insertion sort for small sequences
// ---------------------------
inline void insertionSort(int *a, int low, int cnt)
{
    for (int i = low + 1; i < low + cnt; i++)
    {
        int key = a[i];
        int j = i - 1;
        while (j >= low && a[j] > key)
        {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

// ---------------------------
// Branchless compare & swap (much faster!)
// ---------------------------
inline void compAndSwapAsc(int &a, int &b)
{
    int minv = min(a, b);
    int maxv = max(a, b);
    a = minv;
    b = maxv;
}

// ---------------------------
// SIMD compare & swap for 8 pairs at once (AVX2)
// ---------------------------
inline void simdCompAndSwap8Asc(int *a, int *b)
{
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
    __m256i vmin = _mm256_min_epi32(va, vb);
    __m256i vmax = _mm256_max_epi32(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(a), vmin);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(b), vmax);
}

// ---------------------------
// Optimized bitonic merge with SIMD
// ---------------------------
void bitonicMergeIter(int *a, int low, int cnt)
{
    for (int step = cnt / 2; step > 0; step /= 2)
    {
        // Process pairs
        for (int i = low; i < low + cnt; i += 2 * step)
        {
            int j = i;
            int end = i + step;
            
            // SIMD path: process 8 pairs at a time when step allows
            if (step >= 8)
            {
                for (; j + 7 < end; j += 8)
                {
                    simdCompAndSwap8Asc(a + j, a + j + step);
                }
            }
            
            // Scalar remainder
            for (; j < end; j++)
            {
                compAndSwapAsc(a[j], a[j + step]);
            }
        }
    }
}

// ---------------------------
// Optimized bitonic sort
// ---------------------------
void bitonicSortOpt(int *a, int n)
{
    // Step 1: sort small sequences with insertion sort
    for (int i = 0; i < n; i += INSERTION_THRESHOLD)
    {
        int len = min(INSERTION_THRESHOLD, n - i);
        insertionSort(a, i, len);
    }

    // Step 2: build bitonic sequences iteratively
    for (int size = INSERTION_THRESHOLD; size <= n; size *= 2)
    {
        for (int low = 0; low < n; low += size)
        {
            int cnt = min(size, n - low);
            int mid = cnt / 2;

            // Reverse right half to create bitonic sequence
            if (mid > 0 && mid < cnt)
            {
                int *l = a + low + mid;
                int *r = a + low + cnt - 1;
                while (l < r)
                {
                    swap(*l, *r);
                    l++;
                    r--;
                }
            }

            bitonicMergeIter(a, low, cnt);
        }
    }
}

// ---------------------------
// Wrapper za run_sort
// ---------------------------
void bitonic_sort_wrapper(std::vector<int> &vec)
{
    bitonicSortOpt(vec.data(), vec.size());
}

// ---------------------------
// Main
// ---------------------------
int main(int argc, char *argv[])
{
    return run_sort("bitonic_sort", "sequential_optimized", bitonic_sort_wrapper, argc, argv);
}
