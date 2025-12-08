#include <algorithm>
#include <vector>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include "main_template.hpp"

using namespace std;

// ---------------------------
// Tunable parameters
// ---------------------------
const int INSERTION_THRESHOLD = 128;
const int TASK_THRESHOLD = 1 << 20;

// ---------------------------
// Insertion sort
// ---------------------------
void insertionSort(int *a, int low, int cnt)
{
    for (int i = low + 1; i < low + cnt; i++)
    {
        int key = a[i];
        int j = i;
        while (j > low && a[j - 1] > key)
            j--;
        if (j != i)
        {
            memmove(a + j + 1, a + j, (i - j) * sizeof(int));
            a[j] = key;
        }
    }
}

// ---------------------------
// Branchless compare & swap
// ---------------------------
inline void compAndSwapAsc(int &a, int &b)
{
    int minv = min(a, b);
    int maxv = max(a, b);
    a = minv;
    b = maxv;
}

// ---------------------------
// SIMD compare & swap 8 elem (AVX2)
// ---------------------------
inline void simdCompAndSwap8Asc(int *a, int *b)
{
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b));
    __m256i vmin = _mm256_min_epi32(va, vb);
    __m256i vmax = _mm256_max_epi32(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(a), vmin);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(b), vmax);
}

// ---------------------------
// Bitonički merge sa SIMD + scalar fallback
// ---------------------------
void bitonicMergeIter(int *a, int low, int cnt)
{
    for (int step = cnt / 2; step > 0; step /= 2)
    {
        for (int i = low; i < low + cnt; i += 2 * step)
        {
            int j = i;
            int end = i + step;

            // SIMD path: process 8 pairs at a time
            if (step >= 8)
            {
                for (; j + 7 < end; j += 8)
                {
                    simdCompAndSwap8Asc(a + j, a + j + step);
                }
            }

            // Scalar fallback
            for (; j < end; j++)
            {
                compAndSwapAsc(a[j], a[j + step]);
            }
        }
    }
}

// ---------------------------
// Paralelni bitonički merge
// ---------------------------
void bitonicMergeParallel(int *a, int low, int cnt)
{
    if (cnt <= TASK_THRESHOLD)
    {
        bitonicMergeIter(a, low, cnt);
        return;
    }

    int step = cnt / 2;

    // Parallel compare & swap
#pragma omp parallel for
    for (int i = low; i < low + step; i++)
    {
        compAndSwapAsc(a[i], a[i + step]);
    }

    // Recursive parallel merge
#pragma omp task
    bitonicMergeParallel(a, low, step);
#pragma omp task
    bitonicMergeParallel(a, low + step, cnt - step);
#pragma omp taskwait
}

// ---------------------------
// Paralelni bitonički sort
// ---------------------------
void bitonicSortIterParallel(int *a, int n)
{
    // Step 1: insertion sort male sekvence
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i += INSERTION_THRESHOLD)
    {
        int len = min(INSERTION_THRESHOLD, n - i);
        insertionSort(a, i, len);
    }

    // Step 2: build bitonic sequences
    for (int size = INSERTION_THRESHOLD; size <= n; size <<= 1)
    {
#pragma omp parallel for schedule(dynamic)
        for (int low = 0; low < n; low += size)
        {
            int cnt = min(size, n - low);
            int mid = cnt / 2;

            // Obrni desnu polovinu za bitoničku sekvencu
            if (mid < cnt)
            {
                int *l = a + low + mid;
                int *r = a + low + cnt - 1;
                while (l < r)
                    swap(*l++, *r--);
            }

            // Koristi paralelni merge za velike sekvence
            if (cnt >= TASK_THRESHOLD)
            {
#pragma omp task
                bitonicMergeParallel(a, low, cnt);
            }
            else
            {
                bitonicMergeIter(a, low, cnt);
            }
        }
#pragma omp taskwait
    }
}

// ---------------------------
// Wrapper
// ---------------------------
void bitonic_sort_wrapper(std::vector<int> &vec)
{
    bitonicSortIterParallel(vec.data(), vec.size());
}

// ---------------------------
// Main
// ---------------------------
int main(int argc, char *argv[])
{
    return run_sort("bitonic_sort_parallel_simd", "parallel_simd", bitonic_sort_wrapper, argc, argv);
}
