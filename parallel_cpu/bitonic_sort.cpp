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
const int SIMD_BYTES = 32;

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
// Compare & swap
// ---------------------------
inline void compAndSwap(int *a, int i, int j, bool asc)
{
    if ((asc && a[i] > a[j]) || (!asc && a[i] < a[j]))
        swap(a[i], a[j]);
}

// ---------------------------
// Iterativni bitonički merge
// ---------------------------
void bitonicMergeIter(int *a, int low, int cnt, bool asc)
{
    for (int step = cnt / 2; step > 0; step /= 2)
    {
        for (int i = low; i < low + cnt; i += 2 * step)
        {
            for (int j = i; j < i + step; j++)
            {
                compAndSwap(a, j, j + step, asc);
            }
        }
    }
}

// ---------------------------
// Paralelni bitonički merge - ISPRAVLJEN
// ---------------------------
void bitonicMergeParallel(int *a, int low, int cnt, bool asc)
{
    if (cnt <= TASK_THRESHOLD)
    {
        bitonicMergeIter(a, low, cnt, asc);
        return;
    }

    int step = cnt / 2;

// Paralelno uporedi i swap-uj elemente
#pragma omp parallel for
    for (int i = low; i < low + step; i++)
    {
        compAndSwap(a, i, i + step, asc);
    }

// Paralelno merge-uj podsekvence
#pragma omp task
    bitonicMergeParallel(a, low, step, asc);

#pragma omp task
    bitonicMergeParallel(a, low + step, cnt - step, asc);

#pragma omp taskwait
}

// ---------------------------
// Paralelni bitonički sort - ISPRAVLJEN
// ---------------------------
void bitonicSortIterParallel(int *a, int n)
{
// Korak 1: insertion sort male sekvence
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i += INSERTION_THRESHOLD)
    {
        int len = min(INSERTION_THRESHOLD, n - i);
        insertionSort(a, i, len);
    }

    // Korak 2: gradnja bitoničkih sekvenci
    for (int size = INSERTION_THRESHOLD; size <= n; size <<= 1)
    {
#pragma omp parallel for schedule(dynamic)
        for (int low = 0; low < n; low += size)
        {
            int mid = size / 2;
            int cnt = min(size, n - low);

            // Obrni desnu polovinu
            if (mid < cnt)
            {
                int *l = a + low + mid;
                int *r = a + low + cnt - 1;
                while (l < r)
                    swap(*l++, *r--);
            }

            // ISPRAVKA: Koristi paralelni merge za velike sekvence
            if (cnt >= TASK_THRESHOLD)
            {
#pragma omp task
                bitonicMergeParallel(a, low, cnt, true);
            }
            else
            {
                bitonicMergeIter(a, low, cnt, true);
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
    return run_sort("bitonic_sort_iterative_parallel_fixed", "parallel", bitonic_sort_wrapper, argc, argv);
}
