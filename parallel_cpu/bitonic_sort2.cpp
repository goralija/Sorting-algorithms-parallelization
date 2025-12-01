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
const int INSERTION_THRESHOLD = 128; // male sekvence insertion sortom
const int TASK_THRESHOLD = 1 << 15;  // minimalna veličina niza za OpenMP task
const int SIMD_WIDTH = 8;            // AVX2 = 8 ints po registru

// ---------------------------
// Aligned SIMD copy helper
// ---------------------------
inline void simd_minmax_swap(int *a, int i, int j, bool asc)
{
    __m256i va = _mm256_loadu_si256((__m256i *)(a + i));
    __m256i vb = _mm256_loadu_si256((__m256i *)(a + j));
    __m256i vmin = _mm256_min_epi32(va, vb);
    __m256i vmax = _mm256_max_epi32(va, vb);
    if (asc)
    {
        _mm256_storeu_si256((__m256i *)(a + i), vmin);
        _mm256_storeu_si256((__m256i *)(a + j), vmax);
    }
    else
    {
        _mm256_storeu_si256((__m256i *)(a + i), vmax);
        _mm256_storeu_si256((__m256i *)(a + j), vmin);
    }
}

// ---------------------------
// Insertion sort sa memmove
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
// Bitonic merge (SIMD + scalar tail)
// ---------------------------
void bitonicMergeSIMD(int *a, int low, int cnt, bool asc)
{
    for (int step = cnt / 2; step > 0; step /= 2)
    {
        int i = low;
        int limit = low + cnt - SIMD_WIDTH;
        for (; i < limit; i += 2 * step)
        {
            for (int j = i; j < i + step && j + SIMD_WIDTH - 1 < low + cnt; j += SIMD_WIDTH)
            {
                simd_minmax_swap(a, j, j + step, asc);
            }
        }
        // scalar fallback za tail
        for (; i + step < low + cnt; ++i)
        {
            if ((asc && a[i] > a[i + step]) || (!asc && a[i] < a[i + step]))
                swap(a[i], a[i + step]);
        }
    }
}

// ---------------------------
// Paralelni bitonički merge
// ---------------------------
void bitonicMergeParallel(int *a, int low, int cnt, bool asc)
{
    if (cnt <= TASK_THRESHOLD)
    {
        bitonicMergeSIMD(a, low, cnt, asc);
        return;
    }

    int step = cnt / 2;

    // paralelni swap prvog step-a
#pragma omp parallel for schedule(static)
    for (int i = low; i < low + step; i += SIMD_WIDTH)
    {
        if (i + SIMD_WIDTH <= low + step)
            simd_minmax_swap(a, i, i + step, asc);
        else
        { // tail scalar
            for (int j = i; j < low + step; ++j)
            {
                if ((asc && a[j] > a[j + step]) || (!asc && a[j] < a[j + step]))
                    swap(a[j], a[j + step]);
            }
        }
    }

    // paralelni task merge podsekvenci
#pragma omp task shared(a) firstprivate(low, step, asc)
    bitonicMergeParallel(a, low, step, asc);

#pragma omp task shared(a) firstprivate(low, step, cnt, asc)
    bitonicMergeParallel(a, low + step, cnt - step, asc);

#pragma omp taskwait
}

// ---------------------------
// Iterativni bitonički sort sa paralelizacijom i SIMD
// ---------------------------
void bitonicSortIterParallelSIMD(int *a, int n)
{
    // Step 1: insertion sort male sekvence
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i += INSERTION_THRESHOLD)
    {
        int len = min(INSERTION_THRESHOLD, n - i);
        insertionSort(a, i, len);
    }

    // Step 2: gradnja bitoničkih sekvenci
    for (int size = INSERTION_THRESHOLD; size <= n; size <<= 1)
    {
#pragma omp parallel for schedule(dynamic)
        for (int low = 0; low < n; low += size)
        {
            int cnt = min(size, n - low);
            int mid = cnt / 2;

            // obrni desnu polovinu da bude opadajuća
            if (mid < cnt)
            {
                int *l = a + low + mid;
                int *r = a + low + cnt - 1;
                while (l < r)
                    swap(*l++, *r--);
            }

            // paralelni merge ili scalar za male sekvence
#pragma omp parallel
            {
#pragma omp single nowait
                bitonicMergeParallel(a + low, 0, cnt, true);
            }
        }
    }
}

// ---------------------------
// Wrapper
// ---------------------------
void bitonic_sort_wrapper(std::vector<int> &vec)
{
    bitonicSortIterParallelSIMD(vec.data(), vec.size());
}

// ---------------------------
// Main
// ---------------------------
int main(int argc, char *argv[])
{
    return run_sort("bitonic_sort_turbo_simd", "parallel", bitonic_sort_wrapper, argc, argv);
}
