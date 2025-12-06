// hybrid_pingpong_parallel_merge_sort.cpp
// Bottom-up ping-pong parallel merge sort optimized for cache locality and OpenMP
// Uses insertion sort for small runs and parallel for across merge pairs
// Compile with: -O3 -mavx2 -march=native -fopenmp  (GCC/Clang)
// On Windows MSVC: /O2 /arch:AVX2 /openmp

#include <vector>
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <iostream>
#include <cstdlib>

#if defined(_WIN32)
#include <malloc.h>
#endif
#include <main_template.hpp>

// Tunable parameters (adjust to your CPU / dataset)
constexpr int INSERTION_SORT_THRESHOLD = 64;  // base-case sort size
constexpr int PARALLEL_MERGE_WIDTH = 1 << 20; // start parallel merging when run width >= 64k
constexpr size_t SIMD_BYTES = 32;             // AVX2
constexpr size_t ALIGN_BYTES = 64;

// ---------- aligned allocation helpers ----------
inline int *allocate_temp_buffer(size_t n)
{
    size_t bytes = n * sizeof(int);
#if defined(_WIN32)
    void *p = _aligned_malloc(bytes, ALIGN_BYTES);
    return reinterpret_cast<int *>(p);
#else
    void *p = nullptr;
    if (posix_memalign(&p, ALIGN_BYTES, bytes) == 0)
        return reinterpret_cast<int *>(p);
    return reinterpret_cast<int *>(malloc(bytes));
#endif
}
inline void free_temp_buffer(int *p)
{
    if (!p)
        return;
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

// ---------- SIMD memcpy (AVX2-friendly, falls back to byte copy) ----------
static inline void simd_memcpy(void *dst_v, const void *src_v, size_t bytes)
{
    unsigned char *dst = reinterpret_cast<unsigned char *>(dst_v);
    const unsigned char *src = reinterpret_cast<const unsigned char *>(src_v);
    size_t i = 0;
    const size_t step = SIMD_BYTES;
    size_t nvec = bytes / step;
    for (size_t k = 0; k < nvec; ++k)
    {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), v);
        i += step;
    }
    size_t rem = bytes - i;
    for (size_t j = 0; j < rem; ++j)
        dst[i + j] = src[i + j];
}
static inline void simd_copy_ints(int *dst, const int *src, size_t count)
{
    if (count == 0)
        return;
    simd_memcpy(dst, src, count * sizeof(int));
}

// ---------- insertion sort ----------
static inline void insertion_sort(int arr[], int l, int r)
{
    for (int i = l + 1; i <= r; ++i)
    {
        int key = arr[i];
        int j = i - 1;
        while (j >= l && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

// ---------- merge two runs src[l..m] and src[m+1..r] into dest[l..r] ----------
static inline void merge_runs(const int src[], int dest[], int l, int m, int r)
{
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r)
    {
        dest[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    }
    while (i <= m)
        dest[k++] = src[i++];
    while (j <= r)
        dest[k++] = src[j++];
}

// ---------- hybrid bottom-up ping-pong parallel merge sort ----------
void merge_sort_hybrid_pingpong(int arr[], int n)
{
    if (n <= 1)
        return;

    int *temp = allocate_temp_buffer((size_t)n);
    if (!temp)
    {
        temp = new int[n];
    }

    int *src = arr;
    int *dst = temp;

// Step 1: sort small blocks with insertion sort
#pragma omp parallel for schedule(static)
    for (int start = 0; start < n; start += INSERTION_SORT_THRESHOLD)
    {
        int end = std::min(n, start + INSERTION_SORT_THRESHOLD) - 1;
        insertion_sort(src, start, end);
    }

    // Step 2: bottom-up merge; width is current run size
    for (int width = INSERTION_SORT_THRESHOLD; width < n; width *= 2)
    {
        int two_width = width * 2;
        int num_pairs = (n + two_width - 1) / two_width;

        // Decide whether to parallelize this merge level
        if (width >= PARALLEL_MERGE_WIDTH)
        {
// Parallelize across pairs: each iteration merges one pair (or copies remainder)
#pragma omp parallel for schedule(dynamic)
            for (int p = 0; p < num_pairs; ++p)
            {
                int l = p * two_width;
                int m = std::min(l + width - 1, n - 1);
                int r = std::min(l + two_width - 1, n - 1);
                if (l > r)
                    continue;
                if (m >= r)
                {
                    // No merge, just copy
                    int len = r - l + 1;
                    if (len > 0)
                        simd_copy_ints(dst + l, src + l, (size_t)len);
                }
                else
                {
                    // Skip merge if already ordered
                    if (src[m] <= src[m + 1])
                    {
                        int len = r - l + 1;
                        simd_copy_ints(dst + l, src + l, (size_t)len);
                    }
                    else
                    {
                        merge_runs(src, dst, l, m, r);
                    }
                }
            } // end parallel for
        }
        else
        {
            // Serial merge level (no thread overhead)
            for (int p = 0; p < num_pairs; ++p)
            {
                int l = p * two_width;
                int m = std::min(l + width - 1, n - 1);
                int r = std::min(l + two_width - 1, n - 1);
                if (l > r)
                    continue;
                if (m >= r)
                {
                    int len = r - l + 1;
                    if (len > 0)
                        simd_copy_ints(dst + l, src + l, (size_t)len);
                }
                else
                {
                    if (src[m] <= src[m + 1])
                    {
                        int len = r - l + 1;
                        simd_copy_ints(dst + l, src + l, (size_t)len);
                    }
                    else
                    {
                        merge_runs(src, dst, l, m, r);
                    }
                }
            } // end serial for
        }

        // swap buffers
        std::swap(src, dst);
    } // end for width

    // If result is in temp (src == temp), copy back to arr
    if (src != arr)
    {
        simd_copy_ints(arr, src, (size_t)n);
    }

    free_temp_buffer(temp);
}

// ---------- wrapper ----------
void merge_sort_wrapper(std::vector<int> &vec)
{
    merge_sort_hybrid_pingpong(vec.data(), (int)vec.size());
}
int main(int argc, char *argv[])
{
    return run_sort("merge_sort", "parallel_opt", merge_sort_wrapper, argc, argv);
}
