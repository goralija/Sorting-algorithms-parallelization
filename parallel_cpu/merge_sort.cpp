// parallel_cpu/merge_sort.cpp
// Ultra-stable parallel merge sort (Merge-Path) with AVX2-accelerated memcpy/copy
// Uses OpenMP tasks + Merge Path for parallel merging
// Compile with: -O3 -mavx2 -march=native -fopenmp

#include <vector>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <climits>
#include <immintrin.h>
#include <omp.h>
#include <main_template.hpp>
#include <cstdint>
#include <cstdlib>

#if defined(_WIN32)
#include <malloc.h>
#endif

// ---------------------- Tunable parameters ----------------------
const int INSERTION_SORT_THRESHOLD = 128;   // for small runs
const int TASK_THRESHOLD = 1 << 20;         // spawn OpenMP tasks for ranges >= 65536
const int MERGE_PATH_MIN_CHUNK = 16384 * 4; // min total size to use merge-path splitting
const int SIMD_BYTES = 32;                  // AVX2 = 256 bits = 32 bytes
const size_t ALIGN_BYTES = 64;

// ---------------------- Cross-platform aligned allocation helpers ----------------------
static inline int *allocate_temp_buffer(size_t n)
{
    size_t bytes = n * sizeof(int);
#if defined(_WIN32)
    void *ptr = _aligned_malloc(bytes, ALIGN_BYTES);
    return reinterpret_cast<int *>(ptr);
#else
    void *ptr = nullptr;
    if (posix_memalign(&ptr, ALIGN_BYTES, bytes) == 0)
        return reinterpret_cast<int *>(ptr);
    return reinterpret_cast<int *>(malloc(bytes));
#endif
}

static inline void free_temp_buffer(int *ptr)
{
    if (!ptr)
        return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// ---------------------- SIMD memcpy (AVX2) ----------------------
static inline void simd_memcpy(void *dst_v, const void *src_v, size_t bytes)
{
    unsigned char *dst = reinterpret_cast<unsigned char *>(dst_v);
    const unsigned char *src = reinterpret_cast<const unsigned char *>(src_v);

    size_t i = 0;
    size_t simd_steps = bytes / SIMD_BYTES;
    for (size_t s = 0; s < simd_steps; ++s)
    {
        __m256i t = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), t);
        i += SIMD_BYTES;
    }

    size_t remaining = bytes % SIMD_BYTES;
    for (size_t j = 0; j < remaining; ++j)
        dst[i + j] = src[i + j];
}

static inline void simd_copy_ints(int *dst, const int *src, size_t count)
{
    if (count == 0)
        return;
    simd_memcpy(dst, src, count * sizeof(int));
}

// ---------------------- Insertion sort for small ranges ----------------------
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

// ---------------------- Scalar merge (fallback) ----------------------
static inline void merge_with_temp_scalar(int arr[], int l, int m, int r, int temp[])
{
    int i = l;
    int j = m + 1;
    int k = l;
    while (i <= m && j <= r)
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }
    while (i <= m)
        temp[k++] = arr[i++];
    while (j <= r)
        temp[k++] = arr[j++];

    simd_copy_ints(arr + l, temp + l, (size_t)(r - l + 1));
}

// ---------------------- Merge Path helpers ----------------------
// Returns pair (i_from_A, j_from_B) counts for a given k (k elements taken overall)
static inline std::pair<int, int> merge_path_find_split(int arr[], int l, int m, int r, int k)
{
    int n1 = m - l + 1;
    int n2 = r - m; // number of elements in B (m+1 .. r)

    if (k <= 0)
        return {0, 0};
    if (k >= n1 + n2)
        return {n1, n2};

    int low = std::max(0, k - n2);
    int high = std::min(n1, k);

    while (low < high)
    {
        int mid = (low + high) >> 1;
        int i = mid;
        int j = k - mid;

        int A_i = (i < n1) ? arr[l + i] : INT_MAX;
        int A_im1 = (i - 1 >= 0) ? arr[l + i - 1] : INT_MIN;

        int B_j = (j < n2) ? arr[m + 1 + j] : INT_MAX;
        int B_jm1 = (j - 1 >= 0) ? arr[m + 1 + (j - 1)] : INT_MIN;

        // Use <= to ensure correct boundary partitioning (stable)
        if (A_i <= B_jm1)
            low = mid + 1;
        else
            high = mid;
    }
    int i = low;
    int j = k - low;
    return {i, j};
}

// ---------------------- Parallel merge using Merge Path (task-based, no nested parallel) ----------------------
void parallel_merge(int arr[], int l, int m, int r, int temp[])
{
    int n1 = m - l + 1;
    int n2 = r - m;
    int total = n1 + n2;

    if (total <= MERGE_PATH_MIN_CHUNK)
    {
        merge_with_temp_scalar(arr, l, m, r, temp);
        return;
    }

    int max_threads = omp_get_max_threads();

    int parts = std::min(max_threads, std::max(1, total / MERGE_PATH_MIN_CHUNK));
    if (parts <= 1)
    {
        merge_with_temp_scalar(arr, l, m, r, temp);
        return;
    }

    int chunk = (total + parts - 1) / parts;

    // Create tasks (executed inside the top-level parallel team) so tasks will
    // write to disjoint output ranges in temp[l..r].
    for (int p = 0; p < parts; ++p)
    {
        int start_k = p * chunk;
        int end_k = std::min(total, start_k + chunk);
        if (start_k >= end_k)
            continue;

#pragma omp task firstprivate(start_k, end_k)
        {
            auto start_coord = merge_path_find_split(arr, l, m, r, start_k);
            auto end_coord = merge_path_find_split(arr, l, m, r, end_k);

            int i_start = start_coord.first;
            int j_start = start_coord.second;
            int i_end = end_coord.first;
            int j_end = end_coord.second;

            int a_start = l + i_start;
            int a_end = l + i_end;
            int b_start = m + 1 + j_start;
            int b_end = m + 1 + j_end;

            int ik = l + start_k;
            int ia = a_start;
            int ib = b_start;

            while (ia < a_end && ib < b_end)
            {
                if (arr[ia] <= arr[ib])
                    temp[ik++] = arr[ia++];
                else
                    temp[ik++] = arr[ib++];
            }
            while (ia < a_end)
                temp[ik++] = arr[ia++];
            while (ib < b_end)
                temp[ik++] = arr[ib++];
        } // end task
    } // end for parts

    // Wait for all merge tasks to finish
#pragma omp taskwait

    // Copy back from temp[l..r] to arr[l..r] using tasks (same partitioning)
    for (int p = 0; p < parts; ++p)
    {
        int start_k = p * chunk;
        int end_k = std::min(total, start_k + chunk);
        if (start_k >= end_k)
            continue;

#pragma omp task firstprivate(start_k, end_k)
        {
            int copy_l = l + start_k;
            int copy_len = end_k - start_k;
            simd_copy_ints(arr + copy_l, temp + copy_l, (size_t)copy_len);
        } // end task
    }

#pragma omp taskwait
}

// ---------------------- Recursive parallel merge sort ----------------------
void merge_sort_rec_parallel(int arr[], int l, int r, int temp[])
{
    int len = r - l + 1;
    if (len <= INSERTION_SORT_THRESHOLD)
    {
        insertion_sort(arr, l, r);
        return;
    }

    int m = l + (r - l) / 2;

    // Decide whether to spawn tasks based on len
    if (len >= TASK_THRESHOLD)
    {
#pragma omp task shared(arr, temp) firstprivate(l, m)
        merge_sort_rec_parallel(arr, l, m, temp);

#pragma omp task shared(arr, temp) firstprivate(m, r)
        merge_sort_rec_parallel(arr, m + 1, r, temp);

#pragma omp taskwait
    }
    else
    {
        merge_sort_rec_parallel(arr, l, m, temp);
        merge_sort_rec_parallel(arr, m + 1, r, temp);
    }

    // Use a task (single) to perform merge — parallel_merge itself will spawn tasks
#pragma omp task shared(arr, temp) firstprivate(l, m, r)
    {
        parallel_merge(arr, l, m, r, temp);
    }
#pragma omp taskwait
}

// ---------------------- Entry point ----------------------
void merge_sort_opt_parallel(int arr[], int n)
{
    if (n <= 0)
        return;

    int *temp = allocate_temp_buffer((size_t)n);
    if (!temp)
    {
        temp = reinterpret_cast<int *>(malloc((size_t)n * sizeof(int)));
        if (!temp)
            return;
    }

    // Postavi broj threadova na maksimalan
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);

    // Top-level parallel region + single task to start recursion.
#pragma omp parallel
    {
#pragma omp single nowait
        merge_sort_rec_parallel(arr, 0, n - 1, temp);
    }

    free_temp_buffer(temp);
}

// ---------------------- Wrapper for test harness ----------------------
void merge_sort_wrapper(std::vector<int> &vec)
{
    merge_sort_opt_parallel(vec.data(), (int)vec.size());
}

int main(int argc, char *argv[])
{
    return run_sort("merge_sort", "parallel_opt", merge_sort_wrapper, argc, argv);
}