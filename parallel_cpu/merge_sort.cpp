// parallel_cpu/merge_sort.cpp
// Parallel + Merge-Path pragmatic version of optimized merge sort
// Uses OpenMP tasks + Merge Path for parallel merging
// Compile with -O3 -march=native and link OpenMP

#include <vector>
#include <algorithm>
#include <cstring>  // memcpy
#include <cassert>
#include <climits>
#include <omp.h>
#include <main_template.hpp>

const int INSERTION_SORT_THRESHOLD = 16; // empirical
const int TASK_THRESHOLD = 1 << 16;      // ~65536 - adjust empirically
const int MERGE_PATH_MIN_CHUNK = 1024;   // min chunk for merging per thread

// Simple insertion sort for small ranges
static inline void insertion_sort(int arr[], int l, int r) {
    for (int i = l + 1; i <= r; ++i) {
        int key = arr[i];
        int j = i - 1;
        while (j >= l && arr[j] > key) {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

// scalar merge into temp
static inline void merge_with_temp_scalar(int arr[], int l, int m, int r, int temp[]) {
    int i = l;
    int j = m + 1;
    int k = l;
    while (i <= m && j <= r) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];
    memcpy(arr + l, temp + l, (size_t)(r - l + 1) * sizeof(int));
}

// ---------- Merge Path helpers ----------
// For a requested diagonal k (0..total), find split i in [max(0,k-n2) .. min(n1,k)]
// such that left contributes i items, right contributes k-i items for the prefix.
static inline std::pair<int,int> merge_path_find_split(int arr[], int l, int m, int r, int k) {
    int n1 = m - l + 1;
    int n2 = r - (m + 1) + 1; // = r - m
    int low = std::max(0, k - n2);
    int high = std::min(n1, k);

    while (low < high) {
        int mid = (low + high) >> 1;
        // compute indices in arrays
        int a_idx = l + mid;
        int b_idx = m + 1 + (k - mid) - 1; // could be < m+1

        int a_val = (mid >= n1) ? INT_MAX : arr[a_idx]; // if mid == n1, a_val = +inf
        int b_val = ((k - mid - 1) < 0) ? INT_MIN : arr[b_idx]; // if no b item, -inf

        if (a_val <= b_val) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    int i = low;
    int j = k - i;
    return {i, j};
}

// Parallel merge using Merge Path
void parallel_merge(int arr[], int l, int m, int r, int temp[]) {
    int n1 = m - l + 1;
    int n2 = r - (m + 1) + 1; // = r-m
    int total = n1 + n2;
    if (total <= MERGE_PATH_MIN_CHUNK) {
        merge_with_temp_scalar(arr, l, m, r, temp);
        return;
    }

    int max_threads = omp_get_max_threads();
    int parts = std::min(max_threads, std::max(1, total / MERGE_PATH_MIN_CHUNK));
    if (parts <= 1) {
        merge_with_temp_scalar(arr, l, m, r, temp);
        return;
    }

    int chunk = (total + parts - 1) / parts;

    // parallel workers (each thread handles its portion)
    #pragma omp parallel num_threads(parts)
    {
        int tid = omp_get_thread_num();
        int start_k = tid * chunk;
        int end_k = std::min(total, start_k + chunk);

        // if no work for this thread, just skip (do NOT return/continue from outside loop)
        if (start_k >= end_k) {
            // nothing for this thread
        } else {
            // find start/end coordinates (numbers taken from left and right)
            auto start_coord = merge_path_find_split(arr, l, m, r, start_k);
            auto end_coord   = merge_path_find_split(arr, l, m, r, end_k);

            int a_start = l + start_coord.first;
            int b_start = m + 1 + start_coord.second;
            int a_end   = l + end_coord.first - 1;
            int b_end   = m + end_coord.second;

            int out_pos = l + start_k;

            int ia = a_start;
            int ib = b_start;
            int ik = out_pos;

            // local merge
            while (ia <= a_end && ib <= b_end) {
                if (arr[ia] <= arr[ib]) temp[ik++] = arr[ia++];
                else temp[ik++] = arr[ib++];
            }
            while (ia <= a_end) temp[ik++] = arr[ia++];
            while (ib <= b_end) temp[ik++] = arr[ib++];
        }
    } // omp parallel

    // copy back in parallel
    #pragma omp parallel for schedule(static)
    for (int idx = l; idx <= r; ++idx) {
        arr[idx] = temp[idx];
    }
}

// ---------- Recursive sort with OpenMP tasks ----------
void merge_sort_rec_parallel(int arr[], int l, int r, int temp[]) {
    int len = r - l + 1;
    if (len <= INSERTION_SORT_THRESHOLD) {
        insertion_sort(arr, l, r);
        return;
    }
    int m = l + (r - l) / 2;

    if (len >= TASK_THRESHOLD) {
        #pragma omp task shared(arr, temp) firstprivate(l, m)
        merge_sort_rec_parallel(arr, l, m, temp);

        #pragma omp task shared(arr, temp) firstprivate(m, r)
        merge_sort_rec_parallel(arr, m + 1, r, temp);

        #pragma omp taskwait
    } else {
        merge_sort_rec_parallel(arr, l, m, temp);
        merge_sort_rec_parallel(arr, m + 1, r, temp);
    }

    // skip merge if already ordered
    if (arr[m] <= arr[m + 1]) return;

    // parallel merge using Merge Path
    parallel_merge(arr, l, m, r, temp);
}

void merge_sort_opt_parallel(int arr[], int n) {
    int *temp = new int[n];
    #pragma omp parallel
    {
        #pragma omp single nowait
        merge_sort_rec_parallel(arr, 0, n - 1, temp);
    }
    delete[] temp;
}

// wrapper expected by test harness
void merge_sort_wrapper(std::vector<int>& vec) {
    merge_sort_opt_parallel(vec.data(), (int)vec.size());
}

// main uses run_sort from main_template.hpp
int main(int argc, char* argv[]) {
    return run_sort("merge_sort", "parallel_opt", merge_sort_wrapper, argc, argv);
}