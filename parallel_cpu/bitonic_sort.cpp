// bitonic_sort_parallel_omp_avx2.cpp
#include <algorithm>
#include <vector>
#include <iostream>
#include "main_template.hpp"
#include <cstring>
#include <cstdlib>
#include <omp.h>

#if defined(__AVX2__)
    #include <immintrin.h>
    #define HAVE_AVX2 1
#else
    #define HAVE_AVX2 0
#endif

using namespace std;

// -------- Helper: Set/OpenMP threads --------
static void set_num_threads_from_env_or_arg(int requested_threads = 0) {
    if (requested_threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(requested_threads);
    } else {
        const char* env = getenv("OMP_NUM_THREADS");
        if (env) {
            int t = atoi(env);
            if (t > 0) {
                omp_set_dynamic(0);
                omp_set_num_threads(t);
            }
        }
    }
    #pragma omp parallel
    {
        #pragma omp single
        {
            cerr << "[bitonic] Threads = " << omp_get_max_threads()
                 << " (OMP_NUM_THREADS=" << (getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "unset") << ")\n";
        }
    }
}

// -------- InsertionSort --------
void insertionSort(int* a, int low, int cnt) {
    for (int i = low + 1; i < low + cnt; ++i) {
        int key = a[i];
        int j = i - 1;
        while (j >= low && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

// -------- Scalar compare & swap --------
inline void compAndSwap_scalar(int* a, int i, int j, bool asc) {
    int x = a[i], y = a[j];
    if (asc) {
        if (x > y) { a[i] = y; a[j] = x; }
    } else {
        if (x < y) { a[i] = y; a[j] = x; }
    }
}

// -------- AVX2 compare & swap (8 integera odjednom) --------
#if HAVE_AVX2
inline void compAndSwap_avx2(int* a, int j, int half, bool asc) {
    __m256i v1 = _mm256_loadu_si256((__m256i*)(a + j));
    __m256i v2 = _mm256_loadu_si256((__m256i*)(a + j + half));
    __m256i mn = _mm256_min_epi32(v1, v2);
    __m256i mx = _mm256_max_epi32(v1, v2);

    if (asc) {
        _mm256_storeu_si256((__m256i*)(a + j), mn);
        _mm256_storeu_si256((__m256i*)(a + j + half), mx);
    } else {
        _mm256_storeu_si256((__m256i*)(a + j), mx);
        _mm256_storeu_si256((__m256i*)(a + j + half), mn);
    }
}
#endif

// -------- Parallel merge (bitonic merge) --------
void bitonicMergeIterParallel(int* a, int low, int cnt, bool asc) {
    for (int size = cnt; size > 1; size >>= 1) {
        int half = size >> 1;

        #pragma omp parallel for schedule(static)
        for (int i = low; i < low + cnt; i += size) {
            int j = i;
            int end = i + half;

        #if HAVE_AVX2
            const int VW = 8;
            for (; j + VW - 1 < end; j += VW) {
                compAndSwap_avx2(a, j, half, asc);
            }
        #endif
            for (; j < end; ++j) {
                compAndSwap_scalar(a, j, j + half, asc);
            }
        }
    }
}

// -------- Parallel Bitonic Sort with tasks --------
void bitonicSortParallel(int* a, int low, int cnt, bool asc) {
    const int INSERTION_THRESHOLD = 32;
    if (cnt <= INSERTION_THRESHOLD) {
        insertionSort(a, low, cnt);
        if (!asc) reverse(a + low, a + low + cnt);
        return;
    }

    int k = cnt / 2;

    #pragma omp task shared(a) if (k >= 4096)
    bitonicSortParallel(a, low, k, true);

    #pragma omp task shared(a) if (k >= 4096)
    bitonicSortParallel(a, low + k, k, false);

    #pragma omp taskwait

    bitonicMergeIterParallel(a, low, cnt, asc);
}

// -------- Wrapper used by run_sort(...) --------
void bitonic_sort_parallel_wrapper(vector<int>& vec) {
    set_num_threads_from_env_or_arg(0); // 0 = use env/default
    int* data = vec.data();
    int n = vec.size();

    #pragma omp parallel
    {
        #pragma omp single
        bitonicSortParallel(data, 0, n, true);
    }
}

int main(int argc, char* argv[]) {
    return run_sort("bitonic_sort_omp_avx2", "parallel",
                    bitonic_sort_parallel_wrapper, argc, argv);
}
