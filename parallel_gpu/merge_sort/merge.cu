// merge.cu
// Final GPU mergesort benchmark (clean CSV outputs, hybrid merge, grid harness).
// Build with NVCC:
// nvcc -O3 -arch=sm_86 -lineinfo -Xptxas -v merge.cu -o merge.exe

#include "../include/merge_radix_api.hpp"
#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <climits>
#include <cstdint>
#include <random>
#include <chrono>
#include <cmath>
#include <ctime>
#include <limits>
#include <cerrno>
#include <string>

#ifdef _WIN32
#include <direct.h> // _mkdir
#else
#include <sys/stat.h> // mkdir
#include <sys/types.h>
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Output directory for all CSVs (string); "/" works on Windows and POSIX
const std::string OUTDIR = "../merge_data";

// Small cross-platform helper to create directory if missing
static bool ensure_outdir_exists(const std::string &dir) {
#ifdef _WIN32
    int r = _mkdir(dir.c_str());
    if (r == 0) return true;
    if (errno == EEXIST) return true;
    std::cerr << "mkdir failed (" << dir << "): errno=" << errno << std::endl;
    return false;
#else
    int r = mkdir(dir.c_str(), 0755);
    if (r == 0) return true;
    if (errno == EEXIST) return true;
    std::cerr << "mkdir failed (" << dir << "): errno=" << errno << std::endl;
    return false;
#endif
}

// helper to join OUTDIR + filename
static std::string outpath(const std::string &filename) {
    if (OUTDIR.empty()) return filename;
    if (OUTDIR.back() == '/' || OUTDIR.back() == '\\') return OUTDIR + filename;
    return OUTDIR + "/" + filename;
}

// Timestamp string for filenames: YYYYMMDD_HHMMSS
static std::string timestamp_string() {
    std::time_t t = std::time(nullptr);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_buf);
    return std::string(buf);
}

// Tunables (final deliverable defaults)
static const int DEFAULT_THREADS_MERGE = 256;
static const int DEFAULT_E = 8;
static const int MIN_BLOCK_WIDTH = 16384;     // use block_shared only when width >= this (elements)
static const int PAIR_THRESHOLD = 8192;       // alternative threshold (elements)
static const size_t SHARED_LIMIT = 48 * 1024; // 48 KB shared mem limit per block

#ifndef BLOCK_ELEMS_PER_THREAD
#define BLOCK_ELEMS_PER_THREAD 8
#endif

// -------------------- KERNELS --------------------
// Bitonic block-sort (shared mem, small padding)
__global__ void bitonic_sort_shared_kernel(int* d_data, int n) {
    __shared__ int shared_mem[512 + 16];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    int val = INT_MAX;
    if (idx < n) val = __ldg(reinterpret_cast<const int*>(d_data + idx));

    int pad = tid >> 5;
    int sidx = tid + pad;
    shared_mem[sidx] = val;
    __syncthreads();

    for (int k = 2; k <= 512; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                int s_ixj = ixj + (ixj >> 5);
                if ((tid & k) == 0) {
                    if (shared_mem[sidx] > shared_mem[s_ixj]) {
                        int tmp = shared_mem[sidx];
                        shared_mem[sidx] = shared_mem[s_ixj];
                        shared_mem[s_ixj] = tmp;
                    }
                } else {
                    if (shared_mem[sidx] < shared_mem[s_ixj]) {
                        int tmp = shared_mem[sidx];
                        shared_mem[sidx] = shared_mem[s_ixj];
                        shared_mem[s_ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (idx < n) d_data[idx] = shared_mem[sidx];
}

// Per-element parallel merge (merge-path style)
__global__ void parallel_merge_kernel(const int* __restrict__ d_in, int* __restrict__ d_out, int n, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int pair_idx = idx / (2 * width);
    int local_idx = idx % (2 * width);
    int left_start = pair_idx * 2 * width;
    int mid = min(left_start + width, n);
    int right_end = min(left_start + 2 * width, n);

    if (mid >= n) {
        d_out[idx] = d_in[idx];
        return;
    }

    int len_a = mid - left_start;
    int len_b = right_end - mid;
    const int* A = d_in + left_start;
    const int* B = d_in + mid;

    int k = local_idx;
    int low = max(0, k - len_b);
    int high = min(k, len_a);

    while (low < high) {
        int i = (low + high) >> 1;
        int j = k - 1 - i;
        int Ai = __ldg(A + i);
        int Bj = __ldg(B + j);
        if (Ai < Bj) low = i + 1;
        else high = i;
    }

    int i = low;
    int j = k - i;
    int Ai = (i < len_a) ? __ldg(A + i) : INT_MAX;
    int Bj = (j < len_b) ? __ldg(B + j) : INT_MAX;
    d_out[idx] = (i < len_a && (j >= len_b || Ai <= Bj)) ? Ai : Bj;
}

// Coarsened per-thread merge
__launch_bounds__(512)
__global__ void parallel_merge_coarsened_opt(const int* __restrict__ d_in, int* __restrict__ d_out,
                                             int n, int width, int E) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    long long k_base = (long long)gid * (long long)E;
    long long twow = 2LL * width;
    if (k_base >= n) return;

    long long pair_idx = k_base / twow;
    int local_idx = (int)(k_base % twow);

    for (int e = 0; e < E; ++e) {
        long long k = k_base + e;
        if (k >= n) break;

        int left_start = (int)(pair_idx * twow);
        int mid = left_start + width;
        if (mid > n) mid = n;
        int right_end = left_start + 2 * width;
        if (right_end > n) right_end = n;

        if (mid >= n) {
            d_out[k] = __ldg(d_in + k);
        } else {
            int len_a = mid - left_start;
            int len_b = right_end - mid;
            const int* A = d_in + left_start;
            const int* B = d_in + mid;

            int low = local_idx - len_b;
            if (low < 0) low = 0;
            int high = local_idx;
            if (high > len_a) high = len_a;

            while (low < high) {
                int i = (low + high) >> 1;
                int j = local_idx - 1 - i;
                int Ai = __ldg(A + i);
                int Bj = __ldg(B + j);
                if (Ai < Bj) low = i + 1;
                else high = i;
            }
            int i = low;
            int j = local_idx - i;
            int Ai = (i < len_a) ? __ldg(A + i) : INT_MAX;
            int Bj = (j < len_b) ? __ldg(B + j) : INT_MAX;
            int val = (i < len_a && (j >= len_b || Ai <= Bj)) ? Ai : Bj;

            d_out[k] = val;
        }

        local_idx++;
        if (local_idx >= twow) {
            local_idx = 0;
            pair_idx++;
        }
    }
}

// merge_path (device)
__device__ __forceinline__ int device_merge_path(const int* A, int len_a, const int* B, int len_b, int k) {
    int low = max(0, k - len_b);
    int high = min(k, len_a);
    while (low < high) {
        int mid = (low + high) >> 1;
        int j = k - 1 - mid;
        int Aval = __ldg(A + mid);
        int Bval = __ldg(B + j);
        if (Aval < Bval) low = mid + 1;
        else high = mid;
    }
    return low;
}

// block-level shared-memory merge (tile-based)
__device__ __forceinline__ int merge_path_shared(const int* A, int len_a, const int* B, int len_b, int k) {
    int low = max(0, k - len_b);
    int high = min(k, len_a);
    while (low < high) {
        int mid = (low + high) >> 1;
        int j = k - 1 - mid;
        int Aval = A[mid];
        int Bval = B[j];
        if (Aval < Bval) low = mid + 1;
        else high = mid;
    }
    return low;
}

__global__ void block_merge_shared(const int* __restrict__ d_in, int* __restrict__ d_out,
                                   int n, int width, int elems_per_thread) {
    int pair_idx = blockIdx.x;
    long long left_start_ll = (long long)pair_idx * 2LL * width;
    if (left_start_ll >= n) return;
    int left_start = (int)left_start_ll;

    int mid = min(left_start + width, n);
    int right_end = min(left_start + 2 * width, n);
    if (mid >= n) return;

    const int* A = d_in + left_start;
    const int* B = d_in + mid;
    int len_a = mid - left_start;
    int len_b = right_end - mid;
    int total = len_a + len_b;

    int threads = blockDim.x;
    int tile_elems = threads * elems_per_thread;

    extern __shared__ int sdata[]; // will hold a_count + b_count items for tile

    int num_tiles = (total + tile_elems - 1) / tile_elems;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int k_start = tile * tile_elems;
        int k_end = min(total, k_start + tile_elems);

        int a_start = device_merge_path(A, len_a, B, len_b, k_start);
        int a_end   = device_merge_path(A, len_a, B, len_b, k_end);
        int a_count = a_end - a_start;
        int b_start = k_start - a_start;
        int b_count = (k_end - k_start) - a_count;

        int total_load = a_count + b_count;
        for (int x = threadIdx.x; x < total_load; x += threads) {
            if (x < a_count) {
                sdata[x] = __ldg(A + a_start + x);
            } else {
                int bx = x - a_count;
                sdata[x] = __ldg(B + b_start + bx);
            }
        }
        __syncthreads();

        int* sA = sdata;
        int* sB = sdata + a_count;

        int tid = threadIdx.x;
        int local_k_base = k_start + tid * elems_per_thread;

        for (int e = 0; e < elems_per_thread; ++e) {
            int k = local_k_base + e;
            if (k >= k_end) break;

            int k_local = k - k_start;
            int i = merge_path_shared(sA, a_count, sB, b_count, k_local);
            int j = k_local - i;
            int Ai = (i < a_count) ? sA[i] : INT_MAX;
            int Bj = (j < b_count) ? sB[j] : INT_MAX;
            int v = (i < a_count && (j >= b_count || Ai <= Bj)) ? Ai : Bj;

            d_out[left_start + k] = v;
        }
        __syncthreads();
    }
}

// -------------------- TIMING / UTILS --------------------

struct MergeTimings {
    float h2d_ms = 0.0f;
    float bitonic_ms = 0.0f;
    float merge_ms = 0.0f; // total merge time
    std::vector<float> merge_pass_times; // per-pass times
    float d2h_ms = 0.0f;
    float total_gpu_ms = 0.0f; // bitonic + merge
    bool verified = false;
};

static double mean_d(const std::vector<float>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (float x : v) s += x;
    return s / v.size();
}
static double stddev_d(const std::vector<float>& v) {
    if (v.size() <= 1) return 0.0;
    double m = mean_d(v);
    double s = 0.0;
    for (float x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
}

// merge_sort_gpu_timed: runs kernels and captures per-pass times (CUDA events)
MergeTimings merge_sort_gpu_timed(std::vector<int>& host_arr, int merge_algorithm = 2, int E = DEFAULT_E, int threads_merge = DEFAULT_THREADS_MERGE) {
    MergeTimings t;
    int n = (int)host_arr.size();
    if (n <= 1) return t;

    int *d_ptr1 = nullptr, *d_ptr2 = nullptr;
    gpuErrchk(cudaMalloc(&d_ptr1, (size_t)n * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_ptr2, (size_t)n * sizeof(int)));

    cudaEvent_t e_h2d_s, e_h2d_e, e_bitonic_s, e_bitonic_e, e_merge_s, e_merge_e, e_d2h_s, e_d2h_e;
    cudaEventCreate(&e_h2d_s); cudaEventCreate(&e_h2d_e);
    cudaEventCreate(&e_bitonic_s); cudaEventCreate(&e_bitonic_e);
    cudaEventCreate(&e_merge_s); cudaEventCreate(&e_merge_e);
    cudaEventCreate(&e_d2h_s); cudaEventCreate(&e_d2h_e);

    // H2D
    cudaEventRecord(e_h2d_s);
    gpuErrchk(cudaMemcpy(d_ptr1, host_arr.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(e_h2d_e);
    cudaEventSynchronize(e_h2d_e);
    cudaEventElapsedTime(&t.h2d_ms, e_h2d_s, e_h2d_e);

    // Bitonic
    const int threadsPerBlock = 512;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventRecord(e_bitonic_s);
    bitonic_sort_shared_kernel<<<blocks, threadsPerBlock>>>(d_ptr1, n);
    gpuErrchk(cudaGetLastError());
    cudaEventRecord(e_bitonic_e);
    cudaEventSynchronize(e_bitonic_e);
    cudaEventElapsedTime(&t.bitonic_ms, e_bitonic_s, e_bitonic_e);

    // Merge passes
    int *d_in = d_ptr1, *d_out = d_ptr2;
    float merge_sum = 0.0f;
    t.merge_pass_times.clear();

    for (int width = 512; width < n; width *= 2) {
        cudaEventRecord(e_merge_s);

        if (merge_algorithm == 0) {
            int threads_local = threads_merge;
            long blocks_merge = (n + threads_local - 1) / threads_local;
            parallel_merge_kernel<<<(int)blocks_merge, threads_local>>>(d_in, d_out, n, width);
            gpuErrchk(cudaGetLastError());
        } else if (merge_algorithm == 1) {
            long long elems_per_block = (long long)threads_merge * E;
            long long blocks_merge = (n + elems_per_block - 1) / elems_per_block;
            parallel_merge_coarsened_opt<<<(int)blocks_merge, threads_merge>>>(d_in, d_out, n, width, E);
            gpuErrchk(cudaGetLastError());
        } else {
            int pair_len = 2 * width;
            if (width < MIN_BLOCK_WIDTH || pair_len < PAIR_THRESHOLD) {
                int threads_local = threads_merge;
                long blocks_merge = (n + threads_local - 1) / threads_local;
                parallel_merge_kernel<<<(int)blocks_merge, threads_local>>>(d_in, d_out, n, width);
                gpuErrchk(cudaGetLastError());
            } else {
                int threads = DEFAULT_THREADS_MERGE;
                int elems_per_thread = BLOCK_ELEMS_PER_THREAD;
                if (pair_len >= 65536) elems_per_thread = 16;
                else if (pair_len >= 32768) elems_per_thread = 8;
                else elems_per_thread = BLOCK_ELEMS_PER_THREAD;

                int tile_elems = threads * elems_per_thread;
                size_t shared_bytes = (size_t)tile_elems * sizeof(int);
                while (shared_bytes > SHARED_LIMIT && elems_per_thread > 1) {
                    elems_per_thread /= 2;
                    tile_elems = threads * elems_per_thread;
                    shared_bytes = (size_t)tile_elems * sizeof(int);
                }

                int num_pairs = (n + (2 * width) - 1) / (2 * width);
                int blocks_merge = num_pairs;
                block_merge_shared<<<blocks_merge, threads, shared_bytes>>>(d_in, d_out, n, width, elems_per_thread);
                gpuErrchk(cudaGetLastError());
            }
        }

        cudaEventRecord(e_merge_e);
        cudaEventSynchronize(e_merge_e);
        float pass_ms = 0.0f;
        cudaEventElapsedTime(&pass_ms, e_merge_s, e_merge_e);
        t.merge_pass_times.push_back(pass_ms);
        merge_sum += pass_ms;

        std::swap(d_in, d_out);
    }

    t.merge_ms = merge_sum;
    t.total_gpu_ms = t.bitonic_ms + t.merge_ms;

    // D2H
    std::vector<int> result(n);
    cudaEventRecord(e_d2h_s);
    gpuErrchk(cudaMemcpy(result.data(), d_in, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventRecord(e_d2h_e);
    cudaEventSynchronize(e_d2h_e);
    cudaEventElapsedTime(&t.d2h_ms, e_d2h_s, e_d2h_e);

    t.verified = std::is_sorted(result.begin(), result.end());
    host_arr = std::move(result);

    // cleanup
    cudaEventDestroy(e_h2d_s); cudaEventDestroy(e_h2d_e);
    cudaEventDestroy(e_bitonic_s); cudaEventDestroy(e_bitonic_e);
    cudaEventDestroy(e_merge_s); cudaEventDestroy(e_merge_e);
    cudaEventDestroy(e_d2h_s); cudaEventDestroy(e_d2h_e);
    cudaFree(d_ptr1); cudaFree(d_ptr2);

    return t;
}

// Adapter: IZVAN #ifndef BUILD_MERGE_LIB
void merge_sort_gpu(const std::vector<int>& h_in,
                    std::vector<int>& h_out,
                    Timings &timings,
                    const DeviceInfo &/*dinfo*/,
                    int threads,
                    int /*blocks_override*/) {
    int alg = 2;                 // BlockMerge
    int E   = DEFAULT_E;
    int T   = (threads > 0) ? threads : DEFAULT_THREADS_MERGE;

    std::vector<int> arr = h_in;
    MergeTimings mt = merge_sort_gpu_timed(arr, alg, E, T);

    timings.h2d_ms = mt.h2d_ms;
    timings.total_kernel_ms = mt.bitonic_ms + mt.merge_ms;
    timings.d2h_ms = mt.d2h_ms;
    timings.total_ms = timings.h2d_ms + timings.total_kernel_ms + timings.d2h_ms;
    timings.kernel_invocations = 1 + (int)mt.merge_pass_times.size();

    h_out = std::move(arr);
}


#ifndef BUILD_MERGE_LIB

// -------------------- GRID BENCHMARK (Unified per-algorithm CSV with name + timestamp) --------------------
void run_grid_bench(int n,
                    const std::vector<int>& algs,
                    const std::vector<int>& Es,
                    const std::vector<int>& threads_list,
                    int repetitions) {

    // Ensure out dir exists
    if (!ensure_outdir_exists(OUTDIR)) {
        std::cerr << "Unable to ensure output directory exists: " << OUTDIR << std::endl;
        return;
    }

    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::string devname = prop.name;
    std::string ts = timestamp_string();

    // Prepare per-alg CSV files named with sort + timestamp
    std::ofstream perElementCsv(outpath("merge_PerElement_" + ts + ".csv"));
    std::ofstream coarsenedCsv(outpath("merge_Coarsened_" + ts + ".csv"));
    std::ofstream blockMergeCsv(outpath("merge_BlockMerge_" + ts + ".csv"));
    if (!perElementCsv || !coarsenedCsv || !blockMergeCsv) {
        std::cerr << "Cannot open per-alg CSV outputs\n";
        return;
    }

    // Header (exact structure requested)
    auto write_header = [](std::ofstream &f){
        f << "timestamp,device,ArraySize,AlgName,AlgId,E,ThreadsMerge,Repetition,H2D_ms,Bitonic_ms,Merge_ms,MergePassCount,MergePassAvg_ms,MergePassStddev_ms,D2H_ms,TotalGPU_ms,Verified\n";
        f << std::fixed << std::setprecision(3);
    };
    write_header(perElementCsv);
    write_header(coarsenedCsv);
    write_header(blockMergeCsv);

    // Best summary CSV (named with timestamp as well)
    std::ofstream bestCsv(outpath("merge_best_" + ts + ".csv"));
    if (!bestCsv) {
        std::cerr << "Cannot open best summary CSV\n";
        return;
    }
    bestCsv << "timestamp,device,ArraySize,BestAlgName,BestAlgId,BestE,BestThreads,BestTotalGPU_ms\n";
    bestCsv << std::fixed << std::setprecision(3);

    double best_avg_total = std::numeric_limits<double>::infinity();
    int best_alg = -1, best_E = -1, best_threads = -1;

    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<int> dist(0, 1000000);

    auto write_row = [&](std::ofstream &csv, const std::string &algname, int algId, int E, int threads, int rep, const MergeTimings &t) {
        // timestamp human-readable for row (YYYY-MM-DD HH:MM:SS)
        std::time_t ts_now = std::time(nullptr);
        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &ts_now);
#else
        localtime_r(&ts_now, &tm_buf);
#endif
        char timestr[64];
        std::strftime(timestr, sizeof(timestr), "%Y-%m-%d %H:%M:%S", &tm_buf);

        int pass_count = (int)t.merge_pass_times.size();
        double pass_avg = mean_d(t.merge_pass_times);
        double pass_std = stddev_d(t.merge_pass_times);
        double total_gpu_ms = t.total_gpu_ms;

        csv << "\"" << timestr << "\"," 
            << "\"" << devname << "\"," << n << ","
            << algname << "," << algId << "," << E << "," << threads << "," << rep << ","
            << t.h2d_ms << "," << t.bitonic_ms << "," << t.merge_ms << ","
            << pass_count << "," << pass_avg << "," << pass_std << ","
            << t.d2h_ms << "," << total_gpu_ms << "," << (t.verified?1:0) << "\n";
    };

    for (int alg : algs) {
        for (int E : Es) {
            for (int threads : threads_list) {
                std::vector<double> totals;
                int verified_ok = 0;

                for (int rep = 0; rep < repetitions; ++rep) {
                    // generate input
                    std::vector<int> arr(n);
                    for (int i = 0; i < n; ++i) arr[i] = dist(rng);

                    // run timed (CUDA events inside)
                    MergeTimings t = merge_sort_gpu_timed(arr, alg, E, threads);

                    // write row to appropriate per-alg CSV
                    std::string algname = (alg==0) ? "PerElement" : (alg==1) ? "Coarsened" : "BlockMerge";
                    if (alg == 0) write_row(perElementCsv, algname, alg, E, threads, rep, t);
                    else if (alg == 1) write_row(coarsenedCsv, algname, alg, E, threads, rep, t);
                    else write_row(blockMergeCsv, algname, alg, E, threads, rep, t);

                    totals.push_back(t.total_gpu_ms);
                    if (t.verified) ++verified_ok;
                } // repetitions

                double avg_total = 0.0;
                if (!totals.empty()) {
                    double s=0; for (double x:totals) s+=x; avg_total = s / totals.size();
                }

                std::cout << "ALG=" << alg << " E=" << E << " T=" << threads
                          << " reps=" << repetitions << " avg_total_ms=" << avg_total
                          << " verified=" << verified_ok << "/" << repetitions << std::endl;

                if (avg_total < best_avg_total) {
                    best_avg_total = avg_total;
                    best_alg = alg;
                    best_E = E;
                    best_threads = threads;
                }
            }
        }
    }

    // write best summary row
    {
        std::time_t ts_now = std::time(nullptr);
        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &ts_now);
#else
        localtime_r(&ts_now, &tm_buf);
#endif
        char timestr[64];
        std::strftime(timestr, sizeof(timestr), "%Y-%m-%d %H:%M:%S", &tm_buf);
        std::string best_name = (best_alg==0) ? "PerElement" : (best_alg==1) ? "Coarsened" : "BlockMerge";
        bestCsv << "\"" << timestr << "\"," << "\"" << devname << "\"," << n << ","
                << best_name << "," << best_alg << "," << best_E << "," << best_threads << "," << best_avg_total << "\n";
    }

    perElementCsv.close();
    coarsenedCsv.close();
    blockMergeCsv.close();
    bestCsv.close();

    std::cout << "Per-alg CSVs written: "
              << outpath("merge_PerElement_" + ts + ".csv") << ", "
              << outpath("merge_Coarsened_" + ts + ".csv") << ", "
              << outpath("merge_BlockMerge_" + ts + ".csv") << "\n";
    std::cout << "Best summary CSV written: " << outpath("merge_best_" + ts + ".csv") << std::endl;
}

// -------------------- MAIN --------------------
int main(int argc, char* argv[]) {
    // Ensure outdir exists
    if (!ensure_outdir_exists(OUTDIR)) {
        std::cerr << "Cannot create output directory: " << OUTDIR << std::endl;
        return 1;
    }

    cudaFree(0); // warm-up

    bool do_grid = false;
    for (int i=1;i<argc;++i) if (std::string(argv[i])=="--grid") do_grid = true;

    // Always run grid-style benchmark (use --grid to print banner)
    if (do_grid) {
        std::cout << "Running final grid benchmark...\n";
    }

    int n = 134217728; // final runs
    std::vector<int> algs = {0,1,2};
    std::vector<int> Es = {2,4,8};
    std::vector<int> threads_list = {128,256,512};
    int repetitions = 3;

    run_grid_bench(n, algs, Es, threads_list, repetitions);

    std::cout << "All scenarios finished. CSVs are in: " << OUTDIR << std::endl;
    return 0;
}

#endif // BUILD_MERGE_LIB