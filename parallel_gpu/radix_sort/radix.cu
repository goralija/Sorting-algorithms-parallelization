/*
radix.cu

Multi-bit (4 bits per pass) LSD radix sort on GPU with sweep and CSV export
(same CSV format as the bitonic harness).

Library-free version (GPU-only scans/reduces):
- Per-block 16-bin histograms computed on device.
- Per-bin exclusive scan across blocks computed on device via 3-step scan:
    1) Local exclusive scan per chunk (Blelloch) + write per-chunk sums
    2) Exclusive scan of per-chunk sums (Blelloch)
    3) Add scanned chunk offsets to prefix array
- Per-bin totals computed on device (reduce per-chunk sums + scanned offsets).
- Global bin bases (16 elements) computed on device via small exclusive scan.
- Scatter kernel uses per-warp bin counts and block/bin prefixes (corrected, all threads participate).
- Signed->unsigned mapping on host: key = value ^ 0x80000000u (order-preserving).
- Clamp blocks >= ceil(n/threads).
- Defaults: n=2^27 with min_exp=max_exp=27.
- Sweep CLI and CSV identical to your bitonic harness.

Build:
  nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo -Xptxas -v radix.cu -o radix

Run example (2^27 with sweep over threads):
  ./radix --sweep-threads 256,512,1024 --sweep-blocks -1 --min-exp 27 --max-exp 27 --reps 3 --output ../radix_data
*/

#include "../include/merge_radix_api.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <filesystem>
#include <cstdint>
#include <climits>
#include <limits>
#include <thread>
#include <chrono>
#include <cstring>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << "GPU Error: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
        if (abort) exit(code);
    }
}

static inline int next_pow2_int(int v) { int p=1; while (p<v) p<<=1; return p; }
static inline int next_pow2_up(int v) { int p=1; while (p<v) p<<=1; return p; }


static DeviceInfo queryDevice(int deviceId) {
    cudaDeviceProp prop;
    gpuErrchk(cudaGetDeviceProperties(&prop, deviceId));
    DeviceInfo info;
    info.name = prop.name;
    info.deviceId = deviceId;
    info.multiProcessorCount = prop.multiProcessorCount;
    info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    info.sharedMemPerBlock = prop.sharedMemPerBlock;
    info.major = prop.major;
    info.minor = prop.minor;
    return info;
}

// Thrust baseline s CUDA event timing (H2D/sort/D2H)
static void official_thrust_sort_timed(const std::vector<int>& h_in,
                                       std::vector<int>& h_out,
                                       Timings &timings) {
    const int n = static_cast<int>(h_in.size());
    h_out.resize(n);

    int* d_data = nullptr;
    gpuErrchk(cudaMalloc(&d_data, (size_t)n * sizeof(int)));

    cudaEvent_t ev_h2d_start, ev_h2d_end, ev_kstart, ev_kend, ev_d2h_start, ev_d2h_end;
    gpuErrchk(cudaEventCreate(&ev_h2d_start));
    gpuErrchk(cudaEventCreate(&ev_h2d_end));
    gpuErrchk(cudaEventCreate(&ev_kstart));
    gpuErrchk(cudaEventCreate(&ev_kend));
    gpuErrchk(cudaEventCreate(&ev_d2h_start));
    gpuErrchk(cudaEventCreate(&ev_d2h_end));

    gpuErrchk(cudaEventRecord(ev_h2d_start, 0));
    gpuErrchk(cudaMemcpyAsync(d_data, h_in.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice, 0));
    gpuErrchk(cudaEventRecord(ev_h2d_end, 0));
    gpuErrchk(cudaEventSynchronize(ev_h2d_end));
    gpuErrchk(cudaEventElapsedTime(&timings.h2d_ms, ev_h2d_start, ev_h2d_end));

    gpuErrchk(cudaEventRecord(ev_kstart, 0));
    {
        thrust::device_ptr<int> dev_begin(d_data);
        thrust::device_ptr<int> dev_end(d_data + n);
        thrust::sort(dev_begin, dev_end);
        timings.kernel_invocations += 1;
    }
    gpuErrchk(cudaEventRecord(ev_kend, 0));
    gpuErrchk(cudaEventSynchronize(ev_kend));
    gpuErrchk(cudaEventElapsedTime(&timings.total_kernel_ms, ev_kstart, ev_kend));

    gpuErrchk(cudaEventRecord(ev_d2h_start, 0));
    gpuErrchk(cudaMemcpyAsync(h_out.data(), d_data, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost, 0));
    gpuErrchk(cudaEventRecord(ev_d2h_end, 0));
    gpuErrchk(cudaEventSynchronize(ev_d2h_end));
    gpuErrchk(cudaEventElapsedTime(&timings.d2h_ms, ev_d2h_start, ev_d2h_end));

    timings.total_ms = timings.h2d_ms + timings.total_kernel_ms + timings.d2h_ms;

    gpuErrchk(cudaEventDestroy(ev_h2d_start));
    gpuErrchk(cudaEventDestroy(ev_h2d_end));
    gpuErrchk(cudaEventDestroy(ev_kstart));
    gpuErrchk(cudaEventDestroy(ev_kend));
    gpuErrchk(cudaEventDestroy(ev_d2h_start));
    gpuErrchk(cudaEventDestroy(ev_d2h_end));
    gpuErrchk(cudaFree(d_data));
}



// ------------------------- Multi-bit Radix Kernels -------------------------

// Per-block 16-bin histogram (shared memory + atomicAdd)
// d_block_hist: bin-major [16 * num_blocks], idx = bin * num_blocks + blockIdx.x
__global__ void histogram_kernel(const unsigned int* d_in, int* d_block_hist, int n, int shift) {
    __shared__ int bins[16];
    if (threadIdx.x < 16) bins[threadIdx.x] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int key = d_in[idx];
        int bin = (int)((key >> shift) & 0xFu);
        atomicAdd(&bins[bin], 1);
    }
    __syncthreads();

    if (threadIdx.x < 16) {
        d_block_hist[threadIdx.x * gridDim.x + blockIdx.x] = bins[threadIdx.x];
    }
}
// First pass: local exclusive scan per chunk (Blelloch) + per-chunk sums
// Input slice per bin: in[b*B : (b+1)*B), output: out_prefix[b*B : ...], block_sums[b*numChunks : ...]
template<int SCAN_THREADS>
__global__ void scan_chunks_local_kernel(const int* __restrict__ in, int* __restrict__ out_prefix,
                                         int* __restrict__ block_sums,
                                         int B, int numChunks) {
    extern __shared__ int sdata[]; // size SCAN_THREADS
    int chunk = blockIdx.x;
    int bin   = blockIdx.y;

    int global_base = bin * B;
    int i = chunk * SCAN_THREADS + threadIdx.x;

    int val = 0;
    if (i < B) val = in[global_base + i];
    sdata[threadIdx.x] = val;
    __syncthreads();

    // Blelloch exclusive scan (upsweep)
    for (int d = 1; d < SCAN_THREADS; d <<= 1) {
        int ai = ((threadIdx.x + 1) * d * 2) - 1;
        int bi = ai - d;
        if (ai < SCAN_THREADS) sdata[ai] += sdata[bi];
        __syncthreads();
    }

    // Save block sum (total)
    if (threadIdx.x == 0) {
        int total = sdata[SCAN_THREADS - 1];
        block_sums[bin * numChunks + chunk] = total;
    }

    // Set last element to 0 for exclusive scan
    if (threadIdx.x == 0) sdata[SCAN_THREADS - 1] = 0;
    __syncthreads();

    // Blelloch exclusive scan (downsweep)
    for (int d = SCAN_THREADS >> 1; d >= 1; d >>= 1) {
        int ai = ((threadIdx.x + 1) * d * 2) - 1;
        int bi = ai - d;
        if (ai < SCAN_THREADS) {
            int t = sdata[bi];
            sdata[bi] = sdata[ai];
            sdata[ai] += t;
        }
        __syncthreads();
    }

    if (i < B) out_prefix[global_base + i] = sdata[threadIdx.x];
}

// Second pass: exclusive scan of per-chunk sums (Blelloch) per bin (dynamic SUM_THREADS)
__global__ void scan_block_sums_kernel(const int* __restrict__ block_sums,
                                       int* __restrict__ block_sums_prefix,
                                       int numChunks) {
    extern __shared__ int sdata[]; // size = blockDim.x (SUM_THREADS)
    const int SUM_THREADS = blockDim.x;
    int bin = blockIdx.y;

    int i = threadIdx.x;
    int val = (i < numChunks) ? block_sums[bin * numChunks + i] : 0;
    sdata[i] = val;
    __syncthreads();

    // Upsweep
    for (int d = 1; d < SUM_THREADS; d <<= 1) {
        int ai = ((i + 1) * d * 2) - 1;
        int bi = ai - d;
        if (ai < SUM_THREADS) sdata[ai] += sdata[bi];
        __syncthreads();
    }

    // Set last to 0 for exclusive scan
    if (i == 0) sdata[SUM_THREADS - 1] = 0;
    __syncthreads();

    // Downsweep
    for (int d = SUM_THREADS >> 1; d >= 1; d >>= 1) {
        int ai = ((i + 1) * d * 2) - 1;
        int bi = ai - d;
        if (ai < SUM_THREADS) {
            int t = sdata[bi];
            sdata[bi] = sdata[ai];
            sdata[ai] += t;
        }
        __syncthreads();
    }

    if (i < numChunks) block_sums_prefix[bin * numChunks + i] = sdata[i];
}

// Third pass: add per-chunk offsets to per-element prefix
template<int SCAN_THREADS>
__global__ void add_chunk_offsets_kernel(int* __restrict__ out_prefix,
                                         const int* __restrict__ block_sums_prefix,
                                         int B, int numChunks) {
    int chunk = blockIdx.x;
    int bin   = blockIdx.y;

    int i = chunk * SCAN_THREADS + threadIdx.x;
    int offset = block_sums_prefix[bin * numChunks + chunk];

    if (i < B) {
        int global_base = bin * B;
        out_prefix[global_base + i] += offset;
    }
}

// Compute per-bin totals: total[b] = prefix_last + sum_last
__global__ void compute_bin_totals_kernel(const int* __restrict__ block_sums,
                                          const int* __restrict__ block_sums_prefix,
                                          int* __restrict__ bin_totals,
                                          int numChunks) {
    int bin = blockIdx.y;
    if (threadIdx.x == 0) {
        if (numChunks > 0) {
            int lastIdx = numChunks - 1;
            int total = block_sums_prefix[bin * numChunks + lastIdx] + block_sums[bin * numChunks + lastIdx];
            bin_totals[bin] = total;
        } else {
            bin_totals[bin] = 0;
        }
    }
}

// Exclusive scan over 16 bin totals to produce global bases
__global__ void scan_bin_totals_kernel(const int* __restrict__ bin_totals,
                                       int* __restrict__ bin_global_base) {
    __shared__ int s[16];
    if (threadIdx.x < 16) s[threadIdx.x] = bin_totals[threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {
        int run = 0;
        for (int b = 0; b < 16; ++b) {
            int v = s[b];
            s[b] = run;
            run += v;
        }
    }
    __syncthreads();
    if (threadIdx.x < 16) bin_global_base[threadIdx.x] = s[threadIdx.x];
}

// Scatter for 4-bit pass
// d_bin_block_prefix: bin-major [16 * num_blocks] (exclusive scan across blocks)
// d_bin_global_base: [16] global exclusive bases
__global__ void scatter_multi_kernel(const unsigned int* d_in, unsigned int* d_out, int n, int shift,
                                     const int* __restrict__ d_bin_block_prefix,
                                     const int* __restrict__ d_bin_global_base) {
    int tid   = threadIdx.x;
    int idx   = blockIdx.x * blockDim.x + tid;
    int block = blockIdx.x;

    int active = (idx < n) ? 1 : 0;
    unsigned int key = active ? d_in[idx] : 0u;
    int my_bin = active ? ((int)((key >> shift) & 0xFu)) : 0;

    int lane = tid & 31;
    int wid  = tid >> 5;
    unsigned int left_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);

    int num_warps = (blockDim.x + 31) / 32;
    extern __shared__ int s_mem[]; // warp_bin_counts[16 * num_warps]
    int* warp_bin_counts = s_mem;

    int left_in_warp_for_mybin = 0;
    for (int b = 0; b < 16; ++b) {
        unsigned int mask_b = __ballot_sync(0xFFFFFFFFu, active && (my_bin == b));
        int cnt_b = __popc(mask_b);
        if (lane == 0) {
            warp_bin_counts[b * num_warps + wid] = cnt_b;
        }
        if (b == my_bin) {
            left_in_warp_for_mybin = __popc(mask_b & left_mask);
        }
    }
    __syncthreads();

    if (tid == 0) {
        for (int b = 0; b < 16; ++b) {
            int run = 0;
            for (int w = 0; w < num_warps; ++w) {
                int c = warp_bin_counts[b * num_warps + w];
                warp_bin_counts[b * num_warps + w] = run;
                run += c;
            }
        }
    }
    __syncthreads();

    int warp_base_for_mybin = warp_bin_counts[my_bin * num_warps + wid];
    int local_offset_in_block = left_in_warp_for_mybin + warp_base_for_mybin;

    int num_blocks = gridDim.x;
    int block_prefix = d_bin_block_prefix[my_bin * num_blocks + block];
    int bin_global_base = d_bin_global_base[my_bin];

    if (active) {
        int pos = bin_global_base + block_prefix + local_offset_in_block;
        if (pos < n) d_out[pos] = key;
    }
}

// ------------------------- Multi-bit Radix Driver -------------------------
void radix_sort_gpu(const std::vector<int>& h_in_signed, std::vector<int>& h_out_signed, Timings &timings,
                    const DeviceInfo &dinfo, int threads = 512, int blocks_override = -1) {
    int n = (int)h_in_signed.size();
    if (n == 0) return;

    // Signed -> unsigned keys (order-preserving)
    std::vector<unsigned int> h_keys(n);
    for (int i = 0; i < n; ++i) h_keys[i] = static_cast<unsigned int>(h_in_signed[i]) ^ 0x80000000u;

    int block_size = threads;
    if (block_size > (int)dinfo.maxThreadsPerBlock) block_size = (int)dinfo.maxThreadsPerBlock;
    int t = 1; while (t < block_size) t <<= 1;
    if (t != block_size) block_size = t;

    int natural_blocks = (n + block_size - 1) / block_size;
    int blocks = natural_blocks;
    if (blocks_override > 0 && blocks_override >= natural_blocks) blocks = blocks_override;

    unsigned int *d_in = nullptr, *d_out = nullptr;
    gpuErrchk(cudaMalloc(&d_in, (size_t)n * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&d_out, (size_t)n * sizeof(unsigned int)));

    // per-pass buffers: histograms, block prefixes, global base
    int *d_block_hist = nullptr;
    int *d_bin_block_prefix = nullptr;
    int *d_bin_global_base  = nullptr;
    gpuErrchk(cudaMalloc(&d_block_hist,       (size_t)16 * blocks * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_bin_block_prefix, (size_t)16 * blocks * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_bin_global_base,  16 * sizeof(int)));

    // temporary buffers for device scans/reduces
   const int SCAN_THREADS_CONST = 1024; // power-of-two
    int numChunks = (blocks + SCAN_THREADS_CONST - 1) / SCAN_THREADS_CONST;
    int SUM_THREADS = next_pow2_up(numChunks);
    if (SUM_THREADS > 1024) SUM_THREADS = 1024;

    int *d_block_sums = nullptr;
    int *d_block_sums_prefix = nullptr;
    int *d_bin_totals = nullptr;
    gpuErrchk(cudaMalloc(&d_block_sums,        (size_t)16 * numChunks * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_block_sums_prefix, (size_t)16 * numChunks * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_bin_totals,        16 * sizeof(int)));

    // pinned host buffers
    unsigned int *h_pinned_in = nullptr;
    unsigned int *h_pinned_out = nullptr;
    gpuErrchk(cudaHostAlloc((void**)&h_pinned_in, (size_t)n * sizeof(unsigned int), cudaHostAllocDefault));
    gpuErrchk(cudaHostAlloc((void**)&h_pinned_out, (size_t)n * sizeof(unsigned int), cudaHostAllocDefault));
    std::memcpy(h_pinned_in, h_keys.data(), (size_t)n * sizeof(unsigned int));

    // H2D timing
    cudaEvent_t ev_h2d_start, ev_h2d_end;
    gpuErrchk(cudaEventCreate(&ev_h2d_start));
    gpuErrchk(cudaEventCreate(&ev_h2d_end));
    gpuErrchk(cudaEventRecord(ev_h2d_start, 0));
    gpuErrchk(cudaMemcpyAsync(d_in, h_pinned_in, (size_t)n * sizeof(unsigned int), cudaMemcpyHostToDevice, 0));
    gpuErrchk(cudaEventRecord(ev_h2d_end, 0));
    gpuErrchk(cudaEventSynchronize(ev_h2d_end));
    gpuErrchk(cudaEventElapsedTime(&timings.h2d_ms, ev_h2d_start, ev_h2d_end));
    gpuErrchk(cudaEventDestroy(ev_h2d_start));
    gpuErrchk(cudaEventDestroy(ev_h2d_end));

    // Kernel grouped timing (device-only)
    cudaEvent_t ev_kstart, ev_kend;
    gpuErrchk(cudaEventCreate(&ev_kstart));
    gpuErrchk(cudaEventCreate(&ev_kend));
    gpuErrchk(cudaEventRecord(ev_kstart, 0));

    for (int p = 0; p < 8; ++p) {
        int shift = p * 4;

        // 1) Per-block histogram
        histogram_kernel<<<blocks, block_size>>>(d_in, d_block_hist, n, shift);
        gpuErrchk(cudaGetLastError());
        timings.kernel_invocations += 1;

        // 2) Local scans per chunk (exclusive) + per-chunk sums
        {
            dim3 grid(numChunks, 16, 1);
            dim3 block(SCAN_THREADS_CONST, 1, 1);
            size_t shmem = SCAN_THREADS_CONST * sizeof(int);
            scan_chunks_local_kernel<SCAN_THREADS_CONST><<<grid, block, shmem>>>(d_block_hist, d_bin_block_prefix, d_block_sums, blocks, numChunks);
            gpuErrchk(cudaGetLastError());
            timings.kernel_invocations += 1;
        }

        // 3) Exclusive scan of per-chunk sums per bin (dynamic threads)
        {
            dim3 grid(1, 16, 1);
            dim3 block(SUM_THREADS, 1, 1);
            size_t shmem = (size_t)SUM_THREADS * sizeof(int);
            scan_block_sums_kernel<<<grid, block, shmem>>>(d_block_sums, d_block_sums_prefix, numChunks);
            gpuErrchk(cudaGetLastError());
            timings.kernel_invocations += 1;
        }

        // 4) Add per-chunk offsets to per-element prefix
        {
            dim3 grid(numChunks, 16, 1);
            dim3 block(SCAN_THREADS_CONST, 1, 1);
            add_chunk_offsets_kernel<SCAN_THREADS_CONST><<<grid, block>>>(d_bin_block_prefix, d_block_sums_prefix, blocks, numChunks);
            gpuErrchk(cudaGetLastError());
            timings.kernel_invocations += 1;
        }

        // 5) Compute per-bin totals
        {
            dim3 grid(1, 16, 1);
            compute_bin_totals_kernel<<<grid, 1>>>(d_block_sums, d_block_sums_prefix, d_bin_totals, numChunks);
            gpuErrchk(cudaGetLastError());
            timings.kernel_invocations += 1;
        }

        // 6) Exclusive scan over 16 bin totals → global bases
        {
            scan_bin_totals_kernel<<<1, 16>>>(d_bin_totals, d_bin_global_base);
            gpuErrchk(cudaGetLastError());
            timings.kernel_invocations += 1;
        }

        // 7) Scatter
        {
            int num_warps = (block_size + 31) / 32;
            int shm_bytes = 16 * num_warps * sizeof(int);
            scatter_multi_kernel<<<blocks, block_size, shm_bytes>>>(d_in, d_out, n, shift, d_bin_block_prefix, d_bin_global_base);
            gpuErrchk(cudaGetLastError());
            timings.kernel_invocations += 1;
        }

        std::swap(d_in, d_out);
    }

    gpuErrchk(cudaEventRecord(ev_kend, 0));
    gpuErrchk(cudaEventSynchronize(ev_kend));
    gpuErrchk(cudaEventElapsedTime(&timings.total_kernel_ms, ev_kstart, ev_kend));
    gpuErrchk(cudaEventDestroy(ev_kstart));
    gpuErrchk(cudaEventDestroy(ev_kend));

    // D2H timing
    cudaEvent_t ev_d2h_start, ev_d2h_end;
    gpuErrchk(cudaEventCreate(&ev_d2h_start));
    gpuErrchk(cudaEventCreate(&ev_d2h_end));
    gpuErrchk(cudaEventRecord(ev_d2h_start, 0));
    gpuErrchk(cudaMemcpyAsync(h_pinned_out, d_in, (size_t)n * sizeof(unsigned int), cudaMemcpyDeviceToHost, 0));
    gpuErrchk(cudaEventRecord(ev_d2h_end, 0));
    gpuErrchk(cudaEventSynchronize(ev_d2h_end));
    gpuErrchk(cudaEventElapsedTime(&timings.d2h_ms, ev_d2h_start, ev_d2h_end));
    gpuErrchk(cudaEventDestroy(ev_d2h_start));
    gpuErrchk(cudaEventDestroy(ev_d2h_end));

    for (int i = 0; i < n; ++i) {
        unsigned int k = h_pinned_out[i] ^ 0x80000000u;
        h_out_signed[i] = static_cast<int>(k);
    }

    timings.total_ms = timings.h2d_ms + timings.total_kernel_ms + timings.d2h_ms;

    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_out));
    gpuErrchk(cudaFree(d_block_hist));
    gpuErrchk(cudaFree(d_bin_block_prefix));
    gpuErrchk(cudaFree(d_bin_global_base));
    gpuErrchk(cudaFree(d_block_sums));
    gpuErrchk(cudaFree(d_block_sums_prefix));
    gpuErrchk(cudaFree(d_bin_totals));
    gpuErrchk(cudaFreeHost(h_pinned_in));
    gpuErrchk(cudaFreeHost(h_pinned_out));
}

#ifndef BUILD_RADIX_LIB


// ------------------------- CSV + CLI harness + sweep -------------------------
std::string timestamp_string() {
    std::time_t t = std::time(nullptr);
    std::tm tm;
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return std::string(buf);
}

void ensure_dir(const std::filesystem::path &p) {
    std::error_code ec;
    if (!std::filesystem::exists(p)) {
        std::filesystem::create_directories(p, ec);
        if (ec) {
            std::cerr << "Failed to create directory " << p << ": " << ec.message() << std::endl;
            exit(1);
        }
    }
}

struct CLIOptions {
    int min_exp = 27; // default to 2^27
    int max_exp = 27; // default to 2^27
    int reps = 3;
    std::string output_dir = "../radix_data";
    bool grid_flag = false;
    int device = 0;
    int threads = 512;
    int blocks_override = -1;
    unsigned int seed = 12345u;

    // sweep options
    std::vector<int> sweep_threads;
    std::vector<int> sweep_blocks;
    std::string sweep_name;
    int sweep_delay_ms = 200;
};

static std::vector<int> parse_list(const std::string &s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try { out.push_back(std::stoi(token)); } catch (...) {}
    }
    return out;
}

CLIOptions parse_cli(int argc, char** argv) {
    CLIOptions o;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if (s == "--min-exp" && i+1 < argc) { o.min_exp = std::stoi(argv[++i]); }
        else if (s == "--max-exp" && i+1 < argc) { o.max_exp = std::stoi(argv[++i]); }
        else if (s == "--reps" && i+1 < argc) { o.reps = std::stoi(argv[++i]); }
        else if ((s == "--output" || s == "--output-dir") && i+1 < argc) { o.output_dir = argv[++i]; }
        else if (s == "--grid") { o.grid_flag = true; }
        else if (s == "--device" && i+1 < argc) { o.device = std::stoi(argv[++i]); }
        else if (s == "--threads" && i+1 < argc) { o.threads = std::stoi(argv[++i]); }
        else if (s == "--blocks" && i+1 < argc) { o.blocks_override = std::stoi(argv[++i]); }
        else if (s == "--seed" && i+1 < argc) { o.seed = static_cast<unsigned int>(std::stoul(argv[++i])); }
        else if (s == "--sweep-threads" && i+1 < argc) { o.sweep_threads = parse_list(argv[++i]); }
        else if (s == "--sweep-blocks" && i+1 < argc) { o.sweep_blocks = parse_list(argv[++i]); }
        else if (s == "--sweep-name" && i+1 < argc) { o.sweep_name = argv[++i]; }
        else if (s == "--sweep-delay" && i+1 < argc) { o.sweep_delay_ms = std::stoi(argv[++i]); }
        else {
            std::cerr << "Unknown option: " << s << std::endl;
            exit(1);
        }
    }
    return o;
}

int main(int argc, char** argv) {
    auto opts = parse_cli(argc, argv);

    int devCount = 0;
    gpuErrchk(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }
    if (opts.device < 0 || opts.device >= devCount) {
        std::cerr << "Invalid device id: " << opts.device << ", devices available: " << devCount << std::endl;
        return 1;
    }
    gpuErrchk(cudaSetDevice(opts.device));
    DeviceInfo dinfo = queryDevice(opts.device);

    std::cout << "Using device " << opts.device << ": " << dinfo.name
              << " (SM " << dinfo.major << "." << dinfo.minor
              << ", SMs=" << dinfo.multiProcessorCount
              << ", maxThreads=" << dinfo.maxThreadsPerBlock
              << ", sharedMem=" << dinfo.sharedMemPerBlock << ")\n";

    ensure_dir(opts.output_dir);

    std::string csv_name;
    if (!opts.sweep_threads.empty() || !opts.sweep_blocks.empty()) {
        csv_name = opts.sweep_name.empty() ? "radix_sweep_" + timestamp_string() + ".csv" : opts.sweep_name;
    } else {
        csv_name = "radix_" + timestamp_string() + ".csv";
    }
    std::filesystem::path csv_path = std::filesystem::path(opts.output_dir) / csv_name;
    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV for writing: " << csv_path << std::endl;
        return 1;
    }

        csv << "timestamp,run_id,size,next_pow2,rep,seed,device_id,device_name,compute_capability,sm_count,threads,blocks,algorithm,"
       "h2d_ms,total_kernel_ms,kernel_invocations,d2h_ms,total_ms,throughput_Melems_s,verify_passed\n";    
       
       csv << std::fixed << std::setprecision(3);

    auto run_once = [&](int run_id, int size, int rep, unsigned int seed, int threads, int blocks_override) {
    std::mt19937 rng(seed + run_id + size + rep);
    std::uniform_int_distribution<int> dist(std::numeric_limits<int>::min()/2, std::numeric_limits<int>::max()/2);
    std::vector<int> h_in(size);
    for (int i = 0; i < size; ++i) h_in[i] = dist(rng);

    int block_size = threads;
    if (block_size > (int)dinfo.maxThreadsPerBlock) block_size = (int)dinfo.maxThreadsPerBlock;
    int tt = 1; while (tt < block_size) tt <<= 1; if (tt != block_size) block_size = tt;
    int natural_blocks = (size + block_size - 1) / block_size;
    int blocks_used = (blocks_override > 0 && blocks_override >= natural_blocks) ? blocks_override : natural_blocks;

    // 1) Naš Radix (CUDA event timing iz radix_sort_gpu)
    {
        std::vector<int> h_out(size);
        Timings t{};
        radix_sort_gpu(h_in, h_out, t, dinfo, threads, blocks_override);

        std::vector<int> h_ref = h_in;
        std::sort(h_ref.begin(), h_ref.end());
        bool ok = (h_ref == h_out);

        double throughput = (double)size / (t.total_ms / 1000.0) / 1e6;

        csv << timestamp_string() << "," << run_id << "," << size << "," << next_pow2_int(size) << ","
            << rep << "," << opts.seed << "," << opts.device << ",";
        std::string devname = dinfo.name; for (auto &c: devname) if (c == ',') c = ';';
        csv << "\"" << devname << "\"" << "," << dinfo.major << "." << dinfo.minor << ","
            << dinfo.multiProcessorCount << "," << threads << "," << blocks_used << ","
            << "\"CustomRadix\"" << ","
            << t.h2d_ms << "," << t.total_kernel_ms << "," << t.kernel_invocations << ","
            << t.d2h_ms << "," << t.total_ms << "," << throughput << "," << (ok ? "1" : "0") << "\n";
        csv.flush();

        std::cout << "[custom] run=" << run_id << " size=" << size << " threads=" << threads
                  << " total_ms=" << t.total_ms << " verify=" << (ok ? "OK" : "FAILED") << std::endl;
    }

    // 2) Thrust baseline (CUDA event timing iz official_thrust_sort_timed)
    {
        std::vector<int> h_thrust_out;
        Timings t{};
        official_thrust_sort_timed(h_in, h_thrust_out, t);

        std::vector<int> h_ref = h_in;
        std::sort(h_ref.begin(), h_ref.end());
        bool ok = (h_ref == h_thrust_out);

        double throughput = (double)size / (t.total_ms / 1000.0) / 1e6;

        csv << timestamp_string() << "," << run_id << "," << size << "," << next_pow2_int(size) << ","
            << rep << "," << opts.seed << "," << opts.device << ",";
        std::string devname = dinfo.name; for (auto &c: devname) if (c == ',') c = ';';
        csv << "\"" << devname << "\"" << "," << dinfo.major << "." << dinfo.minor << ","
            << dinfo.multiProcessorCount << "," << threads << "," << blocks_used << ","
            << "\"ThrustBaseline\"" << ","
            << t.h2d_ms << "," << t.total_kernel_ms << "," << t.kernel_invocations << ","
            << t.d2h_ms << "," << t.total_ms << "," << throughput << "," << (ok ? "1" : "0") << "\n";
        csv.flush();

        std::cout << "[thrust] run=" << run_id << " size=" << size << " threads=" << threads
                  << " total_ms=" << t.total_ms << " verify=" << (ok ? "OK" : "FAILED") << std::endl;
    }
};

    int global_run_id = 0;

    if (!opts.sweep_threads.empty() || !opts.sweep_blocks.empty()) {
        std::vector<int> threads_list = opts.sweep_threads.empty() ? std::vector<int>{opts.threads} : opts.sweep_threads;
        std::vector<int> blocks_list = opts.sweep_blocks.empty() ? std::vector<int>{opts.blocks_override > 0 ? opts.blocks_override : -1} : opts.sweep_blocks;

        int total_configs = (int)threads_list.size() * (int)blocks_list.size();
        int cfg_idx = 0;
        for (int th : threads_list) {
            for (int bl : blocks_list) {
                ++cfg_idx;
                std::cout << "=== Sweep config " << cfg_idx << "/" << total_configs << " : threads=" << th << " blocks_override=" << bl << " ===\n";
                for (int exp = opts.min_exp; exp <= opts.max_exp; ++exp) {
                    int size = 1 << exp;
                    for (int rep = 0; rep < opts.reps; ++rep) {
                        ++global_run_id;
                        run_once(global_run_id, size, rep, opts.seed, th, bl);
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(opts.sweep_delay_ms));
            }
        }
    } else {
        for (int exp = opts.min_exp; exp <= opts.max_exp; ++exp) {
            int size = 1 << exp;
            for (int rep = 0; rep < opts.reps; ++rep) {
                ++global_run_id;
                run_once(global_run_id, size, rep, opts.seed, opts.threads, opts.blocks_override);
            }
        }
    }

    csv.close();
    std::cout << "CSV results written to " << csv_path << std::endl;
    return 0;
}

#endif // BUILD_RADIX_LIB