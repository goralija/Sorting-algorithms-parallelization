#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "../../include/main_template.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void count_kernel(const unsigned int* d_in, int* d_block_zeros, int n, int shift) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int is_zero = 0;
    if (idx < n) {
        unsigned int val = d_in[idx];
        is_zero = (shift == 31) ? ((val >> 31) & 1) : !((val >> shift) & 1);
    }

    for (int offset = 16; offset > 0; offset /= 2)
        is_zero += __shfl_down_sync(0xFFFFFFFF, is_zero, offset);

    if (tid % 32 == 0) {
        __shared__ int warp_sums[32];
        warp_sums[tid / 32] = is_zero;
        __syncthreads();
        if (tid == 0) {
            int block_total = 0;
            for (int i = 0; i < blockDim.x / 32; i++) block_total += warp_sums[i];
            d_block_zeros[blockIdx.x] = block_total;
        }
    }
}

__global__ void scan_kernel_multi_block(int* d_block_zeros, int* d_off_zeros, int* d_off_ones, int num_blocks, int n, int* d_total_zeros) {
    if (threadIdx.x == 0) {
        int running_zeros = 0;
        for (int i = 0; i < num_blocks; i++) {
            int z = d_block_zeros[i];
            d_off_zeros[i] = running_zeros;
            d_off_ones[i] = (i * 512) - running_zeros;
            running_zeros += z;
        }
        *d_total_zeros = running_zeros;
    }
}

__global__ void scatter_kernel(const unsigned int* d_in, unsigned int* d_out, int n, int shift, int* d_off_zeros, int* d_off_ones, int* d_total_zeros) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= n) return;

    unsigned int val = d_in[idx];
    int is_zero = (shift == 31) ? ((val >> 31) & 1) : !((val >> shift) & 1);

    unsigned int mask = __ballot_sync(0xFFFFFFFF, is_zero);
    int local_offset = __popc(mask & ((1 << (tid % 32)) - 1));
    
    __shared__ int warp_base[32];
    if (tid % 32 == 0) warp_base[tid / 32] = __popc(mask);
    __syncthreads();

    if (tid == 0) {
        int sum = 0;
        for(int i=0; i < blockDim.x/32; i++) {
            int tmp = warp_base[i];
            warp_base[i] = sum;
            sum += tmp;
        }
    }
    __syncthreads();

    local_offset += warp_base[tid / 32];
    int pos = is_zero ? (d_off_zeros[blockIdx.x] + local_offset) : 
                       ((*d_total_zeros) + d_off_ones[blockIdx.x] + (tid - local_offset));

    if (pos < n) d_out[pos] = val;
}

void radix_sort_gpu(std::vector<int>& host_arr) {
    int n = host_arr.size();
    unsigned int *d_in, *d_out;
    int *d_bz, *d_oz, *d_oo, *d_tz;
    int threads = 512;
    int blocks = (n + threads - 1) / threads;

    gpuErrchk(cudaMalloc(&d_in, n * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_out, n * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_bz, blocks * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_oz, blocks * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_oo, blocks * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_tz, sizeof(int)));

    gpuErrchk(cudaMemcpy(d_in, host_arr.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // --- CUDA EVENT MJERENJE POČETAK ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int s = 0; s < 32; s++) {
        count_kernel<<<blocks, threads>>>(d_in, d_bz, n, s);
        scan_kernel_multi_block<<<1, 1>>>(d_bz, d_oz, d_oo, blocks, n, d_tz);
        scatter_kernel<<<blocks, threads>>>(d_in, d_out, n, s, d_oz, d_oo, d_tz);
        std::swap(d_in, d_out);
    }

    // --- CUDA EVENT MJERENJE KRAJ ---
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "  [Radix Core Kernel Time: " << ms << " ms]" << std::endl;

    gpuErrchk(cudaMemcpy(host_arr.data(), d_in, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_bz); cudaFree(d_oz); cudaFree(d_oo); cudaFree(d_tz);
}
/*
int main(int argc, char* argv[]) {
    cudaFree(0); 
    auto sort_wrapper = [](std::vector<int>& arr) { radix_sort_gpu(arr); };
    return run_sort("Radix Sort", "GPU-Stable-Timed", sort_wrapper, argc, argv);
}*/