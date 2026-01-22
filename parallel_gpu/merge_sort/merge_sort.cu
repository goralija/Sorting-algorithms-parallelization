#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../include/main_template.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// 1. KERNEL: Bitonic Sort unutar bloka (Shared Memory)
__global__ void bitonic_sort_shared_kernel(int* d_data, int n) {
    __shared__ int shared_mem[512];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) shared_mem[tid] = d_data[idx];
    else shared_mem[tid] = 2147483647; // INT_MAX
    __syncthreads();

    for (int k = 2; k <= 512; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (shared_mem[tid] > shared_mem[ixj]) {
                        int temp = shared_mem[tid];
                        shared_mem[tid] = shared_mem[ixj];
                        shared_mem[ixj] = temp;
                    }
                } else {
                    if (shared_mem[tid] < shared_mem[ixj]) {
                        int temp = shared_mem[tid];
                        shared_mem[tid] = shared_mem[ixj];
                        shared_mem[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (idx < n) d_data[idx] = shared_mem[tid];
}

// 2. KERNEL: Paralelni Globalni Merge (Merge Path logika)
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
        int i = (low + high) / 2;
        int j = k - 1 - i;
        if (A[i] < B[j]) low = i + 1;
        else high = i;
    }
    
    int i = low;
    int j = k - i;

    if (i < len_a && (j >= len_b || A[i] <= B[j])) {
        d_out[idx] = A[i];
    } else {
        d_out[idx] = B[j];
    }
}

void merge_sort_gpu(std::vector<int>& host_arr) {
    int n = host_arr.size();
    if (n <= 1) return;

    int *d_ptr1, *d_ptr2;
    gpuErrchk(cudaMalloc(&d_ptr1, n * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_ptr2, n * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_ptr1, host_arr.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // --- CUDA EVENT MJERENJE ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Faza 1: Sortiranje blokova (inicijalni width=512)
    int threadsPerBlock = 512;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    bitonic_sort_shared_kernel<<<blocks, threadsPerBlock>>>(d_ptr1, n);

    // Faza 2: Globalni Merge
    int *d_in = d_ptr1, *d_out = d_ptr2;
    for (int width = 512; width < n; width *= 2) {
        int threads_merge = 256;
        int blocks_merge = (n + threads_merge - 1) / threads_merge;
        parallel_merge_kernel<<<blocks_merge, threads_merge>>>(d_in, d_out, n, width);
        std::swap(d_in, d_out);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "  [Merge Logic Time: " << milliseconds << " ms]" << std::endl;

    // Kopiranje konačnog rezultata (d_in jer je zadnji swap zamijenio d_in i d_out)
    gpuErrchk(cudaMemcpy(host_arr.data(), d_in, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_ptr1); cudaFree(d_ptr2);
}

int main(int argc, char* argv[]) {
    // WARM-UP ovdje osigurava da run_sort dobije "čista" vremena
    cudaFree(0); 

    auto sort_wrapper = [](std::vector<int>& arr) { merge_sort_gpu(arr); };
    return run_sort("Merge Sort", "GPU-Hybrid-Bitonic-Path", sort_wrapper, argc, argv);
}