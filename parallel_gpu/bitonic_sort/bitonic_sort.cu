#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../include/main_template.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// 1. OPTIMIZOVANI KERNEL: Radi sve korake j unutar Shared Memorije za male k
__global__ void bitonic_shared_kernel(int* d_arr, int n) {
    __shared__ int shared_mem[512]; 
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) shared_mem[tid] = d_arr[idx];
    else shared_mem[tid] = 2147483647; // INT_MAX
    __syncthreads();

    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool ascending = (tid & k) == 0;
                if ((shared_mem[tid] > shared_mem[ixj]) == ascending) {
                    int temp = shared_mem[tid];
                    shared_mem[tid] = shared_mem[ixj];
                    shared_mem[ixj] = temp;
                }
            }
            __syncthreads();
        }
    }
    if (idx < n) d_arr[idx] = shared_mem[tid];
}

// 2. GLOBALNI KERNEL: Ostaje za velike k (koristi Grid-Stride princip)
__global__ void bitonic_global_step_kernel(int* d_arr, int n, int j, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride) {
        int ixj = i ^ j;
        if (ixj > i) {
            bool ascending = (i & k) == 0;
            if ((d_arr[i] > d_arr[ixj]) == ascending) {
                int temp = d_arr[i];
                d_arr[i] = d_arr[ixj];
                d_arr[ixj] = temp;
            }
        }
    }
}

void bitonic_sort_gpu(std::vector<int>& h_arr) {
    int n = h_arr.size();
    int next_pow2 = 1;
    while (next_pow2 < n) next_pow2 <<= 1;

    int* d_arr;
    gpuErrchk(cudaMalloc(&d_arr, next_pow2 * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_arr, h_arr.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // --- CUDA EVENT MJERENJE POČETAK ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threadsPerBlock = 512;
    int blocksPerGrid = 160;

    // KORAK 1: Shared Memory za korake do 512
    bitonic_shared_kernel<<< (next_pow2 + 511)/512, 512 >>>(d_arr, next_pow2);

    // KORAK 2: Globalna memorija za korake od 1024 naviše
    for (int k = 1024; k <= next_pow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_global_step_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, next_pow2, j, k);
        }
    }

    // --- CUDA EVENT MJERENJE KRAJ ---
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "  [Bitonic Kernel Time: " << ms << " ms]" << std::endl;

    gpuErrchk(cudaMemcpy(h_arr.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Čišćenje
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_arr);
}

int main(int argc, char* argv[]) {
    // 1. WARM UP - Odmah na početku main-a
    cudaFree(0); 

    auto wrapper = [](std::vector<int>& vec) {
        bitonic_sort_gpu(vec);
    };

    return run_sort("Bitonic Sort", "GPU-Hybrid-Shared-Global", wrapper, argc, argv);
}