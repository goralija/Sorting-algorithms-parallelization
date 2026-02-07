#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../../include/main_template.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// KERNEL 1: Shared Memory (k do 512)
__global__ void bitonic_shared_kernel(int* d_arr, int n_pow2) {
    __shared__ int s_mem[512];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Učitavanje u shared
    s_mem[tid] = d_arr[idx]; 
    __syncthreads();

    for (int k = 2; k <= 512; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool asc = (tid & k) == 0;
                if ((s_mem[tid] > s_mem[ixj]) == asc) {
                    int tmp = s_mem[tid]; s_mem[tid] = s_mem[ixj]; s_mem[ixj] = tmp;
                }
            }
            __syncthreads();
        }
    }
    d_arr[idx] = s_mem[tid];
}

// KERNEL 2: Globalni Grid-Stride (k > 512)
__global__ void bitonic_global_stride_kernel(int* d_arr, int n_pow2, int j, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride petlja za sakrivanje latencije
    for (; i < n_pow2; i += stride) {
        int ixj = i ^ j;
        if (ixj > i) {
            bool asc = (i & k) == 0;
            if ((d_arr[i] > d_arr[ixj]) == asc) {
                int tmp = d_arr[i]; d_arr[i] = d_arr[ixj]; d_arr[ixj] = tmp;
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

    // Inicijalizacija cijelog d_arr sa INT_MAX (jako bitno!)
    // Koristimo pomoćni host vektor da popunimo padding jer je cudaMemset za int nespretan
    std::vector<int> padding(next_pow2, 2147483647);
    gpuErrchk(cudaMemcpy(d_arr, padding.data(), next_pow2 * sizeof(int), cudaMemcpyHostToDevice));
    
    // Prepisujemo originalne podatke preko paddinga
    gpuErrchk(cudaMemcpy(d_arr, h_arr.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // Postavke za sakrivanje latencije
    int threads = 512;
    // blocks = broj SM-ova * 32 (za maksimalni occupancy na sm_86)
    int blocks = 2048; 

    // 1. Shared faza (do 512)
    bitonic_shared_kernel<<<next_pow2 / 512, 512>>>(d_arr, next_pow2);
    gpuErrchk(cudaDeviceSynchronize());

    // 2. Global faza (od 1024 naviše)
    for (int k = 1024; k <= next_pow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_global_stride_kernel<<<blocks, threads>>>(d_arr, next_pow2, j, k);
        }
    }

    gpuErrchk(cudaDeviceSynchronize());
    
    // Vraćamo samo originalni n
    gpuErrchk(cudaMemcpy(h_arr.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_arr);
}

int main(int argc, char* argv[]) {
    cudaFree(0); 
    auto wrapper = [](std::vector<int>& vec) { bitonic_sort_gpu(vec); };
    return run_sort("Manual Bitonic Stride", "GPU-Optimized", wrapper, argc, argv);
}