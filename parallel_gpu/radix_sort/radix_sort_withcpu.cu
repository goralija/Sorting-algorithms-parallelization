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

// Kernel koji radi stabilnu particiju za jedan bit
__global__ void radix_partition_kernel(const unsigned int* d_in, unsigned int* d_out, int n, int shift, int* d_block_zeros) {
    extern __shared__ int temp[]; // shared memorija za prefix sum
    int thid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Svaki thread provjerava da li je bit 0 ili 1
    int val = 0;
    unsigned int element = 0;
    if (idx < n) {
        element = d_in[idx];
        val = ((element >> shift) & 1) == 0 ? 1 : 0; // tražimo nule za prefix sum
    }

    // 2. Ekskluzivni scan unutar bloka (Blelloch/Hillis-Steele varijanta)
    temp[thid] = val;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int t = 0;
        if (thid >= offset) t = temp[thid - offset];
        __syncthreads();
        temp[thid] += t;
        __syncthreads();
    }

    // Sada temp[thid] sadrži broj nula do tog threada (inkluzivno)
    int local_zeros = temp[thid];
    int total_zeros_in_block = temp[blockDim.x - 1];

    // 3. Snimi ukupan broj nula u bloku za globalni offset
    if (thid == blockDim.x - 1) {
        d_block_zeros[blockIdx.x] = total_zeros_in_block;
    }
    __syncthreads();

    // Čekamo da CPU ili drugi kernel izračuna globalne offsete
    // U ovoj verziji koristimo atomics za globalni dio, ali lokalni redoslijed je očuvan scan-om
}

// Finalni kernel koji smješta elemente koristeći izračunate offsete
__global__ void scatter_stable_kernel(const unsigned int* d_in, unsigned int* d_out, int n, int shift, int* d_global_offset_zeros, int* d_global_offset_ones, int total_zeros_global) {
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int is_zero = 0;
    unsigned int element = 0;
    if (idx < n) {
        element = d_in[idx];
        is_zero = ((element >> shift) & 1) == 0 ? 1 : 0;
    }

    // Ponovni scan za pozicije
    temp[thid] = is_zero;
    __syncthreads();
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int t = 0;
        if (thid >= offset) t = temp[thid - offset];
        __syncthreads();
        temp[thid] += t;
        __syncthreads();
    }

    if (idx < n) {
        int pos;
        if (is_zero) {
            // Pozicija = globalni početak za nule u ovom bloku + lokalni indeks nule
            pos = d_global_offset_zeros[blockIdx.x] + (temp[thid] - 1);
        } else {
            // Pozicija = ukupne nule + globalni početak za jedinice u ovom bloku + lokalni indeks jedinice
            int local_one_idx = (thid + 1) - temp[thid];
            pos = total_zeros_global + d_global_offset_ones[blockIdx.x] + (local_one_idx - 1);
        }
        d_out[pos] = element;
    }
}

void radix_sort_gpu(std::vector<int>& host_arr) {
    int n = host_arr.size();
    if (n <= 1) return;

    unsigned int *d_in, *d_out;
    int *d_block_zeros, *d_global_offset_zeros, *d_global_offset_ones;

    int threadsPerBlock = 512;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    gpuErrchk(cudaMalloc(&d_in, n * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&d_out, n * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&d_block_zeros, blocksPerGrid * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_global_offset_zeros, blocksPerGrid * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_global_offset_ones, blocksPerGrid * sizeof(int)));

    gpuErrchk(cudaMemcpy(d_in, host_arr.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    for (int shift = 0; shift < 32; ++shift) {
        // 1. Saznaj koliko svaki blok ima nula
        radix_partition_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_in, d_out, n, shift, d_block_zeros);
        
        // 2. Prefix sum na CPU za te blokove
        std::vector<int> h_block_zeros(blocksPerGrid);
        gpuErrchk(cudaMemcpy(h_block_zeros.data(), d_block_zeros, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));
        
        std::vector<int> h_off_zeros(blocksPerGrid);
        std::vector<int> h_off_ones(blocksPerGrid);
        
        int current_zero_ptr = 0;
        int current_one_ptr = 0;
        
        for (int i = 0; i < blocksPerGrid; ++i) {
            h_off_zeros[i] = current_zero_ptr;
            h_off_ones[i] = current_one_ptr;
            current_zero_ptr += h_block_zeros[i];
            
            int elements_in_block = (i == blocksPerGrid - 1) ? (n - i * threadsPerBlock) : threadsPerBlock;
            current_one_ptr += (elements_in_block - h_block_zeros[i]);
        }
        
        gpuErrchk(cudaMemcpy(d_global_offset_zeros, h_off_zeros.data(), blocksPerGrid * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_global_offset_ones, h_off_ones.data(), blocksPerGrid * sizeof(int), cudaMemcpyHostToDevice));

        // 3. Premjesti elemente (Scatter)
        scatter_stable_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_in, d_out, n, shift, d_global_offset_zeros, d_global_offset_ones, current_zero_ptr);
        
        gpuErrchk(cudaDeviceSynchronize());
        std::swap(d_in, d_out);
    }

    gpuErrchk(cudaMemcpy(host_arr.data(), d_in, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_block_zeros); cudaFree(d_global_offset_zeros); cudaFree(d_global_offset_ones);
}

int main(int argc, char* argv[]) {
    auto sort_wrapper = [](std::vector<int>& arr) {
        radix_sort_gpu(arr);
    };
    return run_sort("Radix Sort", "parallel_gpu", sort_wrapper, argc, argv);
}