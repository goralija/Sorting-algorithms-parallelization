#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../../include/main_template.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// 1. KERNEL: Brojanje nula u blokovima
__global__ void count_kernel(const unsigned int* d_in, int* d_block_zeros, int n, int shift) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    int is_zero = (idx < n && !((d_in[idx] >> shift) & 1)) ? 1 : 0;

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset /= 2)
        is_zero += __shfl_down_sync(0xFFFFFFFF, is_zero, offset);

    if (tid % 32 == 0) {
        __shared__ int warp_sums[16]; 
        warp_sums[tid / 32] = is_zero;
        __syncthreads();

        if (tid == 0) {
            int block_total = 0;
            for (int i = 0; i < blockDim.x / 32; i++) block_total += warp_sums[i];
            d_block_zeros[blockIdx.x] = block_total;
        }
    }
}

// 2. KERNEL: Scan za offsete
__global__ void scan_kernel(int* d_block_zeros, int* d_off_zeros, int* d_off_ones, int num_blocks, int n, int* d_total_zeros) {
    if (threadIdx.x == 0) {
        int count_z = 0;
        int count_o = 0;
        for (int i = 0; i < num_blocks; i++) {
            int z = d_block_zeros[i];
            d_off_zeros[i] = count_z;
            d_off_ones[i] = count_o;
            count_z += z;
            int current_block_size = (i == num_blocks - 1) ? (n - i * 512) : 512;
            count_o += (current_block_size - z);
        }
        *d_total_zeros = count_z;
    }
}

// 3. KERNEL: Stabilni scatter
__global__ void scatter_kernel(const unsigned int* d_in, unsigned int* d_out, int n, int shift, int* d_off_zeros, int* d_off_ones, int* d_total_zeros) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= n) return;

    unsigned int val = d_in[idx];
    int is_zero = !((val >> shift) & 1);

    unsigned int mask = __ballot_sync(0xFFFFFFFF, is_zero);
    int local_offset = __popc(mask & ((1 << (tid % 32)) - 1));
    
    __shared__ int warp_base[16]; 
    if (tid % 32 == 0) warp_base[tid / 32] = __popc(mask);
    __syncthreads();

    if (tid == 0) {
        int sum = 0;
        for(int i=0; i < 16; i++) {
            int tmp = warp_base[i];
            warp_base[i] = sum;
            sum += tmp;
        }
    }
    __syncthreads();

    local_offset += warp_base[tid / 32];

    int pos;
    if (is_zero) {
        pos = d_off_zeros[blockIdx.x] + local_offset;
    } else {
        int ones_local_offset = tid - local_offset;
        pos = (*d_total_zeros) + d_off_ones[blockIdx.x] + ones_local_offset;
    }
    d_out[pos] = val;
}

void radix_sort_gpu(std::vector<int>& host_arr) {
    int n = host_arr.size();
    if (n <= 1) return;

    unsigned int *d_in, *d_out;
    int *d_bz, *d_oz, *d_oo, *d_tz;

    int threads = 512;
    int blocks = (n + threads - 1) / threads;

    gpuErrchk(cudaMalloc(&d_in, n * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&d_out, n * sizeof(unsigned int)));
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
        scan_kernel<<<1, 1>>>(d_bz, d_oz, d_oo, blocks, n, d_tz);
        scatter_kernel<<<blocks, threads>>>(d_in, d_out, n, s, d_oz, d_oo, d_tz);
        std::swap(d_in, d_out);
    }

    // --- CUDA EVENT MJERENJE KRAJ ---
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "  [Radix Kernel (32-pass) Time: " << ms << " ms]" << std::endl;

    gpuErrchk(cudaMemcpy(host_arr.data(), d_in, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_bz); cudaFree(d_oz); cudaFree(d_oo); cudaFree(d_tz);
}

int main(int argc, char* argv[]) {
    // Warm-up na nivou main-a
    cudaFree(0); 

    auto sort_wrapper = [](std::vector<int>& arr) { radix_sort_gpu(arr); };
    return run_sort("Radix Sort", "GPU-1-bit-Stable", sort_wrapper, argc, argv);
}