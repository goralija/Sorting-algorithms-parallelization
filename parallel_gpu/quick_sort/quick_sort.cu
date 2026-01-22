#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "../../include/main_template.hpp"

struct Range { int start; int end; };

__global__ void final_partition_kernel(int* d_in, int* d_out, int start, int end, int pivot, int* d_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = end - start + 1;
    if (tid >= n) return;

    int val = d_in[start + tid];
    if (val < pivot) {
        int pos = atomicAdd(&d_counts[0], 1);
        d_out[start + pos] = val;
    } else if (val > pivot) {
        int pos = atomicAdd(&d_counts[1], 1);
        d_out[end - pos] = val;
    }
}

__global__ void fill_pivots_fast(int* d_out, int start, int count, int pivot) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) d_out[start + tid] = pivot;
}

void quick_sort_gpu_final(std::vector<int>& h_arr) {
    int n = h_arr.size();
    if (n <= 1) return;

    int *d_data, *d_temp, *d_counts;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_temp, n * sizeof(int));
    cudaMalloc(&d_counts, 2 * sizeof(int));
    cudaMemcpy(d_data, h_arr.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<Range> stack;
    stack.push_back({0, n - 1});

    while (!stack.empty()) {
        Range r = stack.back();
        stack.pop_back();
        int len = r.end - r.start + 1;
        if (len <= 1) continue;

        // 1. PAMETAN IZBOR PIVOTA (Median of Three) - Spasava od beskonačnog trajanja
        int v[3];
        cudaMemcpy(&v[0], d_data + r.start, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&v[1], d_data + r.start + len/2, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&v[2], d_data + r.end, sizeof(int), cudaMemcpyDeviceToHost);
        std::sort(v, v + 3);
        int pivot = v[1]; // Uzimamo srednji od tri elementa

        cudaMemset(d_counts, 0, 2 * sizeof(int));
        int threads = 256;
        int blocks = (len + threads - 1) / threads;

        final_partition_kernel<<<blocks, threads>>>(d_data, d_temp, r.start, r.end, pivot, d_counts);
        
        int counts[2];
        cudaMemcpy(counts, d_counts, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        int lt = counts[0];
        int gt = counts[1];
        int mid = len - lt - gt;

        if (mid > 0) fill_pivots_fast<<<(mid + 255)/256, 256>>>(d_temp, r.start + lt, mid, pivot);

        // 2. OPTIMIZACIJA: Kopiramo samo onaj dio koji smo upravo sortirali
        cudaMemcpy(d_data + r.start, d_temp + r.start, len * sizeof(int), cudaMemcpyDeviceToDevice);

        // 3. STABILNOST: Prvo rješavamo manju particiju (smanjuje dubinu stacka)
        if (lt < gt) {
            if (gt > 1) stack.push_back({r.end - gt + 1, r.end});
            if (lt > 1) stack.push_back({r.start, r.start + lt - 1});
        } else {
            if (lt > 1) stack.push_back({r.start, r.start + lt - 1});
            if (gt > 1) stack.push_back({r.end - gt + 1, r.end});
        }
    }

    cudaMemcpy(h_arr.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data); cudaFree(d_temp); cudaFree(d_counts);
}

int main(int argc, char* argv[]) {
    auto wrapper = [](std::vector<int>& vec) { quick_sort_gpu_final(vec); };
    return run_sort("QuickSort", "GPU-Final-Optimized", wrapper, argc, argv);
}