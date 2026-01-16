// Naive Radix Sort on GPU (sequential, 1 thread)
// Executes entirely on GPU, no parallelization

#include <cuda_runtime.h>
#include <vector>
#include "../../include/main_template.hpp"
#include <algorithm> // for std::max

using namespace std;

// ============================
// Device functions
// ============================

// Find max value in device array
__device__ int getMaxDevice(int* arr, int n) {
    int mx = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > mx) mx = arr[i];
    }
    return mx;
}

// Counting sort for a single digit (LSD)
__device__ void countSortDevice(int* arr, int n, int exp) {
    int* output = new int[n]; // VLA-like dynamic allocation on GPU
    int count[10] = {0};

    // Count occurrences of each digit
    for (int i = 0; i < n; i++) {
        int digit = (arr[i] / exp) % 10;
        count[digit]++;
    }

    // Cumulative count
    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];

    // Build output array (traverse from end for stability)
    for (int i = n - 1; i >= 0; i--) {
        int digit = (arr[i] / exp) % 10;
        output[count[digit] - 1] = arr[i];
        count[digit]--;
    }

    // Copy back
    for (int i = 0; i < n; i++)
        arr[i] = output[i];

    delete[] output;
}

// Radix sort on device (sequential)
__device__ void radixSortDevice(int* arr, int n) {
    if (n <= 1) return;
    int m = getMaxDevice(arr, n);

    for (int exp = 1; m / exp > 0; exp *= 10)
        countSortDevice(arr, n, exp);
}

// ============================
// GPU kernel - 1 thread
// ============================
__global__ void radixSortKernel(int* d_arr, int n) {
    // single thread executes entire sort
    radixSortDevice(d_arr, n);
}

// ============================
// Wrapper for run_sort
// ============================
void radix_sort_gpu_naive(std::vector<int>& h_arr) {
    int N = h_arr.size();
    int* d_arr;

    // Allocate device memory
    cudaMalloc(&d_arr, N * sizeof(int));

    // Copy to GPU
    cudaMemcpy(d_arr, h_arr.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: 1 block, 1 thread
    radixSortKernel<<<1, 1>>>(d_arr, N);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_arr.data(), d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_arr);
}

// ============================
// Main
// ============================
//nvcc -O2 radix_sort.cu -o radix_sort.exe -rdc=true
//radix_sort.exe 100000 random
int main(int argc, char* argv[]) {
    return run_sort("radix_sort", "naive_gpu", radix_sort_gpu_naive, argc, argv);
}
