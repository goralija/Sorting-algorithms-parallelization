#include <cuda_runtime.h>
#include <vector>
#include "../../include/main_template.hpp"

using namespace std;

// ============================
// Device merge
// ============================
__device__ void mergeDevice(int* arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int* L = new int[n1];
    int* R = new int[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];

    while (j < n2)
        arr[k++] = R[j++];

    delete[] L;
    delete[] R;
}

// ============================
// Device merge sort (recursive)
// ============================
__device__ void mergeSortDevice(int* arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        mergeSortDevice(arr, l, m);
        mergeSortDevice(arr, m + 1, r);
        mergeDevice(arr, l, m, r);
    }
}

// ============================
// Kernel – 1 thread
// ============================
__global__ void mergeSortKernel(int* d_arr, int n) {
    mergeSortDevice(d_arr, 0, n - 1);
}

// ============================
// Wrapper for run_sort
// ============================
void merge_sort_gpu_naive(std::vector<int>& h_arr) {
    int N = h_arr.size();
    int* d_arr;

    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // 1 block, 1 thread → sekvencijalno
    mergeSortKernel<<<1, 1>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr.data(), d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

// ============================
// Main
// ============================
//nvcc -O2 merge_sort.cu -o merge_sort.exe -rdc=true
//merge_sort.exe 100000 random
int main(int argc, char* argv[]) {
    return run_sort("merge_sort", "naive_gpu", merge_sort_gpu_naive, argc, argv);
}
