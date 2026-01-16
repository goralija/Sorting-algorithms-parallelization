#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "../../include/main_template.hpp"

using namespace std;

__device__ int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }
    int tmp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = tmp;
    return i + 1;
}

__device__ void insertion_sort(int* arr, int low, int high) {
    for (int i = low + 1; i <= high; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= low && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

__device__ void quick_sort(int* arr, int low, int high, int depth, int max_depth) {
    if (low < high) {
        if (depth > max_depth) {
            insertion_sort(arr, low, high);
            return;
        }
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1, depth + 1, max_depth);
        quick_sort(arr, pi + 1, high, depth + 1, max_depth);
    }
}

__global__ void quick_sort_kernel(int* d_arr, int n) {
    int max_depth = 3 * (int)log2f((float)n);
    quick_sort(d_arr, 0, n - 1, 0, max_depth);
}

void quick_sort_gpu_naive(std::vector<int>& h_arr) {
    int N = h_arr.size();
    int* d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Naivno, 1 blok 1 thread
    quick_sort_kernel<<<1,1>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr.data(), d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

//nvcc -O2 quick_sort.cu -o quick_sort.exe -rdc=true
//quick_sort.exe 100000 random
int main(int argc, char* argv[]) {
    return run_sort("quick_sort", "naive_gpu", quick_sort_gpu_naive, argc, argv);
}