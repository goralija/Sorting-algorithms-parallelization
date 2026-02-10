#include <cuda_runtime.h>
#include <vector>
#include "../../include/main_template.hpp"

using namespace std;

// ============================
// Swap funkcija kompatibilna s GPU
// ============================
__device__ void swapDevice(int &a, int &b) {
    int tmp = a;
    a = b;
    b = tmp;
}

// ============================
// Compare & swap na GPU-u
// ============================
__device__ void compAndSwap(int* a, int i, int j, bool asc) {
    if (asc) {
        if (a[i] > a[j])
            swapDevice(a[i], a[j]);
    } else {
        if (a[i] < a[j])
            swapDevice(a[i], a[j]);
    }
}

// ============================
// Merge faza bitoničke sekvence
// ============================
__device__ void bitonicMerge(int* a, int low, int cnt, bool asc) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            compAndSwap(a, i, i + k, asc);
        }
        bitonicMerge(a, low, k, asc);
        bitonicMerge(a, low + k, k, asc);
    }
}

// ============================
// Rekurzivna izgradnja bitoničke sekvence
// ============================
__device__ void bitonicSort(int* a, int low, int cnt, bool asc) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicSort(a, low, k, true);       // lijeva polovina u rastućem
        bitonicSort(a, low + k, k, false);  // desna polovina u opadajućem
        bitonicMerge(a, low, cnt, asc);     // spoji
    }
}

// ============================
// Kernel – jedan thread sekvencijalno
// ============================
__global__ void bitonicSortKernel(int* d_arr, int n) {
    // Jedan thread izvršava cijeli niz
    bitonicSort(d_arr, 0, n, true);
}

// ============================
// Wrapper za run_sort
// ============================
void bitonic_sort_gpu_naive(std::vector<int>& h_arr) {
    int N = h_arr.size();
    int* d_arr;

    // Alociraj memoriju na GPU
    cudaMalloc(&d_arr, N * sizeof(int));

    // Kopiraj niz sa CPU na GPU
    cudaMemcpy(d_arr, h_arr.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Pokreni kernel – 1 blok, 1 thread
    bitonicSortKernel<<<1, 1>>>(d_arr, N);
    cudaDeviceSynchronize();

    // Vrati rezultat sa GPU na CPU
    cudaMemcpy(h_arr.data(), d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Očisti memoriju
    cudaFree(d_arr);
}

// ============================
// Main
// ============================
//nvcc -O2 bitonic_sort.cu -o bitonic_sort.exe -rdc=true
//bitonic_sort.exe 131072 random
// 1048576
//moze cak i default vrijednost a to je 2^27 = 134217728
//samo stepeni 2
int main(int argc, char* argv[]) {
    return run_sort("bitonic_sort", "naive_gpu", bitonic_sort_gpu_naive, argc, argv);
}
