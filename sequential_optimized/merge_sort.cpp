#include <vector>
#include <algorithm>
#include <cstring>  // for memcpy
#include <main_template.hpp>

const int INSERTION_SORT_THRESHOLD = 32; //empirijski odabir

void insertion_sort(int arr[], int l, int r) {
    for (int i = l + 1; i <= r; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= l && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Merge from arr[l..m] and arr[m+1..r] into temp[l..r], then copy back
void merge_with_temp(int arr[], int l, int m, int r, int temp[]) {
    int i = l;
    int j = m + 1;
    int k = l;
    while (i <= m && j <= r) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    // copy remainder
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];
    // copy back
    memcpy(arr + l, temp + l, (r - l + 1) * sizeof(int));
}

// Top-down recursive, but using temp buffer
void merge_sort_rec(int arr[], int l, int r, int temp[]) {
    if (r - l + 1 <= INSERTION_SORT_THRESHOLD) {
        insertion_sort(arr, l, r);
        return;
    }
    int m = l + (r - l) / 2;
    merge_sort_rec(arr, l, m, temp);
    merge_sort_rec(arr, m + 1, r, temp);
    // optional: if arr[m] <= arr[m+1], već spojeno, skip merge
    if (arr[m] <= arr[m + 1]) {
        return;
    }
    merge_with_temp(arr, l, m, r, temp);
}

void merge_sort_opt(int arr[], int n) {
    // temp buffer
    int *temp = new int[n];
    merge_sort_rec(arr, 0, n - 1, temp);
    delete[] temp;
}

void merge_sort_wrapper(std::vector<int>& vec) {
    merge_sort_opt(vec.data(), (int)vec.size());
}


int main(int argc, char* argv[]) {
    return run_sort("merge_sort", "sequential", merge_sort_wrapper, argc, argv);
}
