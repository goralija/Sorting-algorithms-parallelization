// Optimized Quick Sort in C++
#include <iostream>
#include <vector>
#include <algorithm> // for std::swap
#include "main_template.hpp"

using namespace std;

const int INSERTION_SORT_THRESHOLD = 64;

// Insertion sort for small subarrays
inline void insertion_sort(int arr[], int low, int high)
{
    for (int i = low + 1; i <= high; i++)
    {
        int key = arr[i];
        int j = i - 1;
        while (j >= low && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Median-of-three pivot selection
inline int median_of_three(int arr[], int low, int high)
{
    int mid = low + (high - low) / 2;
    if (arr[mid] < arr[low])
        swap(arr[mid], arr[low]);
    if (arr[high] < arr[low])
        swap(arr[high], arr[low]);
    if (arr[mid] < arr[high])
        swap(arr[mid], arr[high]);
    return arr[high]; // pivot chosen as median
}

// Partition using Lomuto scheme (with median-of-three pivot)
int partition(int arr[], int low, int high)
{
    int pivot = median_of_three(arr, low, high);
    int i = low - 1;
    for (int j = low; j < high; j++)
    {
        if (arr[j] <= pivot)
        {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Optimized quicksort with tail recursion elimination
void quick_sort(int arr[], int low, int high)
{
    while (low < high)
    {

        // Early exit if already sorted
        if (arr[low] <= arr[high])
            break;

        // Use insertion sort for small partitions
        if (high - low < INSERTION_SORT_THRESHOLD)
        {
            insertion_sort(arr, low, high);
            break;
        }

        int pi = partition(arr, low, high);

        // Tail recursion optimization
        if (pi - low < high - pi)
        {
            quick_sort(arr, low, pi - 1);
            low = pi + 1;
        }
        else
        {
            quick_sort(arr, pi + 1, high);
            high = pi - 1;
        }
    }
}

// Wrapper for benchmarking template
void quick_sort_optimized_wrapper(std::vector<int> &vec)
{
    if (!vec.empty())
        quick_sort(vec.data(), 0, static_cast<int>(vec.size()) - 1);
}

int main(int argc, char *argv[])
{
    return run_sort("quick_sort_optimized", "sequential", quick_sort_optimized_wrapper, argc, argv);
}
