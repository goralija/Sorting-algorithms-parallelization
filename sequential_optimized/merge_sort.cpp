#include <vector>
#include <algorithm>
#include <cstring> // for memcpy
#include <main_template.hpp>

const int INSERTION_SORT_THRESHOLD = 16; // empirijski odabir

void insertion_sort(int arr[], int l, int r)
{
    for (int i = l + 1; i <= r; i++)
    {
        int key = arr[i];
        int j = i - 1;
        while (j >= l && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j--;
        }

        arr[j + 1] = key;
    }
}

// Merge dva segmenta arr[l..m] i arr[m+1..r] u temp[l..r]
void merge(int src[], int dest[], int l, int m, int r)
{
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r)
    {
        if (src[i] <= src[j])
            dest[k++] = src[i++];
        else
            dest[k++] = src[j++];
    }
    while (i <= m)
        dest[k++] = src[i++];
    while (j <= r)
        dest[k++] = src[j++];
}

// Bottom-up iterativni merge sort s ping-pong temp bufferom
void merge_sort_opt(int arr[], int n)
{
    int *temp = new int[n];
    int *src = arr;
    int *dest = temp;

    // Prvo sortiraj male segmente insertion sort-om
    for (int i = 0; i < n; i += INSERTION_SORT_THRESHOLD)
    {
        int r = std::min(i + INSERTION_SORT_THRESHOLD - 1, n - 1);
        insertion_sort(src, i, r);
    }

    // Bottom-up merge
    for (int width = INSERTION_SORT_THRESHOLD; width < n; width *= 2)
    {
        for (int i = 0; i < n; i += 2 * width)
        {
            int l = i;
            int m = std::min(i + width - 1, n - 1);
            int r = std::min(i + 2 * width - 1, n - 1);
            if (m < r)
            {
                // Skip merge ako su već sortirani
                if (src[m] <= src[m + 1])
                {
                    memcpy(dest + l, src + l, (r - l + 1) * sizeof(int));
                }
                else
                {
                    merge(src, dest, l, m, r);
                }
            }
            else
            {
                // samo kopiraj ostatak
                memcpy(dest + l, src + l, (r - l + 1) * sizeof(int));
            }
        }
        // swap src i dest
        std::swap(src, dest);
    }

    // Ako je posljednji rezultat u temp, kopiraj nazad u arr
    if (src != arr)
    {
        memcpy(arr, src, n * sizeof(int));
    }

    delete[] temp;
}

// Wrapper za std::vector
void merge_sort_wrapper(std::vector<int> &vec)
{
    merge_sort_opt(vec.data(), (int)vec.size());
}

int main(int argc, char *argv[])
{
    return run_sort("merge_sort_optimized", "sequential", merge_sort_wrapper, argc, argv);
}