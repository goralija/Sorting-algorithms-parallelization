#include <algorithm>
#include <vector>
#include <cstring> // memmove
#include "main_template.hpp"

using namespace std;

// ---------------------------
// Insertion sort sa memmove
// ---------------------------
void insertionSort(int *a, int low, int cnt)
{
    for (int i = low + 1; i < low + cnt; i++)
    {
        int key = a[i];
        int j = i;
        while (j > low && a[j - 1] > key)
            j--;
        if (j != i)
        {
            memmove(a + j + 1, a + j, (i - j) * sizeof(int));
            a[j] = key;
        }
    }
}

// ---------------------------
// Compare & swap inline
// ---------------------------
inline void compAndSwapAsc(int *a, int i, int j)
{
    if (a[i] > a[j])
        swap(a[i], a[j]);
}
inline void compAndSwapDesc(int *a, int i, int j)
{
    if (a[i] < a[j])
        swap(a[i], a[j]);
}

// ---------------------------
// Iterativni bitonički merge
// ---------------------------
void bitonicMergeIter(int *a, int low, const int cnt, bool asc)
{
    for (int step = cnt / 2; step > 0; step /= 2)
    {
        for (int i = low; i < low + cnt; i += 2 * step)
        {
            for (int j = i; j < i + step; j++)
            {
                if (asc)
                    compAndSwapAsc(a, j, j + step);
                else
                    compAndSwapDesc(a, j, j + step);
            }
        }
    }
}

// ---------------------------
// Iterativni bitonički sort sa insertion sort pragom
// ---------------------------
void bitonicSortIter(int *a, int n)
{
    const int INSERTION_THRESHOLD = 32; // tweakabilno

    // Step 1: sort male sekvence insertion sortom
    for (int i = 0; i < n; i += INSERTION_THRESHOLD)
    {
        int len = min(INSERTION_THRESHOLD, n - i);
        insertionSort(a, i, len);
    }

    // Step 2: gradimo bitoničke sekvence iterativno
    for (int size = INSERTION_THRESHOLD; size <= n; size <<= 1)
    {
        for (int low = 0; low < n; low += size)
        {
            int mid = size / 2;
            int cnt = min(size, n - low);

            // in-place obrni desnu polovinu da bude opadajuća
            if (mid < cnt)
            {
                int *l = a + low + mid;
                int *r = a + low + cnt - 1;
                while (l < r)
                    swap(*l++, *r--);
            }

            bitonicMergeIter(a, low, cnt, true);
        }
    }
}

// ---------------------------
// Wrapper za run_sort
// ---------------------------
void bitonic_sort_wrapper(std::vector<int> &vec)
{
    bitonicSortIter(vec.data(), vec.size());
}

// ---------------------------
// Main
// ---------------------------
int main(int argc, char *argv[])
{
    return run_sort("bitonic_sort_iterative_optimized", "sequential", bitonic_sort_wrapper, argc, argv);
}
