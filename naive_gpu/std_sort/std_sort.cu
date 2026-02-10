#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <vector>
#include "../../include/main_template.hpp"

/*
================================================================================
Sekvencijalni std::sort na GPU-u – objašnjenje

1. Šta je std::sort?
   - std::sort je standardna C++ bibliotečka funkcija za sortiranje.
   - Implementirana je kao introsort: kombinacija quicksort-a, heapsort-a i insertion sort-a
     za male segmente.
   - Radi **isključivo na CPU-u** i koristi host memoriju.

2. Zašto ne možemo pozvati std::sort direktno na GPU-u?
   - GPU kod (u __device__ ili __global__ funkcijama) ne može pozivati host funkcije.
   - To znači da unutar CUDA kernela ili device funkcije ne možeš uraditi:
       std::sort(arr, arr + n);
     jer je std::sort host-only.
   - Ako pokušaš, dobićeš grešku kompajlera:
       "calling a __host__ function from a __global__ function is not allowed"
       i/ili "identifier std::sort undefined in device code".

3. Šta smo do sada koristili?
   - Kada smo pokušavali sekvencijalni GPU sort, morali smo koristiti **insertion sort** 
     ili neku jednostavnu implementaciju jer se izvršava unutar **jednog GPU thread-a**.
   - Ova insertion sort implementacija NIJE std::sort, ali je jedini način da imamo 
     sekvencijalni GPU kod koji radi sortiranje bez paralelizacije.

4. Zašto Thrust ne rešava problem ako hoćemo “sekvencijalno std::sort”?
   - Thrust je biblioteka za GPU i **uvijek je paralelizovana** (interno koristi hiljade thread-ova).
   - Ako je cilj sekvencijalno izvršavanje i poređenje sa CPU std::sort, Thrust nije opcija.
   - Thrust daje GPU sortiranje, ali to nije “isti algoritam i sekvencijalno” – ne možeš ga porediti direktno sa CPU std::sort.

5. Zaključak:
   - **Ne postoji jednostavan način da std::sort radi na GPU-u** bez paralelizacije.
   - Jedina opcija je **implementirati introsort/quicksort + heapsort + insertion from scratch** 
     kao device funkcije i pokrenuti ga u **jednom thread-u** da bi se ponašao kao “sekvencijalni std::sort”.
   - Na taj način možemo kasnije napraviti paralelizaciju i uporediti performanse.
   
================================================================================
*/

void std_sort_gpu_naive(std::vector<int>& h_arr) {
    // Kopiraj niz sa CPU-a na GPU
    thrust::device_vector<int> d_arr(h_arr.begin(), h_arr.end());

    // Sortiraj na GPU-u (Thrust radi na GPU)
    thrust::sort(d_arr.begin(), d_arr.end());

    // Vrati rezultat sa GPU-a na CPU
    thrust::copy(d_arr.begin(), d_arr.end(), h_arr.begin());
}

int main(int argc, char* argv[]) {
    return run_sort("std_sort", "naive_gpu", std_sort_gpu_naive, argc, argv);
}
