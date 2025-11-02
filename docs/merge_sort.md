## Optimizacija naivne Merge Sort implementacije

### Promjene i optimizacije u odnosu na naivnu verziju:

- **Jedinstveni privremeni buffer (`temp[]`)**  
  Umjesto da se pri svakom merge-u kreiraju novi `std::vector` podnizovi, koristi se jedan globalni buffer koji se prosljeđuje rekurzivno.  
  **Efekat:** manja memorijska alokacija, bolja cache lokalnost.

- **Insertion sort za male segmente (`≤32` elemenata)**  
  Za male segmente rekurzije se koristi `insertion_sort`, koji je brži od merge sort-a na malim nizovima. Vrijednost konstante odredjena nakon nekoliko tesstiranja.
  **Efekat:** smanjenje overhead-a rekurzije i brže sortiranje malih blokova.

- **Preskakanje merge-a kada su segmenti već sortirani**  
  Provjerava se da li je zadnji element lijevog segmenta manji ili jednak prvom elementu desnog (`arr[m] <= arr[m+1]`). Ako jeste, merge se preskače.  
  **Efekat:** smanjuje broj nepotrebnih operacija, posebno za skoro sortirane ili segmentno sortirane nizove.

- **Efikasno kopiranje blokova memorije (`memcpy`)**  
  Ostatak segmenta se kopira blokovima umjesto element-po-element.  
  **Efekat:** brže kopiranje podataka, manji CPU overhead.

- **Opšti efekat optimizacija:**  
  - Smanjen broj alokacija memorije.  
  - Bolja cache lokalnost i manje preskakanja memorije.  
  - Manje nepotrebnih merge operacija.  
  - Brže za male i skoro sortirane nizove.  

### Benchmark (vrijeme u ms)

| Algoritam | 10k | 5M | 50M | 500M |
|-----------|-----|----|-----|------|
| Naivni MergeSort | 2.05 | 469.23 | 5335.81 | 63924.5 |
| Optimizovani MergeSort | 0.84 | 228.05 | 2551.38 | 32449.5 |

**Otprilike ubrzanje:**
- 10k elemenata: ~2.5×  
- 5M elemenata: ~2×  
- 50M elemenata: ~2×  
- 500M elemenata: ~2×  

**Zaključak:**  
Optimizovana sekvencijalna implementacija je **praktična baza za paralelizaciju (CPU/GPU)** i pruža značajno ubrzanje u odnosu na naivnu verziju, posebno za velike i skoro sortirane nizove.




# Paralelni Merge Sort – OpenMP + Merge Path

## Opis implementacije
Ovo je **paralelna, Merge-Path optimizovana verzija Merge Sort-a**, koja koristi **OpenMP taskove** i **Merge Path** tehniku za efikasno paralelno spajanje (merge).  
Sekvencijalni dijelovi su optimizovani kao u sekvencijalnoj verziji (privremeni buffer, insertion sort za male segmente, preskakanje merge-a ako je već sortirano).

---

## Ključne optimizacije i promjene

- **Insertion sort za male segmente (`≤16` elemenata)**  
  Za male dijelove rekurzije koristi se `insertion_sort` jer je brži od merge sort-a na malim nizovima.

- **Jedinstveni privremeni buffer (`temp[]`)**  
  Cijeli niz se sortira koristeći **jedan privremeni buffer** koji se prosljeđuje kroz rekurziju, smanjujući memorijske alokacije i poboljšavajući cache lokalnost.

- **Preskakanje merge-a kada je segment već sortirano**  
  Ako je zadnji element lijevog segmenta ≤ prvom elementu desnog segmenta, merge se preskače.  
  **Efekat:** smanjuje broj nepotrebnih operacija, naročito za skoro sortirane nizove.

- **Merge Path paralelizacija**  
  - `parallel_merge` dijeli merge u nekoliko dijelova, jedan po thread-u.  
  - Svaki thread radi lokalni merge segmenta u `temp[]`, a zatim se kopira nazad u `arr[]`.  
  - Minimalna veličina segmenta za paralelizaciju je `MERGE_PATH_MIN_CHUNK` (npr. 1024 elementa) da se izbjegne overhead za male nizove.

- **Task-based rekurzija (OpenMP tasks)**  
  - Rekurzija se paralelizira samo ako je segment veći od `TASK_THRESHOLD` (~65536).  
  - Manji segmenti se sortiraju sekvencijalno kako bi se izbjegao overhead kreiranja taskova.

- **SIMD i cache-friendly kopiranje**  
  Iako je glavna petlja merge-a scalar (branch-heavy), ostaci segmenta se kopiraju blokovima u temp buffer, što je cache i SIMD-friendly.

---

## Struktura funkcija

| Funkcija | Svrha |
|----------|-------|
| `insertion_sort(int arr[], int l, int r)` | Sortira male segmente direktno |
| `merge_with_temp_scalar(int arr[], int l, int m, int r, int temp[])` | Scalar merge za male segmente |
| `merge_path_find_split(...)` | Računa gdje se segmenti lijevog i desnog niza dijele za paralelni merge |
| `parallel_merge(...)` | Paralelni merge koristeći Merge Path |
| `merge_sort_rec_parallel(...)` | Rekurzivni Merge Sort sa OpenMP taskovima |
| `merge_sort_opt_parallel(...)` | Wrapper koji kreira `temp[]` i pokreće paralelni merge sort |
| `merge_sort_wrapper(...)` | Adapter za test harness |
| `main(...)` | Poziva `run_sort` iz `main_template.hpp` |

---

## Main ideja
1. Ako je segment mali → **sekvencijalni insertion sort**.  
2. Ako je segment dovoljno veliki → **kreiraju se OpenMP taskovi** za lijevu i desnu polovinu.  
3. Nakon sortiranja polovica → **Merge Path paralelni merge** u temp buffer, pa kopiranje nazad.  
4. Za male nizove, sve ostaje sekvencijalno radi brzine i izbjegavanja overhead-a.

---

## Benchmark (vrijeme u ms)

| Algoritam | 10k | 5M | 50M | 500M |
|-----------|-----|----|-----|------|
| Sequential Naive MergeSort | 1.63 | 471.81 | 5146.9 | 62778 |
| Sequential Optimized MergeSort | 0.57 | 225.04 | 2531.16 | 33853.9 |
| **Parallel CPU MergeSort** | 3.14 | 34.87 | 322.18 | 9579.31 |

**Zaključak:**  
- Za male nizove (<~100k), paralelizacija je sporija zbog overhead-a OpenMP taskova i paralelnog merge-a.  
- Za velike nizove (>5M), paralelizacija daje **značajno ubrzanje** (10–30×), što potvrđuje efikasnost Merge Path i task-based strategije.
