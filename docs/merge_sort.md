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
