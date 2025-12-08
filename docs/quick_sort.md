# Optimizacija naivne Quick Sort implementacije

## Promjene i optimizacije u odnosu na naivnu verziju:

- **Insertion sort za male segmente (`≤32` elemenata)**  
  Za male podnizove koristi se `insertion_sort`, koji je brži od quick sort-a na malim nizovima zbog smanjenja overhead-a rekurzije.  
  **Efekat:** brže sortiranje malih blokova i smanjenje broja rekurzivnih poziva.

- **Median-of-three pivot izbor**  
  Pivot se bira kao medijan prvog, srednjeg i zadnjeg elementa segmenta.  
  **Efekat:** smanjuje šanse za loš odabir pivota na skoro sortiranom nizu, smanjujući dubinu rekurzije i broj swap operacija.

- **Tail recursion eliminacija**  
  Umjesto klasične rekurzije za oba segmenta, manji segment se sortira rekurzivno, a veći se obrađuje iterativno (`while` petlja).  
  **Efekat:** smanjuje upotrebu steka i izbjegava preveliku rekurziju za velike nizove.

- **Early exit ako je segment već sortiran**  
  Ako je prvi element segmenta ≤ zadnjemu elementu segmenta, sortiranje se preskače.  
  **Efekat:** smanjuje nepotrebne operacije za skoro sortirane ili segmentno sortirane nizove.

- **Opšti efekat optimizacija:**  
  - Smanjena dubina rekurzije i upotreba steka.  
  - Bolja performansa na skoro sortiranom nizu.  
  - Brže sortiranje malih segmenata.  
  - Robustnija selekcija pivota smanjuje šanse za loš worst-case scenario.

## Benchmark (vrijeme u ms)

| Algoritam | 10k | 5M | 50M | 500M |
|-----------|-----|----|-----|------|
| Naivni QuickSort | 1.94 | 428.12 | 4967.41 | 60123.8 |
| Optimizovani QuickSort | 0.62 | 208.73 | 2511.36 | 30784.2 |

**Otprilike ubrzanje:**
- 10k elemenata: ~3×  
- 5M elemenata: ~2×  
- 50M elemenata: ~2×  
- 500M elemenata: ~2×

**Zaključak:**  
Optimizovana sekvencijalna implementacija Quick Sort-a pruža značajno ubrzanje, posebno za male i skoro sortirane nizove.  
Ovo je **praktična osnova za paralelizaciju** i dalje optimizacije (CPU/GPU).

...