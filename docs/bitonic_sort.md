## Optimizacija sekvencijalne Bitonic Sort implementacije

### Promjene i optimizacije u odnosu na naivnu verziju:

- **Hybrid pristup sa insertion sort-om za male blokove (`≤32` elemenata)**  
  - Mala sekvenca se sortira pomoću insertion sort-a umjesto da prolazi kroz cijelu bitoničku mrežu.
  - **Efekat:** smanjen overhead rekurzije i brže sortiranje malih nizova, bolja iskorištenost cache memorije.

- **Efikasno pomjeranje elemenata u insertion sort-u (`memmove`)**  
  - Umjesto element-po-element pomjeranja, `memmove` se koristi za pomjeranje cijelog bloka elemenata odjednom.
  - **Efekat:** manji broj write operacija, bolja lokalnost cache-a i brži kod za male blokove.

- **Branchless compare & swap (`compAndSwap`)**  
  - Eliminisani su klasični `if` izrazi unutar hot-loop-a i zamijenjeni inline ternarnim operacijama.
  - **Efekat:** smanjen branch misprediction penalty i bolja iskorištenost CPU pipeline-a.

- **Iterativni `bitonicMerge` umjesto rekurzije**  
  - Merge faza je implementirana iterativno, bez rekurzije, koristeći dvije ugniježdene petlje za size i half.
  - **Efekat:** smanjen call overhead, veća throughput i bolje iskorištenje cache memorije.

- **Povećani insertion sort threshold (`32` elemenata)**  
  - Threshold je povećan u odnosu na klasičnu implementaciju (obično 16) kako bi se dodatno smanjio broj rekurzivnih poziva i overhead malih blokova.
  - **Efekat:** optimizacija vremena izvršavanja na srednje velikim blokovima.

- **Opšti efekat optimizacija:**  
  - Manje grananja i funkcijskih poziva unutar hot loop-a.
  - Bolja lokalnost memorije i cache-a.
  - Brže sortiranje za male, srednje i velike nizove.
  - Kompatibilno sa postojećim `run_sort` sistemom.

---


**Zaključak:**  
Ova optimizovana sekvencijalna Bitonic Sort implementacija pruža **značajno ubrzanje** u odnosu na naivnu verziju zahvaljujući:

- Hybrid pristupu (insertion sort za male blokove)
- Efikasnom pomjeranju elemenata (`memmove`)
- Branchless hot loop operacijama
- Iterativnom merge-u

Ova verzija je odlična osnova za dalju **paralelizaciju** (CPU/GPU) ili mikro-optimizacije, bez mijenjanja osnovne logike algoritma.
