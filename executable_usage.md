# Korištenje Izvršnih (EXE) Fajlova

## Pregled

Ovaj dokument opisuje parametre komandne linije za sve izvršne fajlove u projektu i kako upravljati njihovim izvršavanjem.

---

## Struktura Parametara

Svi izvršni fajlovi koriste istu strukturu parametara komandne linije:

```
./ime_executable [veličina_niza] [tip_niza] [seed] [opcija_ispisa]
```

### Parametri

| Pozicija | Parametar | Opis | Zadana vrijednost |
|----------|-----------|------|-------------------|
| 1 | `veličina_niza` | Broj elemenata u nizu za sortiranje | 134217728 (128 miliona) |
| 2 | `tip_niza` | Tip generiranog niza (vidi dolje) | `random` |
| 3 | `seed` | Seed za generator slučajnih brojeva | 12345 |
| 4 | `opcija_ispisa` | Flag za ispis niza | ne ispisuje se |

---

## Tipovi Niza

Dostupni tipovi niza koje možete proslijediti kao drugi parametar:

| Tip | Opis |
|-----|------|
| `random` | Potpuno nasumični niz sa vrijednostima 0 - 1.000.000 |
| `sorted` | Već sortirani niz (0, 1, 2, ..., n-1) |
| `reversed` | Obrnuto sortirani niz (n, n-1, ..., 2, 1) |
| `nearly_sorted` | Skoro sortirani niz (1% elemenata je zamijenjeno) |
| `few_unique` | Niz sa samo 10 jedinstvenih vrijednosti |

---

## Opcije Ispisa Niza

Četvrti parametar omogućuje ispis niza prije i poslije sortiranja (prvih 200 elemenata):

| Vrijednost | Opis |
|------------|------|
| `--print-array` | Ispisuje niz |
| `print` | Ispisuje niz |
| `-p` | Ispisuje niz |

---

## Primjeri Pokretanja

### Osnovni primjer
```bash
# Sortiraj 10.000 nasumičnih elemenata
./build/sequential_naive_quick_sort 10000 random
```

### Sa svim parametrima
```bash
# Sortiraj 50.000 obrnutih elemenata sa seed-om 42
./build/sequential_naive_merge_sort 50000 reversed 42
```

### Sa ispisom niza
```bash
# Sortiraj 1.000 elemenata i ispiši niz prije i poslije sortiranja
./build/sequential_naive_bitonic_sort 1000 random 12345 --print-array
```

### Skoro sortirani niz
```bash
# Testiraj performanse na skoro sortiranom nizu
./build/parallel_cpu_quick_sort 1000000 nearly_sorted 42
```

---

## Lista Izvršnih Fajlova

### Sekvencijalni Naivni Algoritmi (`sequential_naive/`)
| Fajl | Opis |
|------|------|
| `sequential_naive_quick_sort` | Osnovni rekurzivni quick sort |
| `sequential_naive_merge_sort` | Osnovni rekurzivni merge sort |
| `sequential_naive_bitonic_sort` | Osnovni bitonički sort |
| `sequential_naive_std_sort` | `std::sort` sa sekvencijalnom politikom |

### Sekvencijalni Optimizirani Algoritmi (`sequential_optimized/`)
| Fajl | Opis |
|------|------|
| `sequential_optimized_quick_sort` | Optimizirani quick sort (median-of-three, insertion sort fallback) |
| `sequential_optimized_merge_sort` | Iterativni merge sort sa ping-pong bufferom |
| `sequential_optimized_bitonic_sort` | Optimizirani bitonički sort |
| `sequential_optimized_std_sort` | `std::sort` sa sekvencijalnom politikom |

### Paralelni CPU Algoritmi (`parallel_cpu/`)
| Fajl | Opis |
|------|------|
| `parallel_cpu_quick_sort` | Paralelni quick sort (OpenMP + AVX) |
| `parallel_cpu_merge_sort` | Paralelni merge sort (OpenMP tasks + Merge Path + AVX2) |
| `parallel_cpu_bitonic_sort` | Paralelni bitonički sort (OpenMP) |
| `parallel_cpu_std_sort` | `std::sort` sa paralelnom politikom |

---

## Izlaz Programa

Kada se program pokrene (u normalnom modu), ispisuje sljedeće informacije:

```
Algorithm: quick_sort
Mode: sequential
Array size: 10000
Array type: random
Execution time (ms): 1.234
Verification: sorted = true
```

Rezultati se također automatski dodaju u `data/benchmark.csv` fajl u formatu:
```
algorithm,mode,array_size,array_type,time_ms
```

---

## Benchmark Mode

Projekt podržava dva načina kompilacije:

### Normal Mode (zadano)
- Uključuje mjerenje vremena
- Verificira da je niz sortiran
- Ispisuje rezultate na konzolu
- Sprema rezultate u CSV

### Benchmark Mode (za VTune profiliranje)
- Samo sortiranje bez dodatnih operacija
- Minimizira overhead za čista mjerenja performansi
- Koristi se sa alatima poput Intel VTune

Za više detalja o benchmark modu, pogledajte [benchmark_mode.md](benchmark_mode.md).

---

## Korištenje Skripti za Automatsko Pokretanje

Umjesto ručnog pokretanja svakog izvršnog fajla, možete koristiti priložene skripte:

### Linux/macOS
```bash
bash run_project.sh
```

### Windows (MSYS2 MinGW64 Shell)
```powershell
powershell .\run_project.ps1
```

### Filtriranje Algoritama
Skripte podržavaju filtriranje algoritama:
```bash
# Samo quick sort algoritmi
bash run_project.sh --quick

# Samo paralelni CPU algoritmi
bash run_project.sh --cpu-parallel

# Kombinacija filtera
bash run_project.sh --merge --bitonic
```

Dostupni filteri:
- `--quick` - Quick sort algoritmi
- `--merge` - Merge sort algoritmi
- `--bitonic` - Bitonički sort algoritmi
- `--std` - std::sort algoritmi
- `--seq-naive` - Sekvencijalni naivni
- `--seq-optimized` - Sekvencijalni optimizirani
- `--cpu-parallel` - Paralelni CPU

---

## Konfiguracija putem config.yaml

Za automatske skripte, konfigurirajte parametre u `config.yaml` fajlu:

```yaml
sizes: [16384, 8388608, 67108864]   # Veličine nizova za testiranje
types: [random, sorted, reversed, nearly_sorted, few_unique]  # Tipovi nizova
seed: 42                            # Seed za generator slučajnih brojeva
print_array: false                  # Ispis niza (držati na false)
```

Kopirajte `config.yaml.example` u `config.yaml` i prilagodite prema potrebama.

---

## Napomene

1. **Veličina niza**: Za benchmark testove preporučuje se minimalno 10^5 elemenata
2. **Reprodukcija**: Korištenje istog seed-a garantuje iste nasumične nizove
3. **Memorija**: Veliki nizovi (>100 miliona elemenata) zahtijevaju značajnu RAM memoriju
4. **Bitonički sort**: Zahtijeva da veličina niza bude stepen broja 2 (npr. 1024, 2048, 4096...)
