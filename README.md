# Sorting Algorithms Parallelization

## Project Purpose

This project compares **multiple sorting algorithms** implemented in three ways:

1. **Sequential (baseline)** – standard single-threaded implementation
2. **Parallel CPU** – multi-core CPU parallelization
3. **Parallel GPU** – GPU parallelization using PyTorch or other GPU frameworks

The goal is to **benchmark execution times**, **visualize speedups**, and explore differences between CPU and GPU parallelization across different hardware setups.

---

## Folder Structure

```
.
├── README.md             # This developer guide
├── data                  # Input arrays and benchmark CSVs
├── sequential            # Sequential sorting algorithm implementations
├── parallel_cpu           # CPU-parallel sorting implementations
├── parallel_gpu           # GPU-parallel sorting implementations
├── plots                 # Generated performance plots
```

---

## Dependencies and Virtual Environment

### Python Environment

* Python 3.10+ recommended
* Create a virtual environment to isolate project dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/MacOS
venv\Scripts\activate     # On Windows
```

* Install required packages:

```bash
pip install -r requirements.txt
```

* Deactivate environment when done:

```bash
deactivate
```

> Using a virtual environment ensures consistent dependencies across different developers' machines.

---

## Running Sequential Implementations

1. Navigate to `sequential/`:

```bash
cd sequential
```

2. Run the Python script:

```bash
python merge_sort.py
python quick_sort.py
```

3. Scripts output execution time and optionally save results to `../data/`.

---

## Running Parallel CPU Implementations

1. Navigate to `parallel_cpu/`:

```bash
cd parallel_cpu
```

2. Run the Python script:

```bash
python parallel_merge_sort.py
```

3. Scripts use `multiprocessing` or `concurrent.futures` for parallel execution.
4. Results are saved in `../data/benchmark_cpu.csv`.

---

## Running Parallel GPU Implementations

1. Open the notebook or script in `parallel_gpu/`.
2. Ensure a GPU runtime is available (local GPU or cloud platform like Colab, Kaggle, etc.).
3. Run the sorting notebook or script:

```python
# Example
!python gpu_sort.py
```

4. Execution times are saved to `../data/benchmark_gpu.csv`.

> **Note:** GPU frameworks automatically use available GPU devices if properly configured.

---

## Benchmarking & Plotting

1. Navigate to the project root:

```bash
cd <project-root>
```

2. Run the benchmark/plot script:

```bash
python plot_results.py
```

3. Output plots are saved in the `plots/` directory and can be included in reports or presentations.

---

## Development Workflow

1. **Add new sorting algorithm**

   * Create a new file in the corresponding folder (`sequential/`, `parallel_cpu/`, or `parallel_gpu/`)
   * Follow existing conventions:

     * Function accepts a NumPy array input
     * Returns the sorted array
     * Returns execution time if benchmarked

2. **Benchmark new algorithm**

   * Add calls in `plot_results.py` or benchmarking scripts
   * Update CSV file names and plot labels

3. **Test on available hardware**

   * Verify correctness on small arrays
   * Ensure CPU and GPU versions produce identical outputs where applicable

4. **Version control**

   * Use Git to track changes
   * Keep `data/` folder lightweight — commit only small benchmark CSVs or update `.gitignore`

---

## Notes / Best Practices

* Use **array size ≥ 10⁵** to observe CPU parallel speedup.
* Use **array size ≥ 10⁶** to observe GPU speedup.
* Repeat each experiment 5–10 times and average results for accuracy.
* For GPU, synchronize operations (e.g., `torch.cuda.synchronize()`) before measuring execution time for accuracy.
* This guide is generalized for multiple developers; exact hardware and GPU availability may vary.

