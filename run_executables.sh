#!/bin/bash
# run_executables.sh
# Runs all sequential and parallel CPU executables and measures execution time
# Still in development - modify as needed

# Rebuild all executables
mkdir -p build
cd build
#cmake ..       # Configure project
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ ..
make -j        # Build all targets
cd ..

BUILD_DIR="build"
DATA_DIR="data"

mkdir -p ${DATA_DIR}

echo "Algorithm,Mode,ArraySize,TimeMs" > ${DATA_DIR}/benchmark.csv

for exe in ${BUILD_DIR}/sequential_* ${BUILD_DIR}/parallel_cpu_*; do
    if [[ -x "$exe" ]]; then
        exe_name=$(basename $exe)
        echo "Running $exe_name ..."
        # Example: run the executable with an array size argument
        # You can modify your C++ main() to accept array size as argv[1]
        size=1000000
        time_ms=$($exe $size)  # Your executable should print execution time in ms
        echo "${exe_name},${size},${time_ms}" >> ${DATA_DIR}/benchmark.csv
    fi
done

echo "Benchmark finished. Results saved in ${DATA_DIR}/benchmark.csv"
