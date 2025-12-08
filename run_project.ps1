# run_executables.ps1
# Rebuilds all executables, runs only changed ones, and logs results
# Copies results of unchanged executables from previous benchmark

param(
    [string[]]$Algorithms = @()  # Algorithm filters: --quick, --merge, --bitonic, --std, --seq-naive, --seq-optimized, --cpu-parallel
)

$ConfigFile = "config.yaml"
$ExampleFile = "config.yaml.example"

if (-not (Test-Path $ConfigFile)) {
    Write-Host "⚠️  $ConfigFile not found, using $ExampleFile instead."
    $ConfigFile = $ExampleFile
}

# Check if YAML support exists (PowerShell 7+ has ConvertFrom-Yaml)
if (-not (Get-Command ConvertFrom-Yaml -ErrorAction SilentlyContinue)) {
    Write-Host "⚠️  ConvertFrom-Yaml not available. Install module:"
    Write-Host "   Install-Module powershell-yaml -Scope CurrentUser"
    exit 1
}

# Load config
$config = Get-Content $ConfigFile | ConvertFrom-Yaml

$Sizes = $config.sizes
$Types = $config.types
$Seed = if ($null -ne $config.seed) { [int]$config.seed } else { 12345 }
$PrintArray = if ($null -ne $config.print_array) { [bool]$config.print_array } else { $false }

Write-Host "📘 Loaded configuration:"
Write-Host "  Sizes: $($Sizes -join ', ')"
Write-Host "  Types: $($Types -join ', ')"
Write-Host "  Seed: $Seed"
Write-Host "  Print array: $PrintArray"

if ($Algorithms.Count -gt 0) {
    Write-Host "  Algorithm filters: $($Algorithms -join ', ')"
}

# Function to check if an executable matches the algorithm filters
function Test-AlgorithmFilter {
    param([string]$ExeName)
    
    # If no filters specified, run all
    if ($Algorithms.Count -eq 0) {
        return $true
    }
    
    foreach ($filter in $Algorithms) {
        $filterLower = $filter.ToLower().TrimStart('-')
        $exeLower = $ExeName.ToLower()
        
        # Check algorithm type filters
        if ($filterLower -eq "quick" -and $exeLower -like "*quick*") { return $true }
        if ($filterLower -eq "merge" -and $exeLower -like "*merge*") { return $true }
        if ($filterLower -eq "bitonic" -and $exeLower -like "*bitonic*") { return $true }
        if ($filterLower -eq "std" -and $exeLower -like "*std*") { return $true }
        if ($filterLower -eq "radix" -and $exeLower -like "*radix*") { return $true }
        
        # Check implementation category filters
        if ($filterLower -eq "seq-naive" -and $exeLower -like "sequential_naive*") { return $true }
        if ($filterLower -eq "seq-optimized" -and $exeLower -like "sequential_optimized*") { return $true }
        if ($filterLower -eq "cpu-parallel" -and $exeLower -like "parallel_cpu*") { return $true }
        
        # Check exact name match
        if ($exeLower -like "*$filterLower*") { return $true }
    }
    
    return $false
}

$DataDir = "data"
$HashFile = "$DataDir\last_run_hashes.txt"
$OutFile = "$DataDir\benchmark.csv"

# Always build both versions
$BuildDirNormal = "build"
$BuildDirBenchmark = "build_vtune_benchmarking"

Write-Host ""
Write-Host "📁 Building both versions:"
Write-Host "   • Normal (with verification): $BuildDirNormal"
Write-Host "   • Benchmark (VTune ready): $BuildDirBenchmark"

# Detect number of CPU cores
$Cores = [Environment]::ProcessorCount

# Function to build executables in a directory
function Build-Executables {
    param(
        [string]$BuildDir,
        [string]$BenchmarkFlag,
        [string]$Description
    )
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════"
    Write-Host "🔧 Building $Description"
    Write-Host "═══════════════════════════════════════════════════════════"
    
    if (Test-Path $BuildDir) { 
        Write-Host "🗑️  Removing existing build directory..."
        for ($i=0; $i -lt 5; $i++) {
            try {
                Remove-Item -Recurse -Force $BuildDir
                break
            } catch {
                Start-Sleep -Milliseconds 200
            }
        }
    }
    if (-Not (Test-Path $BuildDir)) { New-Item -ItemType Directory -Path $BuildDir | Out-Null }
    
    Push-Location $BuildDir
    
    Write-Host "🧱 Configuring CMake build..."

    # 1️⃣ Preferred configuration (GCC + Ninja)
    Write-Host "🔧 Trying preferred configuration (Ninja + GCC)..."
    $preferredCommand = "cmake -G `"Ninja`" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=C:/msys64/mingw64/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe -DUSE_OPENMP=ON $BenchmarkFlag -DCMAKE_CXX_FLAGS=`"-O3 -march=native -mavx2 -mavx512f -fopenmp`" .."
    Invoke-Expression $preferredCommand

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Preferred configuration succeeded!"
    } else {
        Write-Host "❌ Preferred configuration failed, trying alternative generators..."
        Remove-Item CMakeCache.txt -ErrorAction SilentlyContinue

        # 2️⃣ Fallback options in order of preference
        $generators = @(
            @("MinGW Makefiles", "cmake .. -G `"MinGW Makefiles`" -DCMAKE_BUILD_TYPE=Release $BenchmarkFlag"),
            @("Ninja with Clang", "cmake .. -G `"Ninja`" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release $BenchmarkFlag"),
            @("Visual Studio Clang", "cmake .. -G `"Visual Studio 17 2022`" -A x64 -T ClangCL $BenchmarkFlag"),
            @("Default", "cmake .. -DCMAKE_BUILD_TYPE=Release $BenchmarkFlag")
        )

        $success = $false
        foreach ($generator in $generators) {
            $name = $generator[0]
            $command = $generator[1]
            Write-Host "Trying $name..."
            Invoke-Expression $command

            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Configuration successful with $name"
                $success = $true
                break
            } else {
                Write-Host "❌ $name failed, trying next..."
                Remove-Item CMakeCache.txt -ErrorAction SilentlyContinue
            }
        }

        if (-not $success) {
            Write-Host "❌ All CMake configuration attempts failed."
            Pop-Location
            return $false
        }
    }

    # 3️⃣ Build using all CPU cores
    Write-Host "⚙️ Building project with all available cores ($Cores)..."
    cmake --build . --config Release --parallel $Cores

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Build failed."
        Pop-Location
        return $false
    }

    # 4️⃣ Copy required DLLs to build directory for VTune and standalone execution
    Write-Host "📦 Copying runtime DLLs to build directory..."
    $dllsToCopy = @(
        "C:\msys64\mingw64\bin\libstdc++-6.dll",
        "C:\msys64\mingw64\bin\libgcc_s_seh-1.dll",
        "C:\msys64\mingw64\bin\libwinpthread-1.dll",
        "C:\msys64\mingw64\bin\libgomp-1.dll"  # OpenMP runtime
    )
    foreach ($dll in $dllsToCopy) {
        if (Test-Path $dll) {
            Copy-Item $dll -Destination . -Force
            Write-Host "  ✅ Copied $(Split-Path $dll -Leaf)"
        }
    }

    Pop-Location
    Write-Host "✅ Build completed successfully!"
    return $true
}

# Build both versions
$normalSuccess = Build-Executables -BuildDir $BuildDirNormal -BenchmarkFlag "-DBENCHMARK_MODE=OFF" -Description "NORMAL executables (with verification)"
$benchmarkSuccess = Build-Executables -BuildDir $BuildDirBenchmark -BenchmarkFlag "-DBENCHMARK_MODE=ON" -Description "BENCHMARK executables (VTune ready)"

if (-not $normalSuccess -or -not $benchmarkSuccess) {
    Write-Host ""
    Write-Host "❌ One or more builds failed."
    exit 1
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════"
Write-Host "✅ Both versions built successfully!"
Write-Host "   📂 Normal: $BuildDirNormal (will be executed)"
Write-Host "   📂 Benchmark: $BuildDirBenchmark (ready for VTune)"
Write-Host "═══════════════════════════════════════════════════════════"

if (-Not (Test-Path $DataDir)) { New-Item -ItemType Directory -Path $DataDir | Out-Null }

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════"
Write-Host "🚀 Running normal executables from: $BuildDirNormal"
Write-Host "═══════════════════════════════════════════════════════════"

# Backup current benchmark if exists
if (Test-Path $OutFile) {
    $timestamp = Get-Date -UFormat %s
    Move-Item $OutFile "$DataDir\benchmark_backup_$timestamp.csv"
}

# Find latest benchmark backup
$LatestBackup = Get-ChildItem -Path $DataDir -Filter "benchmark_backup_*.csv" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1

# Initialize new CSV
"Algorithm,ArraySize,ArrayType,TimeMs" | Out-File $OutFile

# Load old hashes
$oldHashes = @{}
if (Test-Path $HashFile) {
    Get-Content $HashFile | ForEach-Object {
        $parts = $_ -split " "
        if ($parts.Length -eq 2) {
            $oldHashes[$parts[0]] = $parts[1]
        }
    }
}

# Temporary hash storage
$newHashes = @{}

# Compute new hashes (only from normal build directory)
Get-ChildItem $BuildDirNormal -File | Where-Object { $_.Name -match '^(sequential_|parallel_cpu_)' } | ForEach-Object {
    $exePath = $_.FullName
    $exeName = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
    $hash = (Get-FileHash $exePath -Algorithm MD5).Hash
    $newHashes[$exePath] = $hash

    $oldHash = if ($oldHashes.ContainsKey($exePath)) { $oldHashes[$exePath] } else { "" }
    
    # Check if this executable matches the algorithm filters
    $matchesFilter = Test-AlgorithmFilter -ExeName $exeName
    
    if ($oldHash -eq $hash) {
        if ($matchesFilter) {
            Write-Host "⏭️  Skipping unchanged (matches filter): $exeName"
        } else {
            Write-Host "⏭️  Skipping (no filter match): $exeName"
        }
        if ($LatestBackup) {
            $lines = Get-Content $LatestBackup.FullName | Where-Object { $_ -match "^$exeName," }
            if ($lines) {
                $lines | Out-File $OutFile -Append
            }
        }
        return
    }
    
    if (-not $matchesFilter) {
        Write-Host "⏭️  Skipping (no filter match): $exeName"
        # Copy previous results even if executable changed but doesn't match filter
        if ($LatestBackup) {
            $lines = Get-Content $LatestBackup.FullName | Where-Object { $_ -match "^$exeName," }
            if ($lines) {
                $lines | Out-File $OutFile -Append
            }
        }
        return
    }

    Write-Host "🚀 Running updated executable: $exeName"
    foreach ($type in $Types) {
        foreach ($size in $Sizes) {
            Write-Host "  -> Type: $type | Size: $size"
            # build arguments: size, type, seed, optional flags
            $args = @($size, $type, $Seed)
            if ($PrintArray) { $args += "--print-array" }
            # Note: benchmark mode is now compile-time, no runtime flag needed

            # Run executable and capture output
            $output = & $exePath @args 2>&1

            # Print full output and parse results
            # Print full output to console so array is visible for debugging
            Write-Host "----- $exeName output start -----"
            Write-Host $output
            Write-Host "----- $exeName output end -----"

            if ($output -match "Error: Array is NOT sorted") {
                Write-Host "❗ Skipping invalid result for $exeName (unsorted output)"
                return
            }
            # Parse time (still works even if array printed)
            $timeLine = ($output | Select-String -Pattern "Execution time").Line

            if ($timeLine) {
                # take last token as ms (your existing format)
                $tokens = $timeLine -split '\s+'
                $timeMs = $tokens[-1]
                "$exeName,$size,$type,$timeMs" | Out-File $OutFile -Append
            } else {
                Write-Host "⚠️  Warning: Could not parse time output for $exeName ($size, $type)"
            }
        }
    }
}

# Save new hashes
$newHashes.GetEnumerator() | ForEach-Object {
    "$($_.Key) $($_.Value)"
} | Out-File $HashFile

Write-Host "✅ Benchmark finished. Results saved in $OutFile"

# ============================================================================
# Python plotting setup and execution
# ============================================================================

# Detect Windows Python (skip MSYS2 Python)
$winPython = Get-Command python -ErrorAction SilentlyContinue | Where-Object { $_.Source -notmatch "msys64" }
if (-not $winPython) {
    Write-Host "⚠️  Could not find Windows Python 3.10+. Skipping plotting."
    $RunPlots = $false
} else {
    $RunPlots = $true
    $PythonExe = $winPython.Source
}

Write-Host ""
Write-Host ("=" * 60)
Write-Host "📊 Setting up Python environment for plotting..."
Write-Host ("=" * 60)

$VenvDir = "venv"
$PlotsScript = "plots/plot_results.py"

# Check if Python is available
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "⚠️  Python not found. Skipping plot generation."
    Write-Host "   Install Python 3.10+ to enable visualization."
    exit 0
}

$pythonVersion = & python --version 2>&1
Write-Host "✅ Found $pythonVersion"

# Create virtual environment if it doesn't exist
if (-not (Test-Path $VenvDir) -and $RunPlots) {
    Write-Host "📦 Creating Python virtual environment..."
    & $PythonExe -m venv $VenvDir
    Write-Host "✅ Virtual environment created"
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..."
# If activation fails, just call Python from venv directly
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

# Install dependencies
if ((Test-Path "requirements.txt") -and $RunPlots) {
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r requirements.txt
} elseif ($RunPlots) {
    & $VenvPython -m pip install matplotlib pandas numpy
}

# Run plotting script
if ($RunPlots -and (Test-Path $PlotsScript)) {
    Write-Host ""
    Write-Host "🎨 Generating plots..."
    & $VenvPython $PlotsScript

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host ("=" * 60)
        Write-Host "✅ SUCCESS! All plots generated."
        Write-Host "📂 Check the plots/ directory for visualization results."
        Write-Host ("=" * 60)
    } else {
        Write-Host "⚠️  Plot generation encountered errors."
    }
}

# Note: Virtual environment will be deactivated when script exits

