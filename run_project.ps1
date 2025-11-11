# run_executables.ps1
# Rebuilds all executables using GCC, runs only changed ones, and logs results

$BuildDir = "build_gcc"
$DataDir = "data"
$HashFile = "$DataDir\last_run_hashes.txt"
$OutFile = "$DataDir\benchmark.csv"
$Sizes = @(10000, 5000000, 50000000, 500000000)
$Types = @("random", "sorted", "reversed", "nearly_sorted", "few_unique")

# Detect number of CPU cores
$Cores = [Environment]::ProcessorCount

Write-Host "Rebuilding executables using GCC..."
if (Test-Path $BuildDir) { 
    # pokušaj nekoliko puta zbog "used by another process" grešaka
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
# FORCE GCC generator
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=C:/msys64/mingw64/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe -DUSE_OPENMP=ON -DCMAKE_CXX_FLAGS="-O3 -march=native -mavx2 -mavx512f -fopenmp" ..
cmake --build . --config Release #-- /m:$Cores
Pop-Location

if (-Not (Test-Path $DataDir)) { New-Item -ItemType Directory -Path $DataDir | Out-Null }

# Backup previous benchmark
$LatestBackup = Get-ChildItem -Path $DataDir -Filter "benchmark_backup_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (Test-Path $OutFile) {
    $timestamp = Get-Date -UFormat %s
    Move-Item $OutFile "$DataDir\benchmark_backup_$timestamp.csv"
}

# Initialize CSV
"Algorithm,ArraySize,ArrayType,TimeMs" | Out-File $OutFile

# Load old hashes
$oldHashes = @{}
if (Test-Path $HashFile) {
    Get-Content $HashFile | ForEach-Object {
        $parts = $_ -split " "
        if ($parts.Length -eq 2) { $oldHashes[$parts[0]] = $parts[1] }
    }
}

$newHashes = @{}

# Run executables
Get-ChildItem $BuildDir -File | Where-Object { $_.Name -match '^(sequential_|parallel_cpu_)' } | ForEach-Object {
    $exePath = $_.FullName
    $exeName = $_.Name
    $hash = (Get-FileHash $exePath -Algorithm MD5).Hash
    $newHashes[$exePath] = $hash

    $oldHash = if ($oldHashes.ContainsKey($exePath)) { $oldHashes[$exePath] } else { "" }

    if ($oldHash -eq $hash) {
        Write-Host "Skipping unchanged executable: $exeName"
        if ($LatestBackup) {
            $lines = Get-Content $LatestBackup.FullName | Where-Object { $_ -match "^$exeName," }
            if ($lines) { $lines | Out-File $OutFile -Append }
        }
        return
    }

    Write-Host "Running updated executable: $exeName"
    foreach ($type in $Types) {
        foreach ($size in $Sizes) {
            Write-Host ("  Type: {0} | Size: {1}" -f $type, $size)
            $output = & $exePath $size $type 2>&1
            $timeLine = ($output | Select-String -Pattern "Execution time").Line
            if ($timeLine) {
                $timeMs = ($timeLine -split " ")[-1]
                "$exeName,$size,$type,$timeMs" | Out-File $OutFile -Append
            } else {
                Write-Host ("Warning: Could not parse time output for {0} ({1}, {2})" -f $exeName, $size, $type)
            }
        }
    }
}

# Save new hashes
$newHashes.GetEnumerator() | ForEach-Object { "$($_.Key) $($_.Value)" } | Out-File $HashFile

Write-Host "Benchmark finished. Results saved in $OutFile"

# ======================
# Python plotting
# ======================

$VenvDir = "venv"
$PlotsScript = "plots/plot_results.py"

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Python not found. Skipping plot generation."
    exit 0
}

$pythonVersion = & python --version 2>&1
Write-Host "Found $pythonVersion"

if (-not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
}

& "$VenvDir\Scripts\Activate.ps1"

if (Test-Path "requirements.txt") {
    python -m pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
} else {
    pip install --quiet matplotlib pandas numpy
}

if (Test-Path $PlotsScript) {
    python $PlotsScript
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Plots generated successfully. Check the plots/ directory."
    } else {
        Write-Host "Plot generation encountered errors."
    }
} else {
    Write-Host "Plotting script not found: $PlotsScript"
}
