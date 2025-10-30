# run_executables.ps1
# Rebuilds all executables, runs only changed ones, and logs results
# Copies results of unchanged executables from previous benchmark

$BuildDir = "build"
$DataDir = "data"
$HashFile = "$DataDir\last_run_hashes.txt"
$OutFile = "$DataDir\benchmark.csv"
$Sizes = @(10000, 5000000, 50000000, 500000000)

# Detect number of CPU cores
$Cores = [Environment]::ProcessorCount

Write-Host "🔧 Rebuilding executables..."
if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
New-Item -ItemType Directory -Path $BuildDir | Out-Null
Push-Location $BuildDir
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -- /m:$Cores
Pop-Location

if (-Not (Test-Path $DataDir)) { New-Item -ItemType Directory -Path $DataDir | Out-Null }

# Find latest benchmark backup
$LatestBackup = Get-ChildItem -Path $DataDir -Filter "benchmark_backup_*.csv" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1

# Backup current benchmark if exists
if (Test-Path $OutFile) {
    $timestamp = Get-Date -UFormat %s
    Move-Item $OutFile "$DataDir\benchmark_backup_$timestamp.csv"
}

# Initialize new CSV
"Algorithm,ArraySize,TimeMs" | Out-File $OutFile

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

# Compute new hashes
Get-ChildItem $BuildDir -File | Where-Object { $_.Name -match '^(sequential_|parallel_cpu_)' } | ForEach-Object {
    $exePath = $_.FullName
    $exeName = $_.Name
    $hash = (Get-FileHash $exePath -Algorithm MD5).Hash
    $newHashes[$exePath] = $hash

    $oldHash = if ($oldHashes.ContainsKey($exePath)) { $oldHashes[$exePath] } else { "" }

    if ($oldHash -eq $hash) {
        Write-Host "⏭️  Skipping unchanged: $exeName"
        if ($LatestBackup) {
            $lines = Get-Content $LatestBackup.FullName | Where-Object { $_ -match "^$exeName," }
            if ($lines) {
                $lines | Out-File $OutFile -Append
            }
        }
        return
    }

    Write-Host "🚀 Running updated executable: $exeName"
    foreach ($size in $Sizes) {
        Write-Host "  -> Size: $size"
        $output = & $exePath $size 2>&1
        $timeLine = ($output | Select-String -Pattern "Execution time").Line
        if ($timeLine) {
            $timeMs = ($timeLine -split " ")[-1]
            "$exeName,$size,$timeMs" | Out-File $OutFile -Append
        } else {
            Write-Host "⚠️  Warning: Could not parse time output for $exeName ($size)"
        }
    }
}

# Save new hashes
$newHashes.GetEnumerator() | ForEach-Object {
    "$($_.Key) $($_.Value)"
} | Out-File $HashFile

Write-Host "✅ Benchmark finished. Results saved in $OutFile"
