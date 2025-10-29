# Still in development - modify as needed

# run_executables.ps1
# Rebuilds the project and runs all sequential and parallel CPU executables

$BuildDir = "build"
$DataDir = "data"

# Ensure build and data directories exist
if (-Not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir
}
if (-Not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir
}

# Rebuild project
Write-Host "Configuring and building project..."
Push-Location $BuildDir
cmake ..                 # Configure
cmake --build . --config Release  # Build all targets
Pop-Location

# Prepare CSV file
$BenchmarkFile = "$DataDir\benchmark.csv"
"Algorithm,Mode,ArraySize,TimeMs" | Out-File $BenchmarkFile

# Run executables
Get-ChildItem $BuildDir\* | Where-Object { $_.Extension -eq ".exe" } | ForEach-Object {
    $exeName = $_.Name
    Write-Host "Running $exeName ..."
    $size = 1000000
    $timeMs = & $_.FullName $size  # Executable should print execution time in ms
    "$exeName,$size,$timeMs" | Out-File $BenchmarkFile -Append
}

Write-Host "Benchmark finished. Results saved in $BenchmarkFile"
