# ============================================================================
# Python plotting setup and execution
# ============================================================================

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
if (-not (Test-Path $VenvDir)) {
    Write-Host "📦 Creating Python virtual environment..."
    python -m venv $VenvDir
    Write-Host "✅ Virtual environment created"
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..."
& "$VenvDir\Scripts\Activate.ps1"

# Install/update dependencies
if (Test-Path "requirements.txt") {
    Write-Host "📥 Installing Python dependencies..."
    python -m pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    Write-Host "✅ Dependencies installed"
} else {
    Write-Host "⚠️  requirements.txt not found. Installing basic packages..."
    pip install --quiet matplotlib pandas numpy
}

# Run plotting script
if (Test-Path $PlotsScript) {
    Write-Host ""
    Write-Host "🎨 Generating plots..."
    python $PlotsScript
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host ("=" * 60)
        Write-Host "✅ SUCCESS! All plots generated."
        Write-Host "📂 Check the plots/ directory for visualization results."
        Write-Host ("=" * 60)
    } else {
        Write-Host "⚠️  Plot generation encountered errors."
    }
} else {
    Write-Host "⚠️  Plotting script not found: $PlotsScript"
}

# Note: Virtual environment will be deactivated when script exits

