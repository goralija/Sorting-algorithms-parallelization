#!/usr/bin/env python3
"""
Script for plotting sorting algorithm benchmark results.
Shows CPU parallelization speedup vs sequential implementation,
improvements compared to previous runs, and performance distribution by array size.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set matplotlib backend for non-interactive use
plt.switch_backend('Agg')

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
PLOTS_DIR = SCRIPT_DIR
BENCHMARK_FILE = DATA_DIR / 'benchmark.csv'


def load_benchmark_data(filepath):
    """Load benchmark CSV data."""
    if not filepath.exists():
        print(f"❌ Error: Benchmark file not found: {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} benchmark results from {filepath}")
        return df
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None


def categorize_algorithm(algo_name):
    """Categorize algorithm by implementation type."""
    if 'sequential_naive' in algo_name:
        return 'Sequential Naive'
    elif 'sequential_optimized' in algo_name:
        return 'Sequential Optimized'
    elif 'parallel_cpu' in algo_name:
        return 'Parallel CPU'
    elif 'parallel_gpu' in algo_name:
        return 'Parallel GPU'
    else:
        return 'Unknown'


def extract_algorithm_name(full_name):
    """Extract clean algorithm name from executable name."""
    # Remove prefix and extension
    name = full_name.replace('sequential_naive_', '')
    name = name.replace('sequential_optimized_', '')
    name = name.replace('parallel_cpu_', '')
    name = name.replace('parallel_gpu_', '')
    name = name.replace('.exe', '')
    return name


def plot_speedup_vs_sequential(df, output_path):
    """
    Plot CPU parallelization speedup compared to sequential implementation.
    Shows speedup ratio for each algorithm and array size.
    """
    print("\n📊 Generating speedup vs sequential plot...")
    
    # Add categories
    df['Category'] = df['Algorithm'].apply(categorize_algorithm)
    df['AlgoName'] = df['Algorithm'].apply(extract_algorithm_name)
    
    # Get sequential baseline (prefer optimized, fallback to naive)
    sequential_df = df[df['Category'].str.contains('Sequential')]
    parallel_df = df[df['Category'] == 'Parallel CPU']
    
    if sequential_df.empty or parallel_df.empty:
        print("⚠️  Not enough data for speedup comparison (need both sequential and parallel results)")
        return
    
    # For each parallel algorithm, compute speedup
    speedup_data = []
    
    for algo_name in parallel_df['AlgoName'].unique():
        for array_size in parallel_df['ArraySize'].unique():
            for array_type in parallel_df['ArrayType'].unique():
                # Get parallel time
                parallel_time = parallel_df[
                    (parallel_df['AlgoName'] == algo_name) & 
                    (parallel_df['ArraySize'] == array_size) & 
                    (parallel_df['ArrayType'] == array_type)
                ]['TimeMs'].values
                
                # Get sequential time (prefer optimized)
                seq_opt = sequential_df[
                    (sequential_df['Category'] == 'Sequential Optimized') &
                    (sequential_df['AlgoName'] == algo_name) & 
                    (sequential_df['ArraySize'] == array_size) & 
                    (sequential_df['ArrayType'] == array_type)
                ]['TimeMs'].values
                
                seq_naive = sequential_df[
                    (sequential_df['Category'] == 'Sequential Naive') &
                    (sequential_df['AlgoName'] == algo_name) & 
                    (sequential_df['ArraySize'] == array_size) & 
                    (sequential_df['ArrayType'] == array_type)
                ]['TimeMs'].values
                
                sequential_time = seq_opt if len(seq_opt) > 0 else seq_naive
                
                if len(parallel_time) > 0 and len(sequential_time) > 0:
                    speedup = sequential_time[0] / parallel_time[0]
                    speedup_data.append({
                        'Algorithm': algo_name,
                        'ArraySize': array_size,
                        'ArrayType': array_type,
                        'Speedup': speedup
                    })
    
    if not speedup_data:
        print("⚠️  No speedup data could be calculated")
        return
    
    speedup_df = pd.DataFrame(speedup_data)
    
    # Create plot with subplots for each array type
    array_types = speedup_df['ArrayType'].unique()
    n_types = len(array_types)
    
    fig, axes = plt.subplots(1, n_types, figsize=(6*n_types, 6), squeeze=False)
    axes = axes.flatten()
    
    for idx, array_type in enumerate(array_types):
        ax = axes[idx]
        type_data = speedup_df[speedup_df['ArrayType'] == array_type]
        
        # Plot speedup by array size for each algorithm
        for algo in type_data['Algorithm'].unique():
            algo_data = type_data[type_data['Algorithm'] == algo].sort_values('ArraySize')
            ax.plot(algo_data['ArraySize'], algo_data['Speedup'], 
                   marker='o', label=algo, linewidth=2, markersize=8)
        
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No speedup')
        ax.set_xlabel('Array Size', fontsize=12)
        ax.set_ylabel('Speedup (Sequential / Parallel)', fontsize=12)
        ax.set_title(f'Array Type: {array_type}', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Speedup plot saved to: {output_path}")
    plt.close()


def plot_comparison_with_backup(current_df, backup_files, output_path):
    """
    Compare current results with previous runs (from backups).
    Shows improvement over time.
    """
    print("\n📊 Generating comparison with previous runs plot...")
    
    if not backup_files:
        print("⚠️  No backup files found for comparison")
        return
    
    # Load most recent backup
    latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Comparing with backup: {latest_backup.name}")
    
    backup_df = load_benchmark_data(latest_backup)
    if backup_df is None:
        return
    
    # Add categories and names
    for df in [current_df, backup_df]:
        df['Category'] = df['Algorithm'].apply(categorize_algorithm)
        df['AlgoName'] = df['Algorithm'].apply(extract_algorithm_name)
    
    # Calculate improvement
    improvement_data = []
    
    for algo in current_df['Algorithm'].unique():
        for array_size in current_df['ArraySize'].unique():
            for array_type in current_df['ArrayType'].unique():
                current_time = current_df[
                    (current_df['Algorithm'] == algo) & 
                    (current_df['ArraySize'] == array_size) & 
                    (current_df['ArrayType'] == array_type)
                ]['TimeMs'].values
                
                backup_time = backup_df[
                    (backup_df['Algorithm'] == algo) & 
                    (backup_df['ArraySize'] == array_size) & 
                    (backup_df['ArrayType'] == array_type)
                ]['TimeMs'].values
                
                if len(current_time) > 0 and len(backup_time) > 0:
                    improvement_pct = ((backup_time[0] - current_time[0]) / backup_time[0]) * 100
                    improvement_data.append({
                        'Algorithm': algo,
                        'ArraySize': array_size,
                        'ArrayType': array_type,
                        'Category': categorize_algorithm(algo),
                        'ImprovementPct': improvement_pct
                    })
    
    if not improvement_data:
        print("⚠️  No comparison data could be calculated")
        return
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Create bar chart grouped by category
    fig, ax = plt.subplots(figsize=(14, 8))
    
    categories = improvement_df['Category'].unique()
    x = np.arange(len(categories))
    width = 0.15
    
    array_sizes = sorted(improvement_df['ArraySize'].unique())
    
    for idx, size in enumerate(array_sizes):
        size_data = improvement_df[improvement_df['ArraySize'] == size]
        avg_improvement = [
            size_data[size_data['Category'] == cat]['ImprovementPct'].mean()
            for cat in categories
        ]
        
        offset = (idx - len(array_sizes)/2) * width
        ax.bar(x + offset, avg_improvement, width, 
               label=f'Size: {size:,}', alpha=0.8)
    
    ax.set_xlabel('Implementation Category', fontsize=12)
    ax.set_ylabel('Average Improvement (%)', fontsize=12)
    ax.set_title('Performance Improvement vs Previous Run', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {output_path}")
    plt.close()


def plot_performance_by_size(df, output_path):
    """
    Plot performance distribution by array size.
    Shows how different algorithms scale with input size.
    """
    print("\n📊 Generating performance by array size plot...")
    
    # Add categories
    df['Category'] = df['Algorithm'].apply(categorize_algorithm)
    df['AlgoName'] = df['Algorithm'].apply(extract_algorithm_name)
    
    # Create subplots for each category
    categories = df['Category'].unique()
    n_cats = len(categories)
    
    fig, axes = plt.subplots(n_cats, 1, figsize=(12, 5*n_cats), squeeze=False)
    axes = axes.flatten()
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        cat_data = df[df['Category'] == category]
        
        # Group by algorithm and plot
        for algo in cat_data['AlgoName'].unique():
            algo_data = cat_data[cat_data['AlgoName'] == algo]
            
            # Average across array types for cleaner visualization
            avg_by_size = algo_data.groupby('ArraySize')['TimeMs'].mean().reset_index()
            avg_by_size = avg_by_size.sort_values('ArraySize')
            
            ax.plot(avg_by_size['ArraySize'], avg_by_size['TimeMs'], 
                   marker='o', label=algo, linewidth=2, markersize=8)
        
        ax.set_xlabel('Array Size', fontsize=12)
        ax.set_ylabel('Execution Time (ms)', fontsize=12)
        ax.set_title(f'{category} - Performance vs Array Size', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Performance by size plot saved to: {output_path}")
    plt.close()


def main():
    """Main function to generate all plots."""
    print("=" * 60)
    print("🎨 Sorting Algorithm Benchmark Visualization")
    print("=" * 60)
    
    # Ensure plots directory exists
    PLOTS_DIR.mkdir(exist_ok=True)
    
    # Load current benchmark data
    df = load_benchmark_data(BENCHMARK_FILE)
    if df is None:
        print("\n❌ Cannot proceed without benchmark data.")
        print(f"   Please run benchmarks first to generate {BENCHMARK_FILE}")
        sys.exit(1)
    
    # Find backup files
    backup_files = list(DATA_DIR.glob('benchmark_backup_*.csv'))
    print(f"\n📁 Found {len(backup_files)} backup file(s)")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    
    plot_speedup_vs_sequential(df, PLOTS_DIR / 'speedup_vs_sequential.png')
    plot_comparison_with_backup(df, backup_files, PLOTS_DIR / 'improvement_vs_previous.png')
    plot_performance_by_size(df, PLOTS_DIR / 'performance_by_array_size.png')
    
    print("\n" + "=" * 60)
    print("✅ All plots generated successfully!")
    print("=" * 60)
    print(f"\n📂 Plots saved in: {PLOTS_DIR}")
    print("\nGenerated files:")
    print("  - speedup_vs_sequential.png")
    print("  - improvement_vs_previous.png")
    print("  - performance_by_array_size.png")
    print()


if __name__ == '__main__':
    main()
