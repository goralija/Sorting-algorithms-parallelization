#!/usr/bin/env python3
"""
Script for plotting sorting algorithm benchmark results.
Shows CPU parallelization speedup vs sequential implementation,
improvements compared to previous runs, and performance distribution by array size.
Generates both logarithmic and linear scale versions of plots.
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
        df = pd.read_csv(filepath, encoding="utf-16")
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

def create_performance_plot(ax, algo_data, title_suffix, array_type, algo_name, use_log_scale=True):
    """Helper function to create a performance plot with specified scaling."""
    # Define colors and styles
    colors = {'Sequential Naive': 'red', 'Sequential Optimized': 'blue', 'Parallel CPU': 'green'}
    
    # Plot each category for this specific algorithm
    plot_added = False
    for category, data, marker in [
        ('Sequential Naive', algo_data['naive'], 'o'), 
        ('Sequential Optimized', algo_data['optimized'], 's'), 
        ('Parallel CPU', algo_data['parallel'], '^')
    ]:
        if data.empty:
            continue
            
        avg_by_size = data.groupby('ArraySize')['TimeMs'].mean().reset_index()
        avg_by_size = avg_by_size.sort_values('ArraySize')
        
        if not avg_by_size.empty:
            ax.plot(avg_by_size['ArraySize'], avg_by_size['TimeMs'], 
                   color=colors[category],
                   marker=marker,
                   label=category, 
                   linewidth=2, 
                   markersize=8,
                   alpha=0.8)
            plot_added = True
    
    if plot_added:
        ax.set_xlabel('Array Size', fontsize=12)
        ax.set_ylabel('Execution Time (ms)', fontsize=12)
        ax.set_title(f'{algo_name.title()} Performance - {array_type} Arrays {title_suffix}', 
                    fontsize=14, fontweight='bold')
        
        if use_log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
    
    return plot_added

def plot_speedup_vs_sequential(df, output_path):
    """
    Plot CPU parallelization speedup compared to sequential implementation.
    Creates separate PNG plots for each array type showing naive, optimized, and parallelized performance.
    Generates both logarithmic and linear scale versions.
    """
    print("\n📊 Generating speedup vs sequential plots...")
    
    # Add categories
    df['Category'] = df['Algorithm'].apply(categorize_algorithm)
    df['AlgoName'] = df['Algorithm'].apply(extract_algorithm_name)
    
    # Get data for each category
    naive_df = df[df['Category'] == 'Sequential Naive']
    optimized_df = df[df['Category'] == 'Sequential Optimized']
    parallel_df = df[df['Category'] == 'Parallel CPU']
    
    if naive_df.empty and optimized_df.empty and parallel_df.empty:
        print("⚠️  Not enough data for comparison")
        return
    
    # Get all unique algorithm names across all categories
    all_algorithms = set()
    for data in [naive_df, optimized_df, parallel_df]:
        if not data.empty:
            all_algorithms.update(data['AlgoName'].unique())
    
    # Create plots for each algorithm and array type combination
    array_types = df['ArrayType'].unique()
    
    for algo_name in all_algorithms:
        # Create algorithm-specific folder structure
        algo_folder = output_path.parent / algo_name
        log_folder = algo_folder / 'logarithmic'
        linear_folder = algo_folder / 'linear'
        
        log_folder.mkdir(parents=True, exist_ok=True)
        linear_folder.mkdir(parents=True, exist_ok=True)
        
        for array_type in array_types:
            # Filter data for this algorithm and array type
            algo_data = {
                'naive': naive_df[(naive_df['AlgoName'] == algo_name) & (naive_df['ArrayType'] == array_type)],
                'optimized': optimized_df[(optimized_df['AlgoName'] == algo_name) & (optimized_df['ArrayType'] == array_type)],
                'parallel': parallel_df[(parallel_df['AlgoName'] == algo_name) & (parallel_df['ArrayType'] == array_type)]
            }
            
            # Create logarithmic scale plot
            fig_log, ax_log = plt.subplots(figsize=(12, 8))
            plot_added_log = create_performance_plot(ax_log, algo_data, '(Log Scale)', array_type, algo_name, use_log_scale=True)
            
            if plot_added_log:
                plt.tight_layout()
                safe_array_type = array_type.replace(' ', '_').replace('/', '_')
                log_output = log_folder / f'performance_{safe_array_type.lower()}.png'
                plt.savefig(log_output, dpi=300, bbox_inches='tight')
                print(f"✅ {algo_name} performance plot (log) for {array_type} saved to: {log_output}")
            plt.close(fig_log)
            
            # Create linear scale plot
            fig_linear, ax_linear = plt.subplots(figsize=(12, 8))
            plot_added_linear = create_performance_plot(ax_linear, algo_data, '(Linear Scale)', array_type, algo_name, use_log_scale=False)
            
            if plot_added_linear:
                plt.tight_layout()
                linear_output = linear_folder / f'performance_{safe_array_type.lower()}.png'
                plt.savefig(linear_output, dpi=300, bbox_inches='tight')
                print(f"✅ {algo_name} performance plot (linear) for {array_type} saved to: {linear_output}")
            plt.close(fig_linear)

def plot_comparison_with_backup(current_df, backup_files, output_path):
    """
    Compare current results with previous runs (from backups).
    Shows improvement percentages for naive, optimized, and parallelized versions in a single bar plot.
    Saves plots in algorithm-specific folders.
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
    
    # Get all unique algorithm names
    all_algorithms = set()
    all_algorithms.update(current_df['AlgoName'].unique())
    all_algorithms.update(backup_df['AlgoName'].unique())
    
    # Create comparison plot for each algorithm
    for algo_name in all_algorithms:
        # Create algorithm-specific folder
        algo_folder = output_path.parent / algo_name
        algo_folder.mkdir(exist_ok=True)
        
        # Calculate improvement by category for this specific algorithm
        categories = ['Sequential Naive', 'Sequential Optimized', 'Parallel CPU']
        improvement_by_category = {}
        
        for category in categories:
            current_cat = current_df[(current_df['Category'] == category) & (current_df['AlgoName'] == algo_name)]
            backup_cat = backup_df[(backup_df['Category'] == category) & (backup_df['AlgoName'] == algo_name)]
            
            if current_cat.empty or backup_cat.empty:
                improvement_by_category[category] = 0
                continue
            
            # Calculate average improvement for this algorithm in this category
            improvements = []
            
            for array_size in current_cat['ArraySize'].unique():
                for array_type in current_cat['ArrayType'].unique():
                    current_time = current_cat[
                        (current_cat['ArraySize'] == array_size) & 
                        (current_cat['ArrayType'] == array_type)
                    ]['TimeMs'].values
                    
                    backup_time = backup_cat[
                        (backup_cat['ArraySize'] == array_size) & 
                        (backup_cat['ArrayType'] == array_type)
                    ]['TimeMs'].values
                    
                    if len(current_time) > 0 and len(backup_time) > 0:
                        improvement_pct = ((backup_time[0] - current_time[0]) / backup_time[0]) * 100
                        improvements.append(improvement_pct)
            
            improvement_by_category[category] = np.mean(improvements) if improvements else 0
        
        # Only create plot if there's data
        if any(improvement_by_category.values()):
            # Create bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            categories_with_data = [cat for cat in categories if cat in improvement_by_category]
            improvements = [improvement_by_category[cat] for cat in categories_with_data]
            
            # Define colors
            colors = ['red', 'blue', 'green']
            
            bars = ax.bar(categories_with_data, improvements, color=colors[:len(categories_with_data)], alpha=0.7)
            
            # Add value labels on bars
            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1.5),
                        f'{improvement:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                        fontsize=11, fontweight='bold')
            
            ax.set_ylabel('Performance Improvement (%)', fontsize=12)
            ax.set_title(f'{algo_name.title()} - Performance Improvement vs Previous Run', fontsize=14, fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            
            # Save in algorithm-specific folder
            algo_output = algo_folder / 'improvement_vs_previous.png'
            plt.savefig(algo_output, dpi=300, bbox_inches='tight')
            print(f"✅ {algo_name} comparison plot saved to: {algo_output}")
            plt.close()

def create_performance_by_size_plot(ax, algo_data, algo_name, title_suffix, use_log_scale=True):
    """Helper function to create performance by size plot with specified scaling."""
    # Define colors for each category
    category_colors = {
        'Sequential Naive': 'red',
        'Sequential Optimized': 'blue', 
        'Parallel CPU': 'green'
    }
    
    categories_to_plot = ['Sequential Naive', 'Sequential Optimized', 'Parallel CPU']
    plot_added = False
    
    # Plot each category for this algorithm
    for category in categories_to_plot:
        cat_data = algo_data[algo_data['Category'] == category]
        
        if cat_data.empty:
            continue
        
        # Average across array types for cleaner visualization
        avg_by_size = cat_data.groupby('ArraySize')['TimeMs'].mean().reset_index()
        avg_by_size = avg_by_size.sort_values('ArraySize')
        
        if not avg_by_size.empty:
            # Plot with category color
            ax.plot(avg_by_size['ArraySize'], avg_by_size['TimeMs'], 
                   color=category_colors[category],
                   marker='o',
                   label=category, 
                   linewidth=2, 
                   markersize=6,
                   alpha=0.8)
            plot_added = True
    
    if plot_added:
        ax.set_xlabel('Array Size', fontsize=12)
        ax.set_ylabel('Execution Time (ms)', fontsize=12)
        ax.set_title(f'{algo_name.title()} - Performance Comparison {title_suffix}', fontsize=14, fontweight='bold')
        
        if use_log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
    
    return plot_added

def plot_performance_by_size(df, output_path):
    """
    Plot performance distribution by array size.
    Shows how different algorithms scale with input size.
    All sequential naive, optimized, and CPU parallelized on the same graph.
    Generates both logarithmic and linear scale versions.
    """
    print("\n📊 Generating performance by array size plot...")
    
    # Add categories
    df['Category'] = df['Algorithm'].apply(categorize_algorithm)
    df['AlgoName'] = df['Algorithm'].apply(extract_algorithm_name)
    
    # Filter for the categories we want to compare
    categories_to_plot = ['Sequential Naive', 'Sequential Optimized', 'Parallel CPU']
    plot_data = df[df['Category'].isin(categories_to_plot)]
    
    if plot_data.empty:
        print("⚠️  No data found for the specified categories")
        return
    
    # Get all unique algorithm names
    all_algorithms = plot_data['AlgoName'].unique()
    
    # Create plot for each algorithm
    for algo_name in all_algorithms:
        # Create algorithm-specific folder structure
        algo_folder = output_path.parent / algo_name
        log_folder = algo_folder / 'logarithmic'
        linear_folder = algo_folder / 'linear'
        
        log_folder.mkdir(parents=True, exist_ok=True)
        linear_folder.mkdir(parents=True, exist_ok=True)
        
        # Filter data for this algorithm
        algo_data = plot_data[plot_data['AlgoName'] == algo_name]
        
        if algo_data.empty:
            continue
        
        # Create logarithmic scale plot
        fig_log, ax_log = plt.subplots(figsize=(14, 8))
        plot_added_log = create_performance_by_size_plot(ax_log, algo_data, algo_name, '(Log Scale)', use_log_scale=True)
        
        if plot_added_log:
            plt.tight_layout()
            log_output = log_folder / 'performance_by_array_size.png'
            plt.savefig(log_output, dpi=300, bbox_inches='tight')
            print(f"✅ {algo_name} performance by size plot (log) saved to: {log_output}")
        plt.close(fig_log)
        
        # Create linear scale plot
        fig_linear, ax_linear = plt.subplots(figsize=(14, 8))
        plot_added_linear = create_performance_by_size_plot(ax_linear, algo_data, algo_name, '(Linear Scale)', use_log_scale=False)
        
        if plot_added_linear:
            plt.tight_layout()
            linear_output = linear_folder / 'performance_by_array_size.png'
            plt.savefig(linear_output, dpi=300, bbox_inches='tight')
            print(f"✅ {algo_name} performance by size plot (linear) saved to: {linear_output}")
        plt.close(fig_linear)

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
    print(f"\n📂 Plots saved in algorithm-specific folders under: {PLOTS_DIR}")
    print("\nGenerated folder structure:")
    print("📁 plots/")
    
    # Show generated folders
    for algo_folder in PLOTS_DIR.iterdir():
        if algo_folder.is_dir():
            print(f"  📁 {algo_folder.name}/")
            print(f"    📁 logarithmic/")
            print(f"    📁 linear/")
            for plot_file in algo_folder.glob('*.png'):
                print(f"    📊 {plot_file.name}")
            for scale_folder in ['logarithmic', 'linear']:
                scale_path = algo_folder / scale_folder
                if scale_path.exists():
                    for plot_file in scale_path.glob('*.png'):
                        print(f"      📊 {plot_file.name}")
    print()

if __name__ == '__main__':
    main()
