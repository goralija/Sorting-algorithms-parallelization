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
    colors = {'Sequential Naive': 'red', 'Sequential Optimized': 'blue', 'Parallel CPU': 'green', 'Sequential': 'purple'}
    
    # Special case for std_sort - naive and optimized are the same
    is_std_sort = algo_name == 'std_sort'
    
    # Plot each category for this specific algorithm
    plot_added = False
    
    if is_std_sort:
        # For std_sort, combine naive and optimized into one "Sequential" line
        # Use optimized data (or naive if optimized is empty)
        seq_data = algo_data['optimized'] if not algo_data['optimized'].empty else algo_data['naive']
        
        if not seq_data.empty:
            avg_by_size = seq_data.groupby('ArraySize')['TimeMs'].mean().reset_index()
            avg_by_size = avg_by_size.sort_values('ArraySize')
            
            if not avg_by_size.empty:
                ax.plot(avg_by_size['ArraySize'], avg_by_size['TimeMs'], 
                       color=colors['Sequential'],
                       marker='s',
                       label='Sequential', 
                       linewidth=2, 
                       markersize=8,
                       alpha=0.8)
                plot_added = True
        
        # Plot parallel
        if not algo_data['parallel'].empty:
            avg_by_size = algo_data['parallel'].groupby('ArraySize')['TimeMs'].mean().reset_index()
            avg_by_size = avg_by_size.sort_values('ArraySize')
            
            if not avg_by_size.empty:
                ax.plot(avg_by_size['ArraySize'], avg_by_size['TimeMs'], 
                       color=colors['Parallel CPU'],
                       marker='^',
                       label='Parallel CPU', 
                       linewidth=2, 
                       markersize=8,
                       alpha=0.8)
                plot_added = True
    else:
        # Normal case - plot all three categories
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
        'Parallel CPU': 'green',
        'Sequential': 'purple'
    }
    
    # Special case for std_sort - naive and optimized are the same
    is_std_sort = algo_name == 'std_sort'
    
    plot_added = False
    
    if is_std_sort:
        # For std_sort, combine naive and optimized into one "Sequential" line
        seq_data = algo_data[algo_data['Category'].isin(['Sequential Naive', 'Sequential Optimized'])]
        
        if not seq_data.empty:
            # Average across array types and both categories
            avg_by_size = seq_data.groupby('ArraySize')['TimeMs'].mean().reset_index()
            avg_by_size = avg_by_size.sort_values('ArraySize')
            
            if not avg_by_size.empty:
                ax.plot(avg_by_size['ArraySize'], avg_by_size['TimeMs'], 
                       color=category_colors['Sequential'],
                       marker='s',
                       label='Sequential', 
                       linewidth=2, 
                       markersize=6,
                       alpha=0.8)
                plot_added = True
        
        # Plot parallel
        parallel_data = algo_data[algo_data['Category'] == 'Parallel CPU']
        if not parallel_data.empty:
            avg_by_size = parallel_data.groupby('ArraySize')['TimeMs'].mean().reset_index()
            avg_by_size = avg_by_size.sort_values('ArraySize')
            
            if not avg_by_size.empty:
                ax.plot(avg_by_size['ArraySize'], avg_by_size['TimeMs'], 
                       color=category_colors['Parallel CPU'],
                       marker='^',
                       label='Parallel CPU', 
                       linewidth=2, 
                       markersize=6,
                       alpha=0.8)
                plot_added = True
    else:
        # Normal case - plot all three categories
        categories_to_plot = ['Sequential Naive', 'Sequential Optimized', 'Parallel CPU']
        
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

def plot_all_optimized_algorithms(df, output_path):
    """
    Plot all parallel_cpu and sequential_optimized algorithms on the same graph.
    Creates one plot for each array type and one combined plot for all array types.
    Generates both logarithmic and linear scale versions.
    """
    print("\n📊 Generating all optimized algorithms comparison plots...")
    
    # Filter for parallel_cpu and sequential_optimized algorithms only
    filtered_df = df[df['Algorithm'].str.contains('parallel_cpu_|sequential_optimized_', regex=True)]
    
    if filtered_df.empty:
        print("⚠️  No parallel_cpu or sequential_optimized algorithms found")
        return
    
    # Create output folder
    comparison_folder = output_path.parent / 'all_algorithms_comparison'
    log_folder = comparison_folder / 'logarithmic'
    linear_folder = comparison_folder / 'linear'
    
    log_folder.mkdir(parents=True, exist_ok=True)
    linear_folder.mkdir(parents=True, exist_ok=True)
    
    # Get unique algorithms and array types
    algorithms = filtered_df['Algorithm'].unique()
    array_types = filtered_df['ArrayType'].unique()
    
    # Define colors - use a colormap for many algorithms
    colors = plt.cm.tab20(np.linspace(0, 1, len(algorithms)))
    algo_colors = {algo: colors[i] for i, algo in enumerate(algorithms)}
    
    # Define markers
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*', 'X', 'P']
    algo_markers = {algo: markers[i % len(markers)] for i, algo in enumerate(algorithms)}
    
    def create_comparison_plot(ax, data, title, use_log_scale=True):
        """Helper to create the comparison plot."""
        plot_added = False
        
        for algo in algorithms:
            algo_data = data[data['Algorithm'] == algo]
            if algo_data.empty:
                continue
            
            # Average by array size
            avg_by_size = algo_data.groupby('ArraySize')['TimeMs'].mean().reset_index()
            avg_by_size = avg_by_size.sort_values('ArraySize')
            
            if not avg_by_size.empty:
                # Create cleaner label (remove prefix)
                label = algo.replace('parallel_cpu_', 'parallel: ').replace('sequential_optimized_', 'seq_opt: ')
                
                ax.plot(avg_by_size['ArraySize'], avg_by_size['TimeMs'],
                       color=algo_colors[algo],
                       marker=algo_markers[algo],
                       label=label,
                       linewidth=2,
                       markersize=6,
                       alpha=0.8)
                plot_added = True
        
        if plot_added:
            ax.set_xlabel('Array Size', fontsize=12)
            ax.set_ylabel('Execution Time (ms)', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            if use_log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
            
            ax.grid(True, alpha=0.3)
            # Place legend outside the plot
            ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))
        
        return plot_added
    
    # 1. Create plot for each array type
    for array_type in array_types:
        type_data = filtered_df[filtered_df['ArrayType'] == array_type]
        
        if type_data.empty:
            continue
        
        safe_array_type = array_type.replace(' ', '_').replace('/', '_').lower()
        
        # Logarithmic scale
        fig_log, ax_log = plt.subplots(figsize=(14, 8))
        title_log = f'All Optimized Algorithms - {array_type} Arrays (Log Scale)'
        plot_added = create_comparison_plot(ax_log, type_data, title_log, use_log_scale=True)
        
        if plot_added:
            plt.tight_layout()
            log_output = log_folder / f'comparison_{safe_array_type}.png'
            plt.savefig(log_output, dpi=300, bbox_inches='tight')
            print(f"✅ Comparison plot (log) for {array_type} saved to: {log_output}")
        plt.close(fig_log)
        
        # Linear scale
        fig_linear, ax_linear = plt.subplots(figsize=(14, 8))
        title_linear = f'All Optimized Algorithms - {array_type} Arrays (Linear Scale)'
        plot_added = create_comparison_plot(ax_linear, type_data, title_linear, use_log_scale=False)
        
        if plot_added:
            plt.tight_layout()
            linear_output = linear_folder / f'comparison_{safe_array_type}.png'
            plt.savefig(linear_output, dpi=300, bbox_inches='tight')
            print(f"✅ Comparison plot (linear) for {array_type} saved to: {linear_output}")
        plt.close(fig_linear)
    
    # 2. Create combined plot (all array types averaged)
    # Logarithmic scale
    fig_log, ax_log = plt.subplots(figsize=(14, 8))
    title_log = 'All Optimized Algorithms - Combined (Log Scale)'
    plot_added = create_comparison_plot(ax_log, filtered_df, title_log, use_log_scale=True)
    
    if plot_added:
        plt.tight_layout()
        log_output = log_folder / 'comparison_all_types.png'
        plt.savefig(log_output, dpi=300, bbox_inches='tight')
        print(f"✅ Combined comparison plot (log) saved to: {log_output}")
    plt.close(fig_log)
    
    # Linear scale
    fig_linear, ax_linear = plt.subplots(figsize=(14, 8))
    title_linear = 'All Optimized Algorithms - Combined (Linear Scale)'
    plot_added = create_comparison_plot(ax_linear, filtered_df, title_linear, use_log_scale=False)
    
    if plot_added:
        plt.tight_layout()
        linear_output = linear_folder / 'comparison_all_types.png'
        plt.savefig(linear_output, dpi=300, bbox_inches='tight')
        print(f"✅ Combined comparison plot (linear) saved to: {linear_output}")
    plt.close(fig_linear)

def load_hyperparameter_tuning_data(filepath):
    """Load hyperparameter tuning data from text file."""
    if not filepath.exists():
        return None, None
    
    configs = {}
    current_config = None
    current_data = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check if this is a config header (e.g., "A: INSERTION=48, ...")
                if ':' in line and ',' in line and 'parallel_cpu' not in line:
                    # Save previous config data
                    if current_config and current_data:
                        configs[current_config] = current_data
                    
                    # Parse new config
                    parts = line.split(':', 1)
                    current_config = parts[0].strip()
                    current_data = []
                
                # Check if this is a data line
                elif line.startswith('parallel_cpu'):
                    parts = line.split(',')
                    if len(parts) >= 4:
                        current_data.append({
                            'Algorithm': parts[0],
                            'ArraySize': int(parts[1]),
                            'ArrayType': parts[2],
                            'TimeMs': float(parts[3])
                        })
            
            # Save last config
            if current_config and current_data:
                configs[current_config] = current_data
        
        # Convert to DataFrames
        config_dfs = {}
        for config_name, data in configs.items():
            if data:
                config_dfs[config_name] = pd.DataFrame(data)
        
        return config_dfs
    
    except Exception as e:
        print(f"❌ Error reading hyperparameter tuning file: {e}")
        return None

def plot_hyperparameter_tuning(output_path):
    """
    Plot hyperparameter tuning results for all algorithms.
    Automatically detects all hyperparameter_tuning_*.txt files in data directory.
    Shows performance comparison across different configurations.
    """
    print("\n📊 Generating hyperparameter tuning plots...")
    
    # Find all hyperparameter tuning files
    tuning_files = list(DATA_DIR.glob('hyperparameter_tuning_*.txt'))
    
    if not tuning_files:
        print("⚠️  No hyperparameter tuning files found")
        return
    
    print(f"📁 Found {len(tuning_files)} hyperparameter tuning file(s)")
    
    for tuning_file in tuning_files:
        # Extract algorithm name from filename (e.g., "hyperparameter_tuning_quick_sort.txt" -> "quick_sort")
        algo_name = tuning_file.stem.replace('hyperparameter_tuning_', '')
        
        config_dfs = load_hyperparameter_tuning_data(tuning_file)
        
        if not config_dfs:
            print(f"⚠️  No data found in {tuning_file.name}")
            continue
        
        print(f"✅ Loaded {len(config_dfs)} configurations from {tuning_file.name}")
        
        # Create output folder for this algorithm
        tuning_folder = output_path.parent / 'hyperparameter_tuning' / algo_name
        tuning_folder.mkdir(parents=True, exist_ok=True)
        
        # Define colors for configurations
        config_colors = plt.cm.tab20(np.linspace(0, 1, len(config_dfs)))
        color_map = {config: config_colors[i] for i, config in enumerate(config_dfs.keys())}
        
        # Get all array types
        all_array_types = set()
        for df in config_dfs.values():
            all_array_types.update(df['ArrayType'].unique())
        
        # Get largest array size in the data
        max_array_size = 0
        for df in config_dfs.values():
            max_array_size = max(max_array_size, df['ArraySize'].max())
        
        # 1. Plot for each array type
        for array_type in all_array_types:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for config_name, df in config_dfs.items():
                type_data = df[df['ArrayType'] == array_type]
                if type_data.empty:
                    continue
                
                avg_by_size = type_data.groupby('ArraySize')['TimeMs'].mean().reset_index()
                avg_by_size = avg_by_size.sort_values('ArraySize')
                
                ax.plot(avg_by_size['ArraySize'], avg_by_size['TimeMs'],
                       color=color_map[config_name],
                       marker='o',
                       label=config_name,
                       linewidth=2,
                       markersize=6,
                       alpha=0.8)
            
            ax.set_xlabel('Array Size', fontsize=12)
            ax.set_ylabel('Execution Time (ms)', fontsize=12)
            ax.set_title(f'Hyperparameter Tuning ({algo_name}) - {array_type} Arrays', fontsize=14, fontweight='bold')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))
            
            plt.tight_layout()
            safe_type = array_type.replace(' ', '_').lower()
            output_file = tuning_folder / f'tuning_{safe_type}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✅ {algo_name} tuning plot for {array_type} saved to: {output_file}")
            plt.close()
        
        # 2. Create summary bar chart for largest array size
        num_types = len(all_array_types)
        if num_types > 0:
            rows = (num_types + 1) // 2
            cols = min(2, num_types)
            fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
            if num_types == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if num_types > 2 else list(axes)
            
            for idx, array_type in enumerate(sorted(all_array_types)):
                ax = axes[idx]
                
                config_times = {}
                for config_name, df in config_dfs.items():
                    type_data = df[(df['ArrayType'] == array_type) & (df['ArraySize'] == max_array_size)]
                    if not type_data.empty:
                        config_times[config_name] = type_data['TimeMs'].mean()
                
                if config_times:
                    configs = list(config_times.keys())
                    times = list(config_times.values())
                    colors = [color_map[c] for c in configs]
                    
                    bars = ax.bar(configs, times, color=colors, alpha=0.8)
                    
                    # Highlight best config
                    min_time = min(times)
                    for bar, time in zip(bars, times):
                        if time == min_time:
                            bar.set_edgecolor('gold')
                            bar.set_linewidth(3)
                    
                    ax.set_xlabel('Configuration', fontsize=11)
                    ax.set_ylabel('Execution Time (ms)', fontsize=11)
                    ax.set_title(f'{array_type} ({max_array_size:,} elements)', fontsize=12, fontweight='bold')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3, axis='y')
            
            # Hide unused subplots
            for idx in range(len(all_array_types), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Hyperparameter Tuning Summary - {algo_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            output_file = tuning_folder / f'tuning_summary_{max_array_size}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✅ {algo_name} tuning summary saved to: {output_file}")
            plt.close()
        
        # 3. Create heatmap of average performance
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Build matrix: rows=configs, cols=array_types
        configs = sorted(config_dfs.keys())
        array_types_sorted = sorted(all_array_types)
        
        matrix = np.zeros((len(configs), len(array_types_sorted)))
        
        for i, config in enumerate(configs):
            df = config_dfs[config]
            for j, array_type in enumerate(array_types_sorted):
                type_data = df[(df['ArrayType'] == array_type) & (df['ArraySize'] == max_array_size)]
                if not type_data.empty:
                    matrix[i, j] = type_data['TimeMs'].mean()
        
        # Normalize each column to show relative performance (best = 1.0)
        col_mins = matrix.min(axis=0)
        col_mins[col_mins == 0] = 1  # Avoid division by zero
        normalized = matrix / col_mins
        
        im = ax.imshow(normalized, cmap='RdYlGn_r', aspect='auto', vmin=1.0, vmax=2.0)
        
        ax.set_xticks(np.arange(len(array_types_sorted)))
        ax.set_yticks(np.arange(len(configs)))
        ax.set_xticklabels(array_types_sorted, fontsize=10)
        ax.set_yticklabels(configs, fontsize=10)
        
        # Add text annotations
        for i in range(len(configs)):
            for j in range(len(array_types_sorted)):
                text = f'{normalized[i, j]:.2f}x'
                color = 'white' if normalized[i, j] > 1.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
        
        ax.set_title(f'{algo_name.replace("_", " ").title()} - Relative Performance (1.0 = Best)\n{max_array_size:,} elements', 
                    fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Relative Time (vs best)', fontsize=11)
        
        plt.tight_layout()
        output_file = tuning_folder / 'tuning_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ {algo_name} tuning heatmap saved to: {output_file}")
        plt.close()
        
        # 4. Print best configuration summary
        print(f"\n📊 Best configurations for {algo_name} ({max_array_size:,} elements):")
        for array_type in sorted(all_array_types):
            best_config = None
            best_time = float('inf')
            for config_name, df in config_dfs.items():
                type_data = df[(df['ArrayType'] == array_type) & (df['ArraySize'] == max_array_size)]
                if not type_data.empty:
                    avg_time = type_data['TimeMs'].mean()
                    if avg_time < best_time:
                        best_time = avg_time
                        best_config = config_name
            if best_config:
                print(f"   {array_type}: {best_config} ({best_time:.1f} ms)")

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
    plot_all_optimized_algorithms(df, PLOTS_DIR / 'all_optimized_comparison.png')
    plot_hyperparameter_tuning(PLOTS_DIR / 'hyperparameter_tuning.png')
    
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
