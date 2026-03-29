"""Shared utilities for SCOUT visualization notebooks."""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_METHOD_METRIC_NAMES = [
    '',
    '-hunyuan',
    '-instant_mesh',
    '-trellis',
    '-triposr',
]

INPUT_AGNOSTIC_METHODS = {'Mean'}
INPUT_AGNOSTIC_VARIANTS = {
    '-hunyuan', '-instant_mesh', '-trellis', '-triposr', '-pass_baseline',
}

VARIANT_COLOR_PALETTE = [
    '#95E1D3', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B',
]

METHOD_COLOR_PALETTE = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#95E1D3',
    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#AED6F1',
    '#A9DFBF', '#F9E79F', '#D7BDE2', '#F1948A',
]

VARIANT_LABELS = {
    '': 'Pure',
    '-hunyuan': 'Hunyuan',
    '-instant_mesh': 'Instant Mesh',
    '-trellis': 'Trellis',
    '-triposr': 'TripoSR',
    '-pass_baseline': 'Pass Baseline',
}

MEANPASS_VARIANTS = [
    '-hunyuan', '-instant_mesh', '-trellis', '-triposr', '-pass_baseline',
]
MEANPASS_VARIANT_LABELS = [
    'MP-Hunyuan', 'MP-InstantMesh', 'MP-Trellis', 'MP-TripoSR', 'MP-PassBase',
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_result_file(filepath):
    """Read a result file and return a dictionary of metric name -> value.

    Args:
        filepath: Path to the result file.

    Returns:
        Dict of metric values, or None if the file does not exist.
    """
    if not os.path.exists(filepath):
        return None

    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ':' in line:
                key, value = line.split(':', 1)
                metrics[key.strip()] = float(value.strip())
    return metrics


def discover_all_variants(result_dirs, file_patterns, exp_metric_names,
                          num_seeds=100):
    """Discover all metric variants that exist across all result files.

    Args:
        result_dirs: Dict mapping method names to result directories.
        file_patterns: Dict mapping method names to filename patterns.
        exp_metric_names: List of experiment metric prefixes.
        num_seeds: Number of seed files to scan for discovery.

    Returns:
        Sorted list of discovered variant suffixes.
    """
    all_variants = set(STANDARD_METHOD_METRIC_NAMES)

    for method_name, result_dir in result_dirs.items():
        file_pattern = file_patterns[method_name]
        for seed in range(min(5, num_seeds)):
            filepath = os.path.join(result_dir, file_pattern.format(seed))
            if not os.path.exists(filepath):
                continue
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    key = line.split(':', 1)[0].strip()
                    for exp in exp_metric_names:
                        if key.startswith(exp):
                            variant = key[len(exp):]
                            if variant:
                                all_variants.add(variant)
                            break

    return sorted(list(all_variants))


def validate_all_files(result_dirs, file_patterns, metric_names, seeds,
                       method_metric_names=None):
    """Validate that all expected files exist and contain all metrics.

    Args:
        result_dirs: Dict mapping method names to result directories.
        file_patterns: Dict mapping method names to filename patterns.
        metric_names: List of expected metric names.
        seeds: Iterable of seed values to check.
        method_metric_names: Optional dict of per-method metric name overrides.

    Returns:
        True if all files are valid, False otherwise.
    """
    print("=" * 80)
    print("VALIDATING ALL FILES")
    print("=" * 80)

    all_valid = True
    for method_name, result_dir in result_dirs.items():
        file_pattern = file_patterns[method_name]
        print(f"\nValidating {method_name}...")

        expected_metrics = metric_names
        if method_metric_names and method_name in method_metric_names:
            expected_metrics = method_metric_names[method_name]

        missing_files = []
        incomplete_files = []

        for seed in seeds:
            filepath = os.path.join(result_dir, file_pattern.format(seed))
            if not os.path.exists(filepath):
                raise ValueError(f"File {filepath} does not exist")
            else:
                metrics = read_result_file(filepath)
                if metrics is None:
                    incomplete_files.append((seed, "Could not read file"))
                    all_valid = False
                else:
                    missing_metrics = [
                        m for m in expected_metrics if m not in metrics
                    ]
                    if missing_metrics:
                        raise ValueError(
                            f"Missing {len(missing_metrics)} metrics: "
                            f"{missing_metrics} of seed {seed}"
                        )

        if missing_files:
            print(f"  Missing files: {len(missing_files)} seeds")
        if incomplete_files:
            print(f"  Incomplete files: {len(incomplete_files)} seeds")
            for seed, reason in incomplete_files[:5]:
                print(f"    Seed {seed}: {reason}")
        if not missing_files and not incomplete_files:
            print(f"  All {len(seeds)} files valid!")

    if all_valid:
        print("\nAll files validated successfully - no missing entries!")
    else:
        print("\nValidation failed - some files are missing or incomplete")

    return all_valid


def create_results_table(method_name, result_dir, file_pattern, metric_names,
                         seeds=range(20)):
    """Create a results DataFrame for a given method across seeds.

    Args:
        method_name: Name of the method.
        result_dir: Path to the results directory.
        file_pattern: Filename pattern with {} for seed.
        metric_names: List of metric names to collect.
        seeds: Iterable of seed values.

    Returns:
        DataFrame with metrics as rows, seeds as columns, plus mean/std_err.
    """
    data = {metric: [] for metric in metric_names}
    available_seeds = []

    for seed in seeds:
        filepath = os.path.join(result_dir, file_pattern.format(seed))
        metrics = read_result_file(filepath)

        if metrics is not None:
            available_seeds.append(seed)
            for metric in metric_names:
                data[metric].append(metrics.get(metric, np.nan))
        else:
            print(f"Warning: {filepath} not found")

    df = pd.DataFrame(data, index=available_seeds).T
    df['mean'] = df.mean(axis=1)
    df['std_err'] = df.sem(axis=1)

    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def is_input_agnostic(method, variant=''):
    """Check whether a method/variant combination is input-agnostic.

    Args:
        method: Method name.
        variant: Variant suffix (e.g. '-hunyuan').

    Returns:
        True if the method or variant is input-agnostic.
    """
    return method in INPUT_AGNOSTIC_METHODS or variant in INPUT_AGNOSTIC_VARIANTS


def get_variants_for_experiment(exp, tables, methods):
    """Get all variant suffixes that exist for a given experiment.

    Args:
        exp: Experiment metric prefix.
        tables: Dict of method name -> results DataFrame.
        methods: List of method names to check.

    Returns:
        Sorted list of variant suffixes.
    """
    variants = set()
    for method in methods:
        if method not in tables:
            continue
        for metric_name in tables[method].index:
            if metric_name == exp:
                variants.add('')
            elif metric_name.startswith(exp + '-'):
                variants.add(metric_name[len(exp):])
            elif metric_name.startswith(exp) and metric_name != exp:
                variant = metric_name[len(exp):]
                if variant:
                    variants.add(variant)
    return sorted(list(variants))


def get_variant_color(variant, variant_color_map):
    """Get or assign a color for a variant suffix.

    Args:
        variant: Variant suffix string.
        variant_color_map: Mutable dict tracking assigned colors.

    Returns:
        Hex color string.
    """
    if variant not in variant_color_map:
        idx = len(variant_color_map)
        variant_color_map[variant] = VARIANT_COLOR_PALETTE[
            idx % len(VARIANT_COLOR_PALETTE)
        ]
    return variant_color_map[variant]


def get_method_colors(methods):
    """Generate a color mapping for a list of methods.

    Args:
        methods: List of method names.

    Returns:
        Dict mapping method name -> hex color string.
    """
    return {
        method: METHOD_COLOR_PALETTE[idx % len(METHOD_COLOR_PALETTE)]
        for idx, method in enumerate(methods)
    }


def plot_experiment_results(tables, exp_metric_names):
    """Generate bar charts for all experiments across methods.

    For each experiment, produces two plots:
    1. All variants (variable bars per method)
    2. Pure experiment only + MeanPass variants

    Args:
        tables: Dict of method name -> results DataFrame.
        exp_metric_names: List of experiment metric prefixes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch

    sns.set_style("whitegrid")

    all_methods = list(tables.keys())
    variant_color_map = {}

    for exp in exp_metric_names:
        methods = all_methods
        variants = get_variants_for_experiment(exp, tables, methods)
        print(f"\n{exp} has variants: {variants}")

        method_colors = get_method_colors(methods)
        print(f"Generating plots for {exp}...")

        # ------------------------------------------------------------------
        # Plot 1: All variants
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(20, 8))

        x_positions = []
        bar_heights = []
        bar_errors = []
        bar_colors = []
        x_tick_positions = []
        x_tick_labels = []
        x_pos = 0

        for method in methods:
            method_start_pos = x_pos
            bars_added = 0

            for variant in variants:
                metric_name = exp + variant
                if metric_name not in tables[method].index:
                    continue

                mean_val = tables[method].loc[metric_name, 'mean']
                std_err_val = tables[method].loc[metric_name, 'std_err']
                if np.isnan(mean_val):
                    continue

                x_positions.append(x_pos)
                bar_heights.append(mean_val)

                if method in INPUT_AGNOSTIC_METHODS:
                    bar_errors.append(0.0)
                else:
                    bar_errors.append(std_err_val)

                bar_colors.append(get_variant_color(variant, variant_color_map))
                x_pos += 1
                bars_added += 1

            if bars_added > 0:
                method_center = method_start_pos + (bars_added - 1) / 2
                x_tick_positions.append(method_center)
                x_tick_labels.append(method)
                x_pos += 0.5

        ax.bar(
            x_positions, bar_heights, yerr=bar_errors,
            color=bar_colors, capsize=3, width=0.8,
            edgecolor='black', linewidth=0.5, alpha=0.8,
        )

        ax.set_xlabel('Methods', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{exp} Value', fontsize=14, fontweight='bold')
        ax.set_title(f'{exp} Results - All Variants',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, fontsize=12, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        legend_elements = [
            Patch(
                facecolor=get_variant_color(v, variant_color_map),
                label=VARIANT_LABELS.get(v, v.lstrip('-')),
                edgecolor='black',
            )
            for v in variants
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
                  framealpha=0.9, edgecolor='black')

        plt.yscale('log')
        plt.tight_layout()
        plt.show()

        # ------------------------------------------------------------------
        # Plot 2: Pure experiment only + MeanPass variants
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(14, 8))

        x_positions = []
        bar_heights = []
        bar_errors = []
        bar_colors = []
        x_tick_labels = []
        print_data = []

        for method in methods:
            if exp not in tables[method].index:
                continue

            mean_val = tables[method].loc[exp, 'mean']
            std_err_val = tables[method].loc[exp, 'std_err']
            if np.isnan(mean_val):
                continue

            x_positions.append(len(x_tick_labels))
            bar_heights.append(mean_val)

            if is_input_agnostic(method):
                bar_errors.append(0.0)
                print_data.append((method, mean_val, None))
            else:
                bar_errors.append(std_err_val)
                print_data.append((method, mean_val, std_err_val))

            bar_colors.append(method_colors.get(method, '#95E1D3'))
            x_tick_labels.append(method)

        separator_pos = len(x_tick_labels)

        if 'Mean' in tables:
            for variant, label in zip(MEANPASS_VARIANTS, MEANPASS_VARIANT_LABELS):
                metric_name = exp + variant
                if metric_name not in tables['Mean'].index:
                    continue

                mean_val = tables['Mean'].loc[metric_name, 'mean']
                if np.isnan(mean_val):
                    continue

                x_positions.append(len(x_tick_labels) + 0.5)
                std_err_val = tables['Mean'].loc[metric_name, 'std_err']
                bar_heights.append(mean_val)
                bar_errors.append(std_err_val)
                bar_colors.append(get_variant_color(variant, variant_color_map))
                x_tick_labels.append(label)
                print_data.append((label, mean_val, std_err_val))

        ax.bar(
            x_positions, bar_heights, yerr=bar_errors,
            color=bar_colors, capsize=5, width=0.6,
            edgecolor='black', linewidth=1.0, alpha=0.8,
        )

        if separator_pos < len(x_positions):
            ax.axvline(x=separator_pos - 0.2, color='gray', linestyle='--',
                        linewidth=2, alpha=0.5)

        ax.set_xlabel('Methods and MeanPass Variants',
                       fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{exp} Value', fontsize=14, fontweight='bold')
        ax.set_title(f'{exp} Results - Pure Methods + MeanPass Variants',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_tick_labels, fontsize=11, fontweight='bold',
                           rotation=45, ha='right')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        for pos, height, error in zip(x_positions, bar_heights, bar_errors):
            ax.text(pos, height + error, f'{height:.6f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.yscale('log')
        plt.tight_layout()
        plt.show()

        # Print summary table
        print(f"\n{exp} - Means and Standard Errors:")
        print("-" * 60)
        print(f"{'Method/Variant':<30} {'Mean':<20} {'Std Err':<20}")
        print("-" * 60)
        for method, mean_val, std_err_val in print_data:
            if std_err_val is not None:
                print(f"{method:<30} {mean_val:<20.6f} {std_err_val:<20.6f}")
            else:
                print(f"{method:<30} {mean_val:<20.6f} {'--':<20}")
        print("-" * 60)
        print()

    print("\n" + "=" * 80)
    print("All graphs generated successfully!")
    print(f"Total plots created: {len(exp_metric_names) * 2}"
          f" ({len(exp_metric_names)} experiments x 2 plots each)")
    print("=" * 80)


# ---------------------------------------------------------------------------
# LaTeX table helpers
# ---------------------------------------------------------------------------

def print_latex_table(summary, experiments, all_methods):
    """Print a LaTeX-formatted summary table normalized by Mean baseline.

    Args:
        summary: Dict of method -> dict of exp -> (norm_mean, norm_std_err).
        experiments: List of experiment names (table columns).
        all_methods: List of method names (table rows).
    """
    print("% LaTeX summary table (normalized by Mean method's mean)")
    print("% Columns:", " & ".join(experiments))
    print()

    for method in all_methods:
        if method not in summary:
            continue
        row_parts = [method]
        for exp in experiments:
            if exp not in summary[method]:
                row_parts.append("--")
            elif is_input_agnostic(method):
                m, _ = summary[method][exp]
                row_parts.append(f"${m:.4f}$")
            else:
                m, s = summary[method][exp]
                row_parts.append(f"${m:.4f}\\pm{s:.4f}$")
        print(" & ".join(row_parts) + r" \\")
