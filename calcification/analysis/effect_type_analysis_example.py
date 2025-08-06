#!/usr/bin/env python3
"""
Example usage of the Effect Type Reliability Analysis Module

This script shows how to use the analysis tools in your Jupyter notebook.
"""

# Import the analysis module
from effect_type_reliability_analysis import (
    EffectTypeReliabilityAnalyzer,
    plot_effect_type_comparison,
    quick_reliability_check,
    run_complete_reliability_analysis,
)


def example_usage_in_notebook():
    """
    Example of how to use the effect type reliability analysis in your notebook.

    Assuming you have a DataFrame called 'your_effect_data' with columns:
    - hedges_g, cohens_d, relative_calcification (effect types)
    - treatment_calcification, control_calcification (raw values)
    - control_calcification_sd (error magnitudes)
    """

    # Example 1: Quick reliability check
    print("=== Quick Reliability Check ===")
    summary_df = quick_reliability_check(your_effect_data)
    print(summary_df)

    # Example 2: Just the comparison plot
    print("\n=== Generating Effect Type Comparison Plot ===")
    comparison_fig = plot_effect_type_comparison(your_effect_data)
    plt.show()

    # Example 3: Complete analysis with all visualizations
    print("\n=== Running Complete Analysis ===")
    results = run_complete_reliability_analysis(
        df=your_effect_data,
        effect_types=["hedges_g", "cohens_d", "relative_calcification"],
        save_plots=True,
        output_dir="./effect_analysis_output/",
    )

    # Display all plots
    if results["heatmap_fig"]:
        plt.figure(results["heatmap_fig"].number)
        plt.show()

    plt.figure(results["comparison_fig"].number)
    plt.show()

    plt.figure(results["distribution_fig"].number)
    plt.show()

    # Display summary
    print("\n=== Summary Report ===")
    print(results["summary_df"])

    return results


def custom_analysis_example():
    """
    Example of using the analyzer class directly for custom analysis.
    """

    # Initialize analyzer with custom effect types
    analyzer = EffectTypeReliabilityAnalyzer(
        df=your_effect_data,
        effect_types=[
            "hedges_g",
            "cohens_d",
            "relative_calcification",
            "absolute_calcification",
        ],
    )

    # Generate individual plots
    print("Generating reliability heatmap...")
    heatmap_fig, heatmap_data = analyzer.create_reliability_heatmap(n_bins=8)
    if heatmap_fig:
        plt.show()

    print("Generating comparison plot...")
    comparison_fig = analyzer.compare_effect_types()
    plt.show()

    print("Generating distribution plots...")
    distribution_fig = analyzer.plot_reliability_distributions(n_bins=4)
    plt.show()

    # Get summary statistics
    summary = analyzer.generate_summary_report()
    print("Summary statistics:")
    print(summary)

    return {"analyzer": analyzer, "heatmap_data": heatmap_data, "summary": summary}


def troubleshooting_tips():
    """
    Common issues and solutions when using the analysis module.
    """

    print("=== Troubleshooting Tips ===")

    # Check if your DataFrame has the required columns
    required_cols = [
        "treatment_calcification",
        "control_calcification",
        "control_calcification_sd",
    ]

    missing_cols = [col for col in required_cols if col not in your_effect_data.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print("Make sure your DataFrame has these columns for the analysis to work.")

    # Check for NaN values
    nan_counts = your_effect_data[required_cols].isna().sum()
    print(f"NaN counts in required columns:\n{nan_counts}")

    # Check effect type columns
    effect_cols = ["hedges_g", "cohens_d", "relative_calcification"]
    available_effects = [col for col in effect_cols if col in your_effect_data.columns]
    print(f"Available effect type columns: {available_effects}")

    if not available_effects:
        print("No effect type columns found. Check your column names.")

    # Sample size check
    for effect_type in available_effects:
        valid_count = (~your_effect_data[effect_type].isna()).sum()
        print(f"{effect_type}: {valid_count} valid values")

    return {
        "missing_cols": missing_cols,
        "available_effects": available_effects,
        "nan_counts": nan_counts,
    }


# Example notebook cell usage:
"""
# Cell 1: Import and setup
from effect_type_reliability_analysis import *

# Cell 2: Quick check
summary = quick_reliability_check(your_effect_data)
print(summary)

# Cell 3: Generate comparison plot
fig = plot_effect_type_comparison(your_effect_data)
plt.show()

# Cell 4: Complete analysis
results = run_complete_reliability_analysis(your_effect_data, save_plots=True)

# Cell 5: Display results
print("Best performing effect type:")
best_effect = results['summary_df'].iloc[0]
print(f"{best_effect['effect_type']}: correlation = {best_effect['correlation']:.3f}")

# Cell 6: Custom analysis
analyzer = EffectTypeReliabilityAnalyzer(your_effect_data)
heatmap_fig, heatmap_data = analyzer.create_reliability_heatmap(n_bins=6)
plt.show()
"""
