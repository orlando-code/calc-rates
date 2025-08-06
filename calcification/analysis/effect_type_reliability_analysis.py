#!/usr/bin/env python3
"""
Effect Type Reliability Analysis Module

This module provides comprehensive analysis and visualization tools to compare
how different effect types (Hedges' g, Cohen's d, relative calcification, etc.)
capture the underlying data patterns across different control values and error magnitudes.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better plots
plt.style.use("default")
sns.set_palette("husl")


class EffectTypeReliabilityAnalyzer:
    """
    Analyzer class for comparing reliability of different effect types.
    """

    def __init__(self, df: pd.DataFrame, effect_types: Optional[List[str]] = None):
        """
        Initialize the analyzer with a DataFrame containing effect size data.

        Args:
            df: DataFrame with effect size columns and control/treatment data
            effect_types: List of effect type column names to analyze
        """
        self.df = df.copy()

        # Default effect types if none provided
        if effect_types is None:
            self.effect_types = ["hedges_g", "cohens_d", "relative_rate"]
        else:
            self.effect_types = effect_types

        # Calculate raw differences for comparison
        self.raw_diff = self.df["calcs"] - self.df["control_calcification"]

    def create_reliability_heatmap(
        self, n_bins: int = 10
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        Create a 2D heatmap showing effect type reliability across control value
        ranges and error magnitudes.

        Args:
            n_bins: Number of bins for control values and error magnitudes

        Returns:
            Tuple of (figure, reliability_dataframe)
        """
        # Create bins for control values and error magnitudes
        control_bins = pd.cut(
            self.df["control_calcification"], bins=n_bins, labels=False
        )
        error_bins = pd.cut(
            self.df["control_calcification_sd"], bins=n_bins, labels=False
        )

        reliability_data = []

        for effect_type in self.effect_types:
            for control_bin in range(n_bins):
                for error_bin in range(n_bins):
                    mask = (control_bins == control_bin) & (error_bins == error_bin)
                    subset = self.df[mask]

                    if len(subset) > 0:
                        # Calculate reliability metric (correlation with raw differences)
                        effect_values = subset[effect_type]

                        # Remove NaN values
                        valid_mask = ~(
                            self.raw_diff[mask].isna() | effect_values.isna()
                        )
                        if valid_mask.sum() > 5:  # Need minimum sample size
                            try:
                                correlation = np.corrcoef(
                                    self.raw_diff[mask][valid_mask],
                                    effect_values[valid_mask],
                                )[0, 1]

                                if not np.isnan(correlation):
                                    reliability_data.append(
                                        {
                                            "effect_type": effect_type,
                                            "control_bin": control_bin,
                                            "error_bin": error_bin,
                                            "reliability": correlation,
                                            "sample_size": valid_mask.sum(),
                                        }
                                    )
                            except Exception:
                                continue

        reliability_df = pd.DataFrame(reliability_data)

        if reliability_df.empty:
            print("Warning: No reliability data could be calculated. Check your data.")
            return None, reliability_df

        # Create heatmap
        fig, axes = plt.subplots(
            1, len(self.effect_types), figsize=(5 * len(self.effect_types), 5)
        )
        if len(self.effect_types) == 1:
            axes = [axes]

        for i, effect_type in enumerate(self.effect_types):
            effect_data = reliability_df[reliability_df["effect_type"] == effect_type]

            if not effect_data.empty:
                pivot_data = effect_data.pivot(
                    index="control_bin", columns="error_bin", values="reliability"
                )

                im = axes[i].imshow(
                    pivot_data, cmap="RdYlBu_r", aspect="auto", vmin=-1, vmax=1
                )
                axes[i].set_title(
                    f"{effect_type} Reliability", fontsize=14, fontweight="bold"
                )
                axes[i].set_xlabel("Error Magnitude Bin", fontsize=12)
                axes[i].set_ylabel("Control Value Bin", fontsize=12)

                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[i])
                cbar.set_label("Correlation with Raw Differences", fontsize=10)

                # Add sample size annotations
                for control_bin in range(n_bins):
                    for error_bin in range(n_bins):
                        sample_data = effect_data[
                            (effect_data["control_bin"] == control_bin)
                            & (effect_data["error_bin"] == error_bin)
                        ]
                        if not sample_data.empty:
                            sample_size = sample_data.iloc[0]["sample_size"]
                            axes[i].text(
                                error_bin,
                                control_bin,
                                f"n={sample_size}",
                                ha="center",
                                va="center",
                                fontsize=8,
                                alpha=0.7,
                            )
            else:
                axes[i].text(
                    0.5,
                    0.5,
                    f"No data for {effect_type}",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(f"{effect_type} Reliability")

        plt.tight_layout()
        return fig, reliability_df

    def compare_effect_types(self) -> plt.Figure:
        """
        Compare different effect types against each other and raw differences.

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Effect types vs raw differences
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.effect_types)))
        for i, effect_type in enumerate(self.effect_types):
            valid_mask = ~(self.raw_diff.isna() | self.df[effect_type].isna())
            axes[0, 0].scatter(
                self.raw_diff[valid_mask],
                self.df[effect_type][valid_mask],
                alpha=0.6,
                label=effect_type,
                s=20,
                color=colors[i],
            )

        # Add 1:1 line
        min_val = min(self.raw_diff.min(), self.df[self.effect_types].min().min())
        max_val = max(self.raw_diff.max(), self.df[self.effect_types].max().max())
        axes[0, 0].plot(
            [min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="1:1 line"
        )
        axes[0, 0].set_xlabel("Raw Difference (Treatment - Control)", fontsize=12)
        axes[0, 0].set_ylabel("Effect Size", fontsize=12)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].set_title(
            "Effect Types vs Raw Differences", fontsize=14, fontweight="bold"
        )
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Hedges' g vs Cohen's d (if both exist)
        if "hedges_g" in self.effect_types and "cohens_d" in self.effect_types:
            valid_mask = ~(self.df["hedges_g"].isna() | self.df["cohens_d"].isna())
            axes[0, 1].scatter(
                self.df["cohens_d"][valid_mask],
                self.df["hedges_g"][valid_mask],
                alpha=0.6,
                s=20,
                color="steelblue",
            )

            # Add 1:1 line
            min_val = min(self.df["hedges_g"].min(), self.df["cohens_d"].min())
            max_val = max(self.df["hedges_g"].max(), self.df["cohens_d"].max())
            axes[0, 1].plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)
            axes[0, 1].set_xlabel("Cohen's d", fontsize=12)
            axes[0, 1].set_ylabel("Hedges' g", fontsize=12)
            axes[0, 1].set_title(
                "Hedges' g vs Cohen's d", fontsize=14, fontweight="bold"
            )
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "Hedges' g and Cohen's d not available",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )
            axes[0, 1].set_title("Hedges' g vs Cohen's d")

        # Plot 3: Effect reliability vs control value
        reliability_by_control = []
        for effect_type in self.effect_types:
            valid_mask = ~(self.raw_diff.isna() | self.df[effect_type].isna())
            if valid_mask.sum() > 5:
                correlation = np.corrcoef(
                    self.raw_diff[valid_mask], self.df[effect_type][valid_mask]
                )[0, 1]
                reliability_by_control.append(
                    {
                        "effect_type": effect_type,
                        "reliability": correlation,
                        "mean_control": self.df["control_calcification"].mean(),
                    }
                )

        if reliability_by_control:
            reliability_df = pd.DataFrame(reliability_by_control)
            for i, (_, row) in enumerate(reliability_df.iterrows()):
                axes[1, 0].scatter(
                    row["mean_control"],
                    row["reliability"],
                    label=row["effect_type"],
                    s=100,
                    color=colors[i],
                )

        axes[1, 0].set_xlabel("Mean Control Value", fontsize=12)
        axes[1, 0].set_ylabel("Correlation with Raw Differences", fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].set_title(
            "Overall Effect Type Reliability", fontsize=14, fontweight="bold"
        )
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Effect reliability vs error magnitude
        reliability_by_error = []
        for effect_type in self.effect_types:
            valid_mask = ~(self.raw_diff.isna() | self.df[effect_type].isna())
            if valid_mask.sum() > 5:
                correlation = np.corrcoef(
                    self.raw_diff[valid_mask], self.df[effect_type][valid_mask]
                )[0, 1]
                reliability_by_error.append(
                    {
                        "effect_type": effect_type,
                        "reliability": correlation,
                        "mean_error": self.df["control_calcification_sd"].mean(),
                    }
                )

        if reliability_by_error:
            reliability_df = pd.DataFrame(reliability_by_error)
            for i, (_, row) in enumerate(reliability_df.iterrows()):
                axes[1, 1].scatter(
                    row["mean_error"],
                    row["reliability"],
                    label=row["effect_type"],
                    s=100,
                    color=colors[i],
                )

        axes[1, 1].set_xlabel("Mean Error Magnitude", fontsize=12)
        axes[1, 1].set_ylabel("Correlation with Raw Differences", fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].set_title(
            "Effect Type Reliability vs Error", fontsize=14, fontweight="bold"
        )
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_reliability_distributions(self, n_bins: int = 5) -> plt.Figure:
        """
        Show distribution of effect type reliability across different conditions.

        Args:
            n_bins: Number of bins for control values and error magnitudes

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Calculate overall reliability for each effect type
        reliability_data = []
        for effect_type in self.effect_types:
            valid_mask = ~(self.raw_diff.isna() | self.df[effect_type].isna())
            if valid_mask.sum() > 5:
                correlation = np.corrcoef(
                    self.raw_diff[valid_mask], self.df[effect_type][valid_mask]
                )[0, 1]
                reliability_data.append(
                    {
                        "effect_type": effect_type,
                        "reliability": correlation,
                        "sample_size": valid_mask.sum(),
                    }
                )

        reliability_df = pd.DataFrame(reliability_data)

        # Plot 1: Overall reliability bar plot
        if not reliability_df.empty:
            colors = plt.cm.Set1(np.linspace(0, 1, len(reliability_df)))
            bars = axes[0, 0].bar(
                reliability_df["effect_type"],
                reliability_df["reliability"],
                color=colors,
            )
            axes[0, 0].set_ylabel("Correlation with Raw Differences", fontsize=12)
            axes[0, 0].set_title(
                "Overall Effect Type Reliability", fontsize=14, fontweight="bold"
            )
            axes[0, 0].tick_params(axis="x", rotation=45)
            axes[0, 0].grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for bar, value in zip(bars, reliability_df["reliability"]):
                height = bar.get_height()
                axes[0, 0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
        else:
            axes[0, 0].text(
                0.5,
                0.5,
                "No reliability data available",
                ha="center",
                va="center",
                transform=axes[0, 0].transAxes,
            )
            axes[0, 0].set_title("Overall Effect Type Reliability")

        # Plot 2: Reliability by control value bins
        try:
            control_bins = pd.qcut(
                self.df["control_calcification"],
                q=n_bins,
                labels=["Very Low", "Low", "Medium", "High", "Very High"],
            )

            reliability_by_control = []
            for effect_type in self.effect_types:
                for bin_name in control_bins.unique():
                    mask = (control_bins == bin_name) & ~(
                        self.raw_diff.isna() | self.df[effect_type].isna()
                    )
                    if mask.sum() > 5:
                        correlation = np.corrcoef(
                            self.raw_diff[mask], self.df[effect_type][mask]
                        )[0, 1]
                        reliability_by_control.append(
                            {
                                "effect_type": effect_type,
                                "control_level": bin_name,
                                "reliability": correlation,
                            }
                        )

            if reliability_by_control:
                reliability_control_df = pd.DataFrame(reliability_by_control)
                colors = plt.cm.Set1(np.linspace(0, 1, len(self.effect_types)))

                for i, effect_type in enumerate(self.effect_types):
                    subset = reliability_control_df[
                        reliability_control_df["effect_type"] == effect_type
                    ]
                    if not subset.empty:
                        axes[0, 1].plot(
                            range(len(subset)),
                            subset["reliability"],
                            marker="o",
                            label=effect_type,
                            color=colors[i],
                            linewidth=2,
                            markersize=8,
                        )

                axes[0, 1].set_xlabel("Control Value Level", fontsize=12)
                axes[0, 1].set_ylabel("Reliability", fontsize=12)
                axes[0, 1].set_title(
                    "Reliability by Control Value", fontsize=14, fontweight="bold"
                )
                axes[0, 1].legend(fontsize=10)
                axes[0, 1].set_xticks(range(n_bins))
                axes[0, 1].set_xticklabels(
                    ["Very Low", "Low", "Medium", "High", "Very High"]
                )
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(
                    0.5,
                    0.5,
                    "No control-level reliability data",
                    ha="center",
                    va="center",
                    transform=axes[0, 1].transAxes,
                )
                axes[0, 1].set_title("Reliability by Control Value")
        except Exception:
            axes[0, 1].text(
                0.5,
                0.5,
                "Error calculating control-level reliability",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )
            axes[0, 1].set_title("Reliability by Control Value")

        # Plot 3: Reliability by error magnitude bins
        try:
            error_bins = pd.qcut(
                self.df["control_calcification_sd"],
                q=n_bins,
                labels=["Very Low", "Low", "Medium", "High", "Very High"],
            )

            reliability_by_error = []
            for effect_type in self.effect_types:
                for bin_name in error_bins.unique():
                    mask = (error_bins == bin_name) & ~(
                        self.raw_diff.isna() | self.df[effect_type].isna()
                    )
                    if mask.sum() > 5:
                        correlation = np.corrcoef(
                            self.raw_diff[mask], self.df[effect_type][mask]
                        )[0, 1]
                        reliability_by_error.append(
                            {
                                "effect_type": effect_type,
                                "error_level": bin_name,
                                "reliability": correlation,
                            }
                        )

            if reliability_by_error:
                reliability_error_df = pd.DataFrame(reliability_by_error)
                colors = plt.cm.Set1(np.linspace(0, 1, len(self.effect_types)))

                for i, effect_type in enumerate(self.effect_types):
                    subset = reliability_error_df[
                        reliability_error_df["effect_type"] == effect_type
                    ]
                    if not subset.empty:
                        axes[1, 0].plot(
                            range(len(subset)),
                            subset["reliability"],
                            marker="o",
                            label=effect_type,
                            color=colors[i],
                            linewidth=2,
                            markersize=8,
                        )

                axes[1, 0].set_xlabel("Error Magnitude Level", fontsize=12)
                axes[1, 0].set_ylabel("Reliability", fontsize=12)
                axes[1, 0].set_title(
                    "Reliability by Error Magnitude", fontsize=14, fontweight="bold"
                )
                axes[1, 0].legend(fontsize=10)
                axes[1, 0].set_xticks(range(n_bins))
                axes[1, 0].set_xticklabels(
                    ["Very Low", "Low", "Medium", "High", "Very High"]
                )
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(
                    0.5,
                    0.5,
                    "No error-level reliability data",
                    ha="center",
                    va="center",
                    transform=axes[1, 0].transAxes,
                )
                axes[1, 0].set_title("Reliability by Error Magnitude")
        except Exception:
            axes[1, 0].text(
                0.5,
                0.5,
                "Error calculating error-level reliability",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_title("Reliability by Error Magnitude")

        # Plot 4: Sample size distribution
        sample_sizes = []
        effect_names = []
        for effect_type in self.effect_types:
            valid_mask = ~(self.raw_diff.isna() | self.df[effect_type].isna())
            sample_sizes.append(valid_mask.sum())
            effect_names.append(effect_type)

        if sample_sizes:
            colors = plt.cm.Set1(np.linspace(0, 1, len(sample_sizes)))
            bars = axes[1, 1].bar(effect_names, sample_sizes, color=colors)
            axes[1, 1].set_ylabel("Sample Size", fontsize=12)
            axes[1, 1].set_title(
                "Sample Size by Effect Type", fontsize=14, fontweight="bold"
            )
            axes[1, 1].tick_params(axis="x", rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for bar, value in zip(bars, sample_sizes):
                height = bar.get_height()
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(sample_sizes) * 0.01,
                    f"{value}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No sample size data available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Sample Size by Effect Type")

        plt.tight_layout()
        return fig

    def generate_summary_report(self) -> pd.DataFrame:
        """
        Generate a summary report of effect type reliability.

        Returns:
            DataFrame with summary statistics
        """
        summary_data = []

        for effect_type in self.effect_types:
            valid_mask = ~(self.raw_diff.isna() | self.df[effect_type].isna())

            if valid_mask.sum() > 5:
                correlation = np.corrcoef(
                    self.raw_diff[valid_mask], self.df[effect_type][valid_mask]
                )[0, 1]

                # Calculate additional statistics
                effect_values = self.df[effect_type][valid_mask]
                raw_values = self.raw_diff[valid_mask]

                # Mean absolute error
                mae = np.mean(np.abs(effect_values - raw_values))

                # Root mean square error
                rmse = np.sqrt(np.mean((effect_values - raw_values) ** 2))

                # R-squared
                r_squared = correlation**2

                summary_data.append(
                    {
                        "effect_type": effect_type,
                        "correlation": correlation,
                        "r_squared": r_squared,
                        "mae": mae,
                        "rmse": rmse,
                        "sample_size": valid_mask.sum(),
                        "mean_effect": effect_values.mean(),
                        "std_effect": effect_values.std(),
                        "min_effect": effect_values.min(),
                        "max_effect": effect_values.max(),
                    }
                )

        summary_df = pd.DataFrame(summary_data)

        if not summary_df.empty:
            # Sort by correlation (best first)
            summary_df = summary_df.sort_values("correlation", ascending=False)

        return summary_df


def run_complete_reliability_analysis(
    df: pd.DataFrame,
    effect_types: Optional[List[str]] = None,
    save_plots: bool = False,
    output_dir: str = "./",
) -> dict:
    """
    Run a complete reliability analysis and generate all visualizations.

    Args:
        df: DataFrame with effect size data
        effect_types: List of effect type columns to analyze
        save_plots: Whether to save plots to files
        output_dir: Directory to save plots (if save_plots=True)

    Returns:
        Dictionary containing all analysis results
    """
    # Initialize analyzer
    analyzer = EffectTypeReliabilityAnalyzer(df, effect_types)

    # Generate all analyses
    results = {}

    # 1. Reliability heatmap
    print("Generating reliability heatmap...")
    heatmap_fig, heatmap_data = analyzer.create_reliability_heatmap()
    results["heatmap_fig"] = heatmap_fig
    results["heatmap_data"] = heatmap_data

    if save_plots and heatmap_fig:
        heatmap_fig.savefig(
            f"{output_dir}/effect_type_reliability_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )

    # 2. Effect type comparison
    print("Generating effect type comparison...")
    comparison_fig = analyzer.compare_effect_types()
    results["comparison_fig"] = comparison_fig

    if save_plots:
        comparison_fig.savefig(
            f"{output_dir}/effect_type_comparison.png", dpi=300, bbox_inches="tight"
        )

    # 3. Reliability distributions
    print("Generating reliability distributions...")
    distribution_fig = analyzer.plot_reliability_distributions()
    results["distribution_fig"] = distribution_fig

    if save_plots:
        distribution_fig.savefig(
            f"{output_dir}/reliability_distributions.png", dpi=300, bbox_inches="tight"
        )

    # 4. Summary report
    print("Generating summary report...")
    summary_df = analyzer.generate_summary_report()
    results["summary_df"] = summary_df

    if save_plots:
        summary_df.to_csv(
            f"{output_dir}/effect_type_reliability_summary.csv", index=False
        )

    print("Analysis complete!")
    return results


# Example usage functions
def quick_reliability_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick check of effect type reliability - returns summary table only.

    Args:
        df: DataFrame with effect size data

    Returns:
        Summary DataFrame
    """
    analyzer = EffectTypeReliabilityAnalyzer(df)
    return analyzer.generate_summary_report()


def plot_effect_type_comparison(
    df: pd.DataFrame, effect_types: Optional[List[str]] = None
) -> plt.Figure:
    """
    Quick function to just generate the effect type comparison plot.

    Args:
        df: DataFrame with effect size data
        effect_types: List of effect type columns to analyze

    Returns:
        matplotlib Figure
    """
    analyzer = EffectTypeReliabilityAnalyzer(df, effect_types)
    return analyzer.compare_effect_types()
