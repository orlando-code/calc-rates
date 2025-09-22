#!/usr/bin/env python3
"""
Example: Creating uncertainty surfaces for meta-analysis predictions.

This script demonstrates how to generate prediction surfaces with standard errors,
confidence intervals, and prediction intervals for plotting.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import metafor


def create_uncertainty_surfaces_example():
    """Example of creating prediction surfaces with uncertainty estimates."""

    # Load your data (replace with actual data path)
    try:
        df = pd.read_csv("data/tmp/coral_effects_df.csv")
    except FileNotFoundError:
        print("Please ensure coral_effects_df.csv exists in data/tmp/")
        return

    # Create and fit model
    model = metafor.MetaforModel(
        df,
        effect_type="yi",
        effect_type_var="vi",
        formula="yi ~ dt + dph + dt:dph",
        random="~ 1 | doi/ID",
        process_data=False,
        verbose=True,
    ).fit_model()

    print("✅ Model fitted successfully!")

    # Define prediction grid
    dt_range = np.linspace(df["dt"].min(), df["dt"].max(), 50)
    dph_range = np.linspace(df["dph"].min(), df["dph"].max(), 50)

    # Generate comprehensive prediction surfaces
    print("🔄 Generating prediction surfaces...")

    surfaces = model.predict_nd_surface_from_model(
        moderator_names=["dt", "dph"],
        moderator_values=[dt_range, dph_range],
        include_se=True,
        include_ci=True,
        include_pi=True,
        confidence_level=0.95,
    )

    print(f"✅ Generated surfaces: {list(surfaces.keys())}")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Meta-Analysis Prediction Surfaces with Uncertainty", fontsize=16)

    # Get meshgrids for plotting
    dt_mesh, dph_mesh = surfaces["meshgrids"]

    # Plot 1: Predictions
    im1 = axes[0, 0].contourf(
        dt_mesh, dph_mesh, surfaces["pred"], levels=20, cmap="RdYlBu_r"
    )
    axes[0, 0].set_title("Predictions")
    axes[0, 0].set_xlabel("Temperature Change (°C)")
    axes[0, 0].set_ylabel("pH Change")
    plt.colorbar(im1, ax=axes[0, 0])

    # Plot 2: Standard Errors
    im2 = axes[0, 1].contourf(
        dt_mesh, dph_mesh, surfaces["se"], levels=20, cmap="viridis"
    )
    axes[0, 1].set_title("Standard Errors")
    axes[0, 1].set_xlabel("Temperature Change (°C)")
    axes[0, 1].set_ylabel("pH Change")
    plt.colorbar(im2, ax=axes[0, 1])

    # Plot 3: Confidence Interval Width
    ci_width = surfaces["ci_ub"] - surfaces["ci_lb"]
    im3 = axes[0, 2].contourf(dt_mesh, dph_mesh, ci_width, levels=20, cmap="plasma")
    axes[0, 2].set_title("95% CI Width")
    axes[0, 2].set_xlabel("Temperature Change (°C)")
    axes[0, 2].set_ylabel("pH Change")
    plt.colorbar(im3, ax=axes[0, 2])

    # Plot 4: Lower Confidence Interval
    im4 = axes[1, 0].contourf(
        dt_mesh, dph_mesh, surfaces["ci_lb"], levels=20, cmap="RdYlBu_r"
    )
    axes[1, 0].set_title("95% CI Lower Bound")
    axes[1, 0].set_xlabel("Temperature Change (°C)")
    axes[1, 0].set_ylabel("pH Change")
    plt.colorbar(im4, ax=axes[1, 0])

    # Plot 5: Upper Confidence Interval
    im5 = axes[1, 1].contourf(
        dt_mesh, dph_mesh, surfaces["ci_ub"], levels=20, cmap="RdYlBu_r"
    )
    axes[1, 1].set_title("95% CI Upper Bound")
    axes[1, 1].set_xlabel("Temperature Change (°C)")
    axes[1, 1].set_ylabel("pH Change")
    plt.colorbar(im5, ax=axes[1, 1])

    # Plot 6: Prediction Interval Width
    pi_width = surfaces["pi_ub"] - surfaces["pi_lb"]
    im6 = axes[1, 2].contourf(dt_mesh, dph_mesh, pi_width, levels=20, cmap="magma")
    axes[1, 2].set_title("95% PI Width")
    axes[1, 2].set_xlabel("Temperature Change (°C)")
    axes[1, 2].set_ylabel("pH Change")
    plt.colorbar(im6, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig("uncertainty_surfaces.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✅ Plots saved as 'uncertainty_surfaces.png'")

    # Example: Extract data for specific point
    print("\n🔍 Example uncertainty at specific point:")
    mid_dt = len(dt_range) // 2
    mid_dph = len(dph_range) // 2

    print(f"At dt={dt_range[mid_dt]:.2f}, dph={dph_range[mid_dph]:.3f}:")
    print(f"  Prediction: {surfaces['pred'][mid_dt, mid_dph]:.3f}")
    print(f"  Standard Error: {surfaces['se'][mid_dt, mid_dph]:.3f}")
    print(
        f"  95% CI: [{surfaces['ci_lb'][mid_dt, mid_dph]:.3f}, {surfaces['ci_ub'][mid_dt, mid_dph]:.3f}]"
    )
    print(
        f"  95% PI: [{surfaces['pi_lb'][mid_dt, mid_dph]:.3f}, {surfaces['pi_ub'][mid_dt, mid_dph]:.3f}]"
    )


def compare_with_metafor_predict():
    """Compare surface predictions with metafor's native predict function."""

    # Load data
    try:
        df = pd.read_csv("data/tmp/coral_effects_df.csv")
    except FileNotFoundError:
        print("Please ensure coral_effects_df.csv exists in data/tmp/")
        return

    # Fit model
    model = metafor.MetaforModel(
        df,
        effect_type="yi",
        effect_type_var="vi",
        formula="yi ~ dt + dph + dt:dph",
        random="~ 1 | doi/ID",
        process_data=False,
    ).fit_model()

    # Test specific points
    test_points = np.array(
        [
            [1.0, -0.1],  # dt=1.0, dph=-0.1
            [2.0, -0.2],  # dt=2.0, dph=-0.2
            [3.0, -0.3],  # dt=3.0, dph=-0.3
        ]
    )

    print("🔄 Comparing surface method vs metafor predict...")

    # Method 1: Using our surface method
    dt_vals = test_points[:, 0]
    dph_vals = test_points[:, 1]

    surfaces = model.predict_nd_surface_from_model(
        moderator_names=["dt", "dph"],
        moderator_values=[dt_vals, dph_vals],
        include_se=True,
        include_ci=True,
    )

    # Method 2: Using metafor's predict function
    metafor_results = model.predict_on_moderator(
        ["dt", "dph"], n_points=len(test_points)
    )

    print("Comparison:")
    print("Point\tSurface_Pred\tMetafor_Pred\tDifference")
    for i in range(len(test_points)):
        surf_pred = surfaces["pred"].flatten()[i]
        metafor_pred = (
            metafor_results["pred"][i] if "pred" in metafor_results else "N/A"
        )
        diff = abs(surf_pred - metafor_pred) if metafor_pred != "N/A" else "N/A"
        print(f"{i + 1}\t{surf_pred:.6f}\t{metafor_pred}\t{diff}")


if __name__ == "__main__":
    create_uncertainty_surfaces_example()
    # compare_with_metafor_predict()



