"""
Marginal effects plotting for meta-analysis models.

This module creates marginal plots showing the effect of individual moderators
while holding other moderators constant at their mean values.
"""

import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.metafor import MetaforModel


def extract_moderator_effects(model: MetaforModel, verbose: bool = False) -> Dict:
    """
    Extract coefficient information for each moderator from the fitted model.

    Args:
        model: Fitted MetaforModel
        verbose: Whether to print detailed information

    Returns:
        Dictionary mapping moderator names to their coefficient information
    """
    if not model.fitted:
        raise RuntimeError("Model must be fitted before extracting effects")

    if not hasattr(model, "coefficients") or not hasattr(model, "coefficient_names"):
        raise RuntimeError("Model coefficients not available")

    coefficients = model.coefficients
    coef_names = model.coefficient_names

    if verbose:
        print(f"📊 Extracting effects for {len(coef_names)} coefficients")
        for i, name in enumerate(coef_names):
            coef_val = coefficients[i] if i < len(coefficients) else "N/A"
            print(f"  {name}: {coef_val}")

    # Group coefficients by moderator
    moderator_effects = {}

    for i, coef_name in enumerate(coef_names):
        if i >= len(coefficients):
            continue

        coef_val = coefficients[i]

        # Skip intercept
        if coef_name in ["(Intercept)", "intrcpt"]:
            moderator_effects["intercept"] = coef_val
            continue

        # Parse coefficient name to extract moderator
        moderator = parse_coefficient_name(coef_name)

        if moderator not in moderator_effects:
            moderator_effects[moderator] = {
                "linear": 0,
                "quadratic": 0,
                "interactions": {},
                "all_terms": [],
            }

        # Determine term type
        if "I(" in coef_name and "^2" in coef_name:
            # Quadratic term
            moderator_effects[moderator]["quadratic"] = coef_val
        elif ":" in coef_name:
            # Interaction term
            interaction_vars = coef_name.split(":")
            other_var = (
                interaction_vars[1]
                if interaction_vars[0] == moderator
                else interaction_vars[0]
            )
            moderator_effects[moderator]["interactions"][other_var] = coef_val
        else:
            # Linear term
            moderator_effects[moderator]["linear"] = coef_val

        moderator_effects[moderator]["all_terms"].append(
            {"name": coef_name, "coefficient": coef_val}
        )

    if verbose:
        print(
            f"\n📈 Extracted effects for moderators: {list(moderator_effects.keys())}"
        )

    return moderator_effects


def parse_coefficient_name(coef_name: str) -> str:
    """
    Parse coefficient name to extract the base moderator variable name.

    Args:
        coef_name: Coefficient name from model

    Returns:
        Base moderator name
    """
    # Handle different coefficient naming patterns

    # Remove function wrappers like I()
    clean_name = re.sub(r"I\((.*?)\)", r"\1", coef_name)

    # Handle interactions (take first variable)
    if ":" in clean_name:
        clean_name = clean_name.split(":")[0]

    # Remove power operators
    clean_name = re.sub(r"\^.*", "", clean_name)

    # Remove factor() wrapper
    clean_name = re.sub(r"factor\((.*?)\)", r"\1", clean_name)

    return clean_name.strip()


def calculate_marginal_effect(
    moderator_name: str,
    moderator_values: np.ndarray,
    moderator_effects: Dict,
    other_moderators: Dict[str, float],
    verbose: bool = False,
) -> np.ndarray:
    """
    Calculate the marginal effect of one moderator while holding others constant.

    Args:
        moderator_name: Name of the moderator to vary
        moderator_values: Array of values for this moderator
        moderator_effects: Dictionary of all moderator effects from model
        other_moderators: Dictionary of constant values for other moderators
        verbose: Whether to print calculation details

    Returns:
        Array of predicted effects
    """
    if moderator_name not in moderator_effects:
        raise ValueError(f"Moderator '{moderator_name}' not found in model effects")

    effects = moderator_effects[moderator_name]

    # Start with intercept
    predictions = np.full_like(
        moderator_values, moderator_effects.get("intercept", 0.0)
    )

    # Add linear effect of target moderator
    if effects["linear"] != 0:
        predictions += effects["linear"] * moderator_values
        if verbose:
            print(f"  Added linear term: {effects['linear']} * {moderator_name}")

    # Add quadratic effect of target moderator
    if effects["quadratic"] != 0:
        predictions += effects["quadratic"] * (moderator_values**2)
        if verbose:
            print(
                f"  Added quadratic term: {effects['quadratic']} * {moderator_name}^2"
            )

    # Add interaction effects with other moderators (held constant)
    for other_mod, interaction_coef in effects["interactions"].items():
        if other_mod in other_moderators:
            interaction_effect = (
                interaction_coef * moderator_values * other_moderators[other_mod]
            )
            predictions += interaction_effect
            if verbose:
                print(
                    f"  Added interaction: {interaction_coef} * {moderator_name} * {other_mod}({other_moderators[other_mod]})"
                )

    # Add effects of other moderators (constant contributions)
    for other_mod, other_val in other_moderators.items():
        if other_mod in moderator_effects and other_mod != moderator_name:
            other_effects = moderator_effects[other_mod]

            # Linear effect of other moderator
            if other_effects["linear"] != 0:
                predictions += other_effects["linear"] * other_val
                if verbose:
                    print(
                        f"  Added constant {other_mod} linear: {other_effects['linear']} * {other_val}"
                    )

            # Quadratic effect of other moderator
            if other_effects["quadratic"] != 0:
                predictions += other_effects["quadratic"] * (other_val**2)
                if verbose:
                    print(
                        f"  Added constant {other_mod} quadratic: {other_effects['quadratic']} * {other_val}^2"
                    )

    return predictions


def create_marginal_plots_plotly(
    model: MetaforModel,
    moderators: List[str],
    n_points: int = 50,
    constant_values: Optional[Dict[str, float]] = None,
    title: str = "Marginal Effects",
    width: int = 1000,
    height: int = 400,
) -> go.Figure:
    """
    Create interactive marginal effect plots using Plotly.

    Args:
        model: Fitted MetaforModel
        moderators: List of moderators to plot
        n_points: Number of points for each curve
        constant_values: Values to hold other moderators constant (defaults to means)
        title: Plot title
        width: Plot width
        height: Plot height

    Returns:
        Plotly figure with marginal effect plots
    """
    if len(moderators) == 0:
        raise ValueError("Must specify at least one moderator")

    # Extract effects from model
    moderator_effects = extract_moderator_effects(model)

    # Get data ranges for moderators
    moderator_ranges = {}
    for mod in moderators:
        if mod in model.df_processed.columns:
            data = model.df_processed[mod].dropna()
            moderator_ranges[mod] = {
                "min": data.min(),
                "max": data.max(),
                "mean": data.mean(),
                "std": data.std(),
            }
        else:
            print(f"⚠️ Warning: {mod} not found in data, using default range")
            moderator_ranges[mod] = {"min": -2, "max": 2, "mean": 0, "std": 1}

    # Set constant values for other moderators
    if constant_values is None:
        constant_values = {}

    # Fill in missing constant values with means
    all_moderators = [mod for mod in moderator_effects.keys() if mod != "intercept"]
    for mod in all_moderators:
        if mod not in constant_values and mod in moderator_ranges:
            constant_values[mod] = moderator_ranges[mod]["mean"]

    # Create subplots
    n_plots = len(moderators)
    cols = min(n_plots, 3)
    rows = (n_plots + cols - 1) // cols

    subplot_titles = [f"{mod.replace('_', ' ').title()} Effect" for mod in moderators]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        x_title="Moderator Value",
        y_title="Predicted Effect",
    )

    # Create marginal plots
    for i, moderator in enumerate(moderators):
        row = (i // cols) + 1
        col = (i % cols) + 1

        if moderator not in moderator_ranges:
            continue

        # Generate moderator values
        mod_range = moderator_ranges[moderator]
        moderator_values = np.linspace(mod_range["min"], mod_range["max"], n_points)

        # Calculate predictions
        other_mods = {k: v for k, v in constant_values.items() if k != moderator}
        predictions = calculate_marginal_effect(
            moderator, moderator_values, moderator_effects, other_mods
        )

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=moderator_values,
                y=predictions,
                mode="lines",
                name=f"{moderator} effect",
                line=dict(width=3),
                hovertemplate=f"<b>{moderator}</b>: %{{x:.3f}}<br><b>Effect</b>: %{{y:.3f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Update axis labels
        fig.update_xaxes(
            title_text=moderator.replace("_", " ").title(), row=row, col=col
        )
        fig.update_yaxes(title_text="Predicted Effect", row=row, col=col)

    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height * rows,
        showlegend=False,
        template="plotly_white",
    )

    return fig


def create_marginal_plots_matplotlib(
    model: MetaforModel,
    moderators: List[str],
    n_points: int = 50,
    constant_values: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create marginal effect plots using Matplotlib.

    Args:
        model: Fitted MetaforModel
        moderators: List of moderators to plot
        n_points: Number of points for each curve
        constant_values: Values to hold other moderators constant
        figsize: Figure size

    Returns:
        Matplotlib figure and axes array
    """
    # Extract effects from model
    moderator_effects = extract_moderator_effects(model)

    # Get data ranges
    moderator_ranges = {}
    for mod in moderators:
        if mod in model.df_processed.columns:
            data = model.df_processed[mod].dropna()
            moderator_ranges[mod] = {
                "min": data.min(),
                "max": data.max(),
                "mean": data.mean(),
            }
        else:
            moderator_ranges[mod] = {"min": -2, "max": 2, "mean": 0}

    # Set constant values
    if constant_values is None:
        constant_values = {}

    all_moderators = [mod for mod in moderator_effects.keys() if mod != "intercept"]
    for mod in all_moderators:
        if mod not in constant_values and mod in moderator_ranges:
            constant_values[mod] = moderator_ranges[mod]["mean"]

    # Create figure
    n_plots = len(moderators)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, moderator in enumerate(moderators):
        if i >= len(axes):
            break

        ax = axes[i]

        if moderator not in moderator_ranges:
            ax.text(
                0.5,
                0.5,
                f"No data for\n{moderator}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Generate values and predictions
        mod_range = moderator_ranges[moderator]
        moderator_values = np.linspace(mod_range["min"], mod_range["max"], n_points)

        other_mods = {k: v for k, v in constant_values.items() if k != moderator}
        predictions = calculate_marginal_effect(
            moderator, moderator_values, moderator_effects, other_mods
        )

        # Plot
        ax.plot(moderator_values, predictions, linewidth=3, color="steelblue")
        ax.set_xlabel(moderator.replace("_", " ").title())
        ax.set_ylabel("Predicted Effect")
        ax.set_title(f"{moderator.replace('_', ' ').title()} Effect")
        ax.grid(True, alpha=0.3)

    # Remove extra subplots
    for i in range(len(moderators), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    return fig, axes


def print_marginal_effects_summary(model: MetaforModel, moderators: List[str]) -> None:
    """
    Print a summary of marginal effects for specified moderators.

    Args:
        model: Fitted MetaforModel
        moderators: List of moderators to summarize
    """
    print("📊 Marginal Effects Summary")
    print("=" * 50)

    moderator_effects = extract_moderator_effects(model, verbose=False)

    print(f"Model Formula: {model.formula}")
    print(f"Intercept: {moderator_effects.get('intercept', 0):.4f}")
    print()

    for moderator in moderators:
        if moderator in moderator_effects:
            effects = moderator_effects[moderator]
            print(f"📈 {moderator.replace('_', ' ').title()}:")

            if effects["linear"] != 0:
                print(f"  Linear coefficient: {effects['linear']:.4f}")

            if effects["quadratic"] != 0:
                print(f"  Quadratic coefficient: {effects['quadratic']:.4f}")

            if effects["interactions"]:
                print("  Interactions:")
                for other_var, coef in effects["interactions"].items():
                    print(f"    with {other_var}: {coef:.4f}")

            print()
        else:
            print(f"⚠️ {moderator} not found in model")


# Example usage function
def example_marginal_plots():
    """
    Example of how to create marginal effects plots.
    """
    print("📊 Example: Creating Marginal Effects Plots")
    print("=" * 50)

    print("""
    # Example usage:
    from app.marginal_effects_plots import create_marginal_plots_plotly, print_marginal_effects_summary
    
    # Assuming you have a fitted model
    moderators = ['delta_t', 'delta_ph']
    
    # Print summary
    print_marginal_effects_summary(model, moderators)
    
    # Create interactive plots
    fig = create_marginal_plots_plotly(
        model, 
        moderators=moderators,
        title="Marginal Effects of Temperature and pH"
    )
    fig.show()
    
    # Or create static plots
    from app.marginal_effects_plots import create_marginal_plots_matplotlib
    fig, axes = create_marginal_plots_matplotlib(model, moderators)
    plt.show()
    """)


if __name__ == "__main__":
    example_marginal_plots()
