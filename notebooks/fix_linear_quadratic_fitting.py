"""
Fix for the fit_linear_and_quadratic function issues

The original function has two major bugs:
1. Incorrect data sorting that scrambles the x-y relationship
2. Parameter order confusion in plotting

This script provides the corrected function and demonstrates the issues.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_linear_and_quadratic_BROKEN(x, y):
    """Original broken version - DO NOT USE"""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    # ❌ BUG: This sorting is wrong!
    x = np.sort(x)
    y = y[np.argsort(x)]  # This doesn't work when x is already sorted!

    # Linear fit
    linear_model = sm.OLS(y, sm.add_constant(x)).fit()
    # Quadratic fit
    X_quad = pd.DataFrame({"x": x, "x2": x**2})
    X_quad = sm.add_constant(X_quad)
    quadratic_model = sm.OLS(y, X_quad).fit()

    return (
        linear_model.params,
        linear_model.rsquared,
        quadratic_model.params,
        quadratic_model.rsquared,
    )


def fit_linear_and_quadratic_FIXED(x, y):
    """Fixed version - USE THIS"""
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    # ✅ FIXED: Sort both x and y together to maintain relationship
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    y = y[sort_indices]

    # Linear fit: y = ax + b
    linear_model = sm.OLS(y, sm.add_constant(x)).fit()

    # Quadratic fit: y = ax^2 + bx + c
    X_quad = pd.DataFrame({"x": x, "x2": x**2})
    X_quad = sm.add_constant(X_quad)
    quadratic_model = sm.OLS(y, X_quad).fit()

    return (
        linear_model.params,
        linear_model.rsquared,
        quadratic_model.params,
        quadratic_model.rsquared,
    )


def demonstrate_bug():
    """Demonstrate the sorting bug with a clear example"""
    print("🐛 DEMONSTRATING THE SORTING BUG")
    print("=" * 50)

    # Create test data with clear quadratic relationship
    x_true = np.array([1, 2, 3, 4, 5])
    y_true = x_true**2  # Perfect quadratic: y = x^2

    print("Original data:")
    print(f"x: {x_true}")
    print(f"y: {y_true}")
    print("Expected relationship: y = x^2")

    # Add some random order to simulate real data
    random_order = np.array([2, 0, 4, 1, 3])  # Scrambled indices
    x_scrambled = x_true[random_order]
    y_scrambled = y_true[random_order]

    print("\nScrambled data:")
    print(f"x: {x_scrambled}")
    print(f"y: {y_scrambled}")

    # Test broken function
    print("\n❌ BROKEN FUNCTION RESULTS:")
    linear_params_broken, _, quad_params_broken, _ = fit_linear_and_quadratic_BROKEN(
        x_scrambled, y_scrambled
    )
    print(f"Quadratic params (broken): {quad_params_broken}")

    # Test fixed function
    print("\n✅ FIXED FUNCTION RESULTS:")
    linear_params_fixed, _, quad_params_fixed, _ = fit_linear_and_quadratic_FIXED(
        x_scrambled, y_scrambled
    )
    print(f"Quadratic params (fixed): {quad_params_fixed}")

    print("\nExpected quadratic params for y = x^2:")
    print("const: 0, x: 0, x2: 1")


def plot_comparison(x, y, title="Data Fitting Comparison"):
    """Compare broken vs fixed fitting"""

    # Get fits from both functions
    linear_broken, _, quad_broken, _ = fit_linear_and_quadratic_BROKEN(x, y)
    linear_fixed, _, quad_fixed, _ = fit_linear_and_quadratic_FIXED(x, y)

    # Create plotting range
    x_plot = np.linspace(np.min(x), np.max(x), 100)

    # Calculate fits for plotting
    # BROKEN fits
    linear_plot_broken = linear_broken[0] + linear_broken[1] * x_plot
    quad_plot_broken = (
        quad_broken[0] + quad_broken[1] * x_plot + quad_broken[2] * x_plot**2
    )

    # FIXED fits (note: using .loc for named access)
    linear_plot_fixed = linear_fixed[0] + linear_fixed[1] * x_plot
    quad_plot_fixed = (
        quad_fixed.loc["const"]
        + quad_fixed.loc["x"] * x_plot
        + quad_fixed.loc["x2"] * x_plot**2
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Broken version
    ax1.scatter(x, y, alpha=0.6, c="black", s=30, label="Data")
    ax1.plot(x_plot, linear_plot_broken, "b-", label="Linear (broken)", linewidth=2)
    ax1.plot(x_plot, quad_plot_broken, "r-", label="Quadratic (broken)", linewidth=2)
    ax1.set_title("❌ Broken Function")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Fixed version
    ax2.scatter(x, y, alpha=0.6, c="black", s=30, label="Data")
    ax2.plot(x_plot, linear_plot_fixed, "b-", label="Linear (fixed)", linewidth=2)
    ax2.plot(x_plot, quad_plot_fixed, "r-", label="Quadratic (fixed)", linewidth=2)
    ax2.set_title("✅ Fixed Function")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return fig


def correct_plotting_code_example():
    """Show the correct way to use quadratic parameters"""
    print("\n📊 CORRECT PLOTTING CODE")
    print("=" * 30)

    print(
        "The quadratic model returns parameters as a pandas Series with named indices:"
    )
    print("- quad_params.loc['const']  # Intercept")
    print("- quad_params.loc['x']      # Linear coefficient")
    print("- quad_params.loc['x2']     # Quadratic coefficient")
    print()
    print("❌ WRONG (your current code):")
    print(
        "quadratic_plot = quadratic_params[0] + quadratic_params[1]*x + quadratic_params[2]*x**2"
    )
    print()
    print("✅ CORRECT:")
    print("quadratic_plot = (")
    print("    quadratic_params.loc['const'] +")
    print("    quadratic_params.loc['x'] * x +")
    print("    quadratic_params.loc['x2'] * x**2")
    print(")")


if __name__ == "__main__":
    # Demonstrate the bug
    demonstrate_bug()

    # Show correct plotting
    correct_plotting_code_example()

    # Test with realistic data
    print("\n🧪 TESTING WITH REALISTIC DATA")
    print("=" * 40)

    # Create some realistic temperature/pH vs calcification data
    np.random.seed(42)
    temp = np.random.uniform(18, 28, 50)  # Temperature range
    calc_temp = (
        100 - 2 * temp + 0.1 * temp**2 + np.random.normal(0, 5, 50)
    )  # Quadratic relationship with noise

    ph = np.random.uniform(7.8, 8.2, 50)  # pH range
    calc_ph = (
        -500 + 120 * ph - 7 * ph**2 + np.random.normal(0, 5, 50)
    )  # Different quadratic relationship

    # Plot comparison
    plot_comparison(temp, calc_temp, "Temperature vs Calcification")
    plot_comparison(ph, calc_ph, "pH vs Calcification")
