#!/usr/bin/env python3
"""
Migration Guide for Unified Metafor Model

This script demonstrates how to migrate from the old implementations
(meta_regression.py and hybrid_metafor_adapter.py) to the new unified
metafor_unified.py module.

Examples show equivalent functionality and highlight improvements.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.metafor_unified import UnifiedMetaforModel, fit_metafor_model


def migration_example_basic():
    """
    Example: Basic model fitting migration

    OLD (meta_regression.py):
        from calcification.analysis.meta_regression import MetaforModel
        model = MetaforModel(df, effect_type="hedges_g")
        model.fit_model()
        coefficients = model.get_coefficients()

    NEW (unified):
        from app.metafor_unified import UnifiedMetaforModel
        model = UnifiedMetaforModel(df, effect_type="hedges_g")
        model.fit_model()
        coefficients = model.get_coefficients()
    """
    print("=== Basic Model Fitting Migration ===")

    # Create sample data for demonstration
    np.random.seed(42)
    sample_data = pd.DataFrame(
        {
            "st_relative_calcification": np.random.normal(0, 1, 100),
            "st_relative_calcification_var": np.random.exponential(0.1, 100),
            "original_doi": [f"doi_{i // 10}" for i in range(100)],
            "ID": range(100),
            "delta_t": np.random.normal(2, 1, 100),
            "delta_ph": np.random.normal(-0.3, 0.1, 100),
            "treatment": np.random.choice(["OA", "Control"], 100),
        }
    )

    print("Creating and fitting model with unified implementation...")

    # NEW unified approach
    model = UnifiedMetaforModel(
        df=sample_data,
        effect_type="st_relative_calcification",
        formula="st_relative_calcification ~ delta_t + delta_ph - 1",
        verbose=True,
    )

    # Fit the model
    fitted_model = model.fit_model()

    # Get results
    coefficients = fitted_model.get_coefficients()
    coef_df = fitted_model.get_coefficients_dataframe()
    summary_text = fitted_model.get_model_summary_text()

    print("Model fitted successfully!")
    print(f"Coefficients shape: {coefficients.shape}")
    print(f"Coefficient names: {fitted_model.get_coefficient_names()}")
    print("\nCoefficients DataFrame:")
    print(coef_df)

    return fitted_model


def migration_example_streamlit_adapter():
    """
    Example: Streamlit adapter migration

    OLD (hybrid_metafor_adapter.py):
        from app.hybrid_metafor_adapter import StreamlitMetaforAdapter
        adapter = StreamlitMetaforAdapter(df, effect_type="hedges_g")
        adapter.fit_model()
        coef_df = adapter.get_coefficients_dataframe()

    NEW (unified):
        from app.metafor_unified import UnifiedMetaforModel
        model = UnifiedMetaforModel(df, effect_type="hedges_g")
        model.fit_model()
        coef_df = model.get_coefficients_dataframe()
    """
    print("\n=== Streamlit Adapter Migration ===")

    # Sample data
    np.random.seed(123)
    sample_data = pd.DataFrame(
        {
            "hedges_g": np.random.normal(0.5, 0.8, 50),
            "hedges_g_var": np.random.exponential(0.15, 50),
            "original_doi": [f"study_{i // 5}" for i in range(50)],
            "ID": range(50),
            "temp_change": np.random.normal(3, 1.5, 50),
            "ph_change": np.random.normal(-0.4, 0.15, 50),
            "species": np.random.choice(["A", "B", "C"], 50),
        }
    )

    print("Creating model with factor terms...")

    # NEW unified approach with factor terms
    model = UnifiedMetaforModel(
        df=sample_data,
        effect_type="hedges_g",
        formula="hedges_g ~ temp_change + ph_change + factor(species) - 1",
        verbose=True,
    )

    fitted_model = model.fit_model()

    # Demonstrate Streamlit-ready features
    coef_df = fitted_model.get_coefficients_dataframe()
    metadata = fitted_model.get_model_metadata()

    print("Model metadata:")
    for key, value in metadata.items():
        if isinstance(value, (list, dict)) and len(str(value)) > 50:
            print(f"  {key}: {type(value).__name__} with {len(value)} items")
        else:
            print(f"  {key}: {value}")

    print("\nStreamlit-ready coefficients DataFrame:")
    print(coef_df)

    return fitted_model


def migration_example_prediction():
    """
    Example: Prediction functionality migration

    OLD (meta_regression.py):
        model.predict_on_moderator_values(moderator_names, moderator_vals)

    NEW (unified):
        model.predict_on_moderator_values(moderator_names, moderator_vals)
        # Same API, but with improved context management
    """
    print("\n=== Prediction Migration ===")

    # Use model from previous example
    model = migration_example_basic()

    # Create prediction points
    prediction_points = np.array(
        [
            [1.0, -0.2],  # delta_t=1.0, delta_ph=-0.2
            [2.0, -0.3],  # delta_t=2.0, delta_ph=-0.3
            [3.0, -0.4],  # delta_t=3.0, delta_ph=-0.4
        ]
    )

    print("Making predictions...")

    # Make predictions (same API as before)
    predictions = model.predict_on_moderator_values(
        moderator_names=["delta_t", "delta_ph"],
        moderator_vals=prediction_points,
        confidence_level=95,
    )

    print("Predictions:")
    print(predictions)

    return predictions


def convenience_functions_example():
    """
    Example: Using convenience functions for quick model creation

    NEW convenience functions:
        create_metafor_model()  # Create but don't fit
        fit_metafor_model()     # Create and fit in one step
    """
    print("\n=== Convenience Functions ===")

    # Sample data
    np.random.seed(456)
    sample_data = pd.DataFrame(
        {
            "st_relative_calcification": np.random.normal(-0.2, 0.6, 30),
            "st_relative_calcification_var": np.random.exponential(0.12, 30),
            "original_doi": [f"paper_{i // 3}" for i in range(30)],
            "ID": range(30),
            "temperature": np.random.normal(25, 3, 30),
            "aragonite_sat": np.random.normal(2.5, 0.5, 30),
        }
    )

    print("Using convenience function to create and fit model...")

    # One-step model creation and fitting
    model = fit_metafor_model(
        df=sample_data,
        effect_type="st_relative_calcification",
        formula="st_relative_calcification ~ temperature + aragonite_sat - 1",
    )

    print("Model summary:")
    print(model.get_model_summary_text())

    return model


def advanced_features_example():
    """
    Example: Advanced features in the unified implementation

    NEW features:
        - Better error handling with context managers
        - Comprehensive metadata extraction
        - Improved coefficient name parsing
        - Robust factor variable handling
    """
    print("\n=== Advanced Features ===")

    # Sample data with complex structure
    np.random.seed(789)
    sample_data = pd.DataFrame(
        {
            "effect_size": np.random.normal(0.3, 0.7, 80),
            "effect_size_var": np.random.exponential(0.1, 80),
            "original_doi": [f"study_{i // 8}" for i in range(80)],
            "ID": range(80),
            "moderator1": np.random.normal(0, 1, 80),
            "moderator2": np.random.normal(1, 0.5, 80),
            "category": np.random.choice(["Group1", "Group2", "Group3"], 80),
            "binary_factor": np.random.choice(["Yes", "No"], 80),
        }
    )

    print("Creating model with complex formula including interactions...")

    # Complex formula with interactions and factors
    model = UnifiedMetaforModel(
        df=sample_data,
        effect_type="effect_size",
        formula="effect_size ~ moderator1 + moderator2 + factor(category) + "
        "moderator1:moderator2 + factor(binary_factor) - 1",
        verbose=True,
    )

    fitted_model = model.fit_model()

    # Demonstrate advanced features
    print("\nAdvanced model information:")
    print(f"Model representation: {repr(fitted_model)}")

    metadata = fitted_model.get_model_metadata()
    print(f"\nFormula components: {metadata['formula_components']}")
    print(f"Number of coefficients: {metadata['n_coefficients']}")
    print(f"Coefficient names: {metadata['coefficient_names']}")

    # Show detailed coefficients
    coef_df = fitted_model.get_coefficients_dataframe()
    print("\nDetailed coefficients with significance:")
    print(coef_df)

    return fitted_model


def migration_benefits():
    """
    Summary of benefits when migrating to the unified implementation
    """
    print("\n" + "=" * 60)
    print("MIGRATION BENEFITS")
    print("=" * 60)

    benefits = [
        "✅ Single unified API instead of multiple classes",
        "✅ Streamlit-safe context management for all R operations",
        "✅ Better error handling and recovery",
        "✅ Comprehensive model metadata extraction",
        "✅ Improved coefficient name parsing",
        "✅ Robust factor variable handling",
        "✅ Python-friendly data structures",
        "✅ Backwards compatible API",
        "✅ Enhanced logging and debugging",
        "✅ Consistent interface for all operations",
        "✅ Reduced code duplication",
        "✅ Better documentation and type hints",
    ]

    for benefit in benefits:
        print(benefit)

    print("\nKey architectural improvements:")
    print("• Context managers prevent rpy2/Streamlit conflicts")
    print("• Unified coefficient extraction works across formula types")
    print("• Better separation of R operations from Python data handling")
    print("• Comprehensive error handling with informative messages")
    print("• Support for complex formulas with interactions and factors")


def main():
    """Run all migration examples"""
    print("METAFOR UNIFIED MIGRATION GUIDE")
    print("=" * 50)

    try:
        # Run examples
        migration_example_basic()
        migration_example_streamlit_adapter()
        migration_example_prediction()
        convenience_functions_example()
        advanced_features_example()
        migration_benefits()

        print("\n" + "=" * 60)
        print("✅ All migration examples completed successfully!")
        print("✅ Ready to replace old implementations with metafor_unified.py")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error in migration examples: {e}")
        print("This might be due to missing R packages or data issues.")
        print(
            "The unified module is ready for use when R environment is properly configured."
        )


if __name__ == "__main__":
    main()
