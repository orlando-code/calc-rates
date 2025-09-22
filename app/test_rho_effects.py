"""
Test script to verify that rho parameter affects metafor model standard errors and statistics.

According to metafor documentation, rho should NOT change coefficient values but should
affect standard errors, confidence intervals, and test statistics.
"""

import pandas as pd

from app.metafor import MetaforModel


def test_rho_effects():
    """
    Test different rho values and check what changes in the model output.
    """
    print("🔬 Testing RHO Parameter Effects in Metafor Models")
    print("=" * 60)

    # Load data
    print("📊 Loading data...")
    df = pd.read_csv("data/clean/analysis_ready_data.csv")

    if df is None:
        print("❌ No data found!")
        return None

    # Filter for a small subset to make testing faster
    filtered_df = df[df["treatment"] == "OA"].head(100).copy()

    print(
        f"📊 Using {len(filtered_df)} observations from {len(filtered_df['original_doi'].unique())} studies"
    )

    # Test parameters
    effect_type = "st_relative_calcification"
    formula = "~ delta_t"
    random_structure = "~ 1 | original_doi"

    # Test different rho values
    rho_values = [0.0, 0.3, 0.5, 0.7, 0.9]

    results = []

    for rho in rho_values:
        print(f"\n🔄 Testing rho = {rho}")

        try:
            # Create model with specific rho value
            model = MetaforModel(
                df=filtered_df,
                effect_type=effect_type,
                formula=formula,
                random=random_structure,
                verbose=False,
                metafor_model_kwargs={"rho": rho},  # Pass rho through kwargs
            )

            # Fit model
            fitted_model = model.fit_model()

            # Extract results that should be affected by rho
            result = {
                "rho": rho,
                "fitted_successfully": True,
            }

            # Extract coefficients (should NOT change)
            if (
                hasattr(fitted_model, "coefficients")
                and fitted_model.coefficients is not None
            ):
                coef_names = (
                    fitted_model.coefficient_names
                    if hasattr(fitted_model, "coefficient_names")
                    else []
                )
                for i, name in enumerate(coef_names):
                    if i < len(fitted_model.coefficients):
                        result[f"coef_{name}"] = fitted_model.coefficients[i]

            # Extract standard errors and other statistics (SHOULD change)
            if hasattr(fitted_model, "model_dict") and fitted_model.model_dict:
                # Standard errors
                if "se" in fitted_model.model_dict:
                    se_values = fitted_model.model_dict["se"]
                    if isinstance(se_values, list):
                        for i, se in enumerate(se_values):
                            result[f"se_{i}"] = se

                # Test statistics
                if "zval" in fitted_model.model_dict:
                    z_values = fitted_model.model_dict["zval"]
                    if isinstance(z_values, list):
                        for i, z in enumerate(z_values):
                            result[f"zval_{i}"] = z

                # P-values
                if "pval" in fitted_model.model_dict:
                    p_values = fitted_model.model_dict["pval"]
                    if isinstance(p_values, list):
                        for i, p in enumerate(p_values):
                            result[f"pval_{i}"] = p

                # Confidence intervals
                if "ci.lb" in fitted_model.model_dict:
                    ci_lb = fitted_model.model_dict["ci.lb"]
                    if isinstance(ci_lb, list):
                        for i, ci in enumerate(ci_lb):
                            result[f"ci_lb_{i}"] = ci

                if "ci.ub" in fitted_model.model_dict:
                    ci_ub = fitted_model.model_dict["ci.ub"]
                    if isinstance(ci_ub, list):
                        for i, ci in enumerate(ci_ub):
                            result[f"ci_ub_{i}"] = ci

                # Model fit statistics
                for stat in ["QE", "QM", "LogLik", "AIC", "AICc", "BIC"]:
                    if stat in fitted_model.model_dict:
                        value = fitted_model.model_dict[stat]
                        if isinstance(value, list) and len(value) > 0:
                            result[stat] = value[0]
                        elif isinstance(value, (int, float)):
                            result[stat] = value

                # Extract fit.stats if available
                if "fit.stats" in fitted_model.model_dict:
                    fit_stats = fitted_model.model_dict["fit.stats"]
                    if isinstance(fit_stats, dict):
                        for stat_name, stat_dict in fit_stats.items():
                            if isinstance(stat_dict, dict) and "REML" in stat_dict:
                                result[f"fit_{stat_name}"] = stat_dict["REML"]
                            elif isinstance(stat_dict, (int, float)):
                                result[f"fit_{stat_name}"] = stat_dict

            results.append(result)
            print("  ✅ Model fitted successfully")

        except Exception as e:
            print(f"  ❌ Model failed: {e}")
            results.append({"rho": rho, "fitted_successfully": False, "error": str(e)})

    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("❌ No results to analyze")
        return None

    print("\n📊 Analysis of RHO Effects")
    print("=" * 40)

    # Check if rho is having any effect
    successful_results = results_df[results_df["fitted_successfully"] == True]

    if len(successful_results) == 0:
        print("❌ No successful model fits")
        return results_df

    print(f"✅ {len(successful_results)} successful models fitted")

    # Check coefficient stability (should be constant)
    coef_cols = [col for col in successful_results.columns if col.startswith("coef_")]
    print("\n🎯 Coefficient Stability (should NOT change with rho):")
    for col in coef_cols:
        if col in successful_results.columns:
            values = successful_results[col].dropna()
            if len(values) > 1:
                print(
                    f"  {col}: Range = {values.max() - values.min():.6f} (std = {values.std():.6f})"
                )

    # Check standard error variation (should change)
    se_cols = [col for col in successful_results.columns if col.startswith("se_")]
    print("\n📏 Standard Error Changes (SHOULD change with rho):")
    for col in se_cols:
        if col in successful_results.columns:
            values = successful_results[col].dropna()
            if len(values) > 1:
                print(
                    f"  {col}: Range = {values.max() - values.min():.6f} (std = {values.std():.6f})"
                )

    # Check test statistic variation (should change)
    test_cols = [col for col in successful_results.columns if col.startswith("zval_")]
    print("\n📈 Test Statistic Changes (SHOULD change with rho):")
    for col in test_cols:
        if col in successful_results.columns:
            values = successful_results[col].dropna()
            if len(values) > 1:
                print(
                    f"  {col}: Range = {values.max() - values.min():.6f} (std = {values.std():.6f})"
                )

    # Check confidence interval variation (should change)
    ci_cols = [col for col in successful_results.columns if col.startswith("ci_")]
    print("\n🎯 Confidence Interval Changes (SHOULD change with rho):")
    for col in ci_cols:
        if col in successful_results.columns:
            values = successful_results[col].dropna()
            if len(values) > 1:
                print(
                    f"  {col}: Range = {values.max() - values.min():.6f} (std = {values.std():.6f})"
                )

    # Summary assessment
    print("\n🔍 DIAGNOSIS:")

    # Check if ANY statistics are changing
    stats_changing = False
    for col in successful_results.columns:
        if col.startswith(("se_", "zval_", "pval_", "ci_")):
            values = successful_results[col].dropna()
            if len(values) > 1 and values.std() > 1e-10:  # Some meaningful variation
                stats_changing = True
                break

    if not stats_changing:
        print(
            "❌ RHO PROBLEM: No standard errors, test statistics, or CIs are changing!"
        )
        print("   This suggests rho is not being passed correctly to metafor.")
        print("   Check that 'rho' parameter is in metafor_model_kwargs.")
    else:
        print("✅ RHO WORKING: Standard errors/statistics are changing as expected.")
        print("   Coefficients staying constant is CORRECT behavior.")

    # Check if coefficients are inappropriately stable
    coef_changing = False
    for col in coef_cols:
        values = successful_results[col].dropna()
        if len(values) > 1 and values.std() > 1e-10:
            coef_changing = True
            break

    if coef_changing:
        print("⚠️ WARNING: Coefficients are changing - this is unexpected!")

    return results_df


def diagnose_rho_parameter_passing():
    """
    Quick test to see if rho is being passed correctly.
    """
    print("\n🔧 Testing RHO Parameter Passing")
    print("-" * 40)

    # Load minimal data
    data_loader = DataLoader(data_path="data/clean/analysis_ready_data.csv")
    df = data_loader.load_data()

    if df is None:
        print("❌ No data found!")
        return

    # Use tiny subset
    test_df = df[df["treatment"] == "OA"].head(20).copy()

    try:
        # Test with explicit rho
        model = MetaforModel(
            df=test_df,
            effect_type="st_relative_calcification",
            formula="~ delta_t",
            random="~ 1 | original_doi",
            verbose=True,  # Enable verbose to see what's happening
            metafor_model_kwargs={"rho": 0.5},
        )

        print("✅ Model created successfully with rho=0.5")
        print(f"   metafor_model_kwargs: {model.metafor_model_kwargs}")

        fitted_model = model.fit_model()
        print("✅ Model fitted successfully")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run diagnosis
    try:
        # First test basic rho passing
        if diagnose_rho_parameter_passing():
            print("\n" + "=" * 60)
            # Then run full rho effects test
            results_df = test_rho_effects()

            if results_df is not None:
                print(f"\n💾 Results shape: {results_df.shape}")
                print("\nSample results:")
                display_cols = ["rho", "fitted_successfully"] + [
                    col
                    for col in results_df.columns
                    if col.startswith(("coef_", "se_"))
                ][:5]
                print(results_df[display_cols].to_string(index=False))

    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback

        traceback.print_exc()
