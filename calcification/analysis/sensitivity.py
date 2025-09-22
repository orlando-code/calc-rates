"""
Leave-one-out cross-validation for meta-analysis models.

This module provides functionality to perform leave-one-study-out cross-validation
for meta-analysis, removing one study at a time and fitting the model on the
remaining data to assess model stability and influence of individual studies.
"""

import logging
from typing import Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from app.metafor import MetaforModel

logger = logging.getLogger(__name__)


def leave_one_out(
    df: pd.DataFrame,
    effect_type: str,
    treatment: Union[str, list],
    formula: str,
    random_structure: str = "~ 1 | original_doi",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Perform leave-one-study-out cross-validation for meta-analysis.

    Args:
        df (pd.DataFrame): Full dataset containing all studies
        effect_type (str): Name of the effect size column (e.g., 'st_relative_calcification')
        treatment (str or list): Treatment type(s) to filter for
        formula (str): Model formula (e.g., '~ delta_t + I(delta_t^2)')
        random_structure (str): Random effects structure, default: "~ 1 | original_doi"
        verbose (bool): Whether to print progress and summary statistics

    Returns:
        pd.DataFrame: Results dataframe with model statistics for each excluded study.
                     Each row represents a model fitted with one study excluded.

    Columns include:
        - excluded_study: DOI of the excluded study
        - n_studies_excluded: Number of studies excluded
        - n_training_studies: Number of studies used for training
        - n_training_observations: Number of observations used for training
        - coef_*: Coefficient values for each model parameter
        - LogLik, AIC, AICc, BIC: Model fit statistics
        - residual_heterogeneity_QE: Residual heterogeneity statistic
        - model_test_QM: Model test statistic
        - excluded_study_*: Statistics about the excluded study
    """

    # Validate inputs
    if "original_doi" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'original_doi' column for study identification"
        )

    if effect_type not in df.columns:
        raise ValueError(f"Effect type '{effect_type}' not found in DataFrame columns")

    # Get unique studies
    unique_studies = df["original_doi"].unique()
    n_studies = len(unique_studies)

    if verbose:
        print(f"🔬 Performing leave-one-out analysis for {n_studies} studies...")
        print(f"📊 Effect type: {effect_type}")
        print(f"🧪 Treatment: {treatment}")
        print(f"📐 Formula: {formula}")
        print(f"🎲 Random structure: {random_structure}")
        print("-" * 60)

    results = []
    n_successful = 0
    n_failed = 0

    # Progress bar for studies
    iterator = tqdm(unique_studies, desc="Leave-one-out CV", unit="study")

    for excluded_study in iterator:
        try:
            # Create dataset excluding one study
            train_df = df[df["original_doi"] != excluded_study].copy()
            excluded_data = df[df["original_doi"] == excluded_study].copy()

            # Skip if training set too small
            if len(train_df) < 3:
                if verbose:
                    logger.warning(
                        f"⚠️ Skipping {excluded_study}: insufficient training data ({len(train_df)} observations)"
                    )
                continue

            # Skip if no unique studies left
            if len(train_df["original_doi"].unique()) < 2:
                if verbose:
                    logger.warning(
                        f"⚠️ Skipping {excluded_study}: insufficient training studies"
                    )
                continue

            # Fit model on reduced dataset
            model = MetaforModel(
                df=train_df,
                effect_type=effect_type,
                treatment=treatment,
                formula=formula,
                random=random_structure,
                verbose=False,  # Keep quiet during loop
            )

            fitted_model = model.fit_model()

            # Extract basic model information
            model_stats = {
                "excluded_study": excluded_study,
                "n_studies_excluded": len(excluded_data["original_doi"].unique()),
                "n_observations_excluded": len(excluded_data),
                "n_training_studies": len(train_df["original_doi"].unique()),
                "n_training_observations": len(train_df),
                "formula": formula,
                "random_structure": random_structure,
                "successful_fit": True,
            }

            # Extract coefficient information
            if (
                hasattr(fitted_model, "coefficients")
                and fitted_model.coefficients is not None
            ):
                coef_names = (
                    fitted_model.coefficient_names
                    if hasattr(fitted_model, "coefficient_names")
                    else []
                )
                for i, coef_name in enumerate(coef_names):
                    if i < len(fitted_model.coefficients):
                        # Clean coefficient name for column naming
                        clean_name = (
                            coef_name.replace(":", "_")
                            .replace("(", "")
                            .replace(")", "")
                            .replace(" ", "_")
                        )
                        model_stats[f"coef_{clean_name}"] = fitted_model.coefficients[i]

            # Extract key model statistics from model_dict
            if hasattr(fitted_model, "model_dict") and fitted_model.model_dict:
                stats_to_extract = {
                    "k": "n_samples",
                    "QE": "residual_heterogeneity_QE",
                    "QEp": "QE_pvalue",
                    "QM": "model_test_QM",
                    "QMp": "QM_pvalue",
                    "AICc": "AICc",
                    "BIC": "BIC",
                    "LogLik": "LogLik",
                }

                for key, new_name in stats_to_extract.items():
                    if key in fitted_model.model_dict:
                        value = fitted_model.model_dict[key]
                        if isinstance(value, list) and len(value) > 0:
                            model_stats[new_name] = value[0]
                        elif isinstance(value, (int, float, np.number)):
                            model_stats[new_name] = value

                # Extract fit statistics (LogLik, AIC, AICc, BIC)
                if "fit.stats" in fitted_model.model_dict:
                    fit_stats = fitted_model.model_dict["fit.stats"][
                        fitted_model.model_dict["method"][0]
                    ]
                    if isinstance(fit_stats, dict):
                        for stat_name, stat_dict in fit_stats.items():
                            if isinstance(stat_dict, dict) and "REML" in stat_dict:
                                model_stats[stat_name] = stat_dict["REML"]
                            elif isinstance(stat_dict, (int, float, np.number)):
                                model_stats[stat_name] = stat_dict

            # Add information about excluded study
            if len(excluded_data) > 0:
                model_stats["excluded_study_effect_mean"] = excluded_data[
                    effect_type
                ].mean()
                model_stats["excluded_study_effect_std"] = excluded_data[
                    effect_type
                ].std()
                model_stats["excluded_study_effect_min"] = excluded_data[
                    effect_type
                ].min()
                model_stats["excluded_study_effect_max"] = excluded_data[
                    effect_type
                ].max()

            results.append(model_stats)
            n_successful += 1

        except Exception as e:
            n_failed += 1
            if verbose:
                logger.error(f"❌ Failed to fit model excluding {excluded_study}: {e}")

            # Record the failure
            error_stats = {
                "excluded_study": excluded_study,
                "successful_fit": False,
                "error": str(e),
                "formula": formula,
                "random_structure": random_structure,
            }

            # Try to get basic info about excluded study even if model failed
            try:
                excluded_data = df[df["original_doi"] == excluded_study].copy()
                error_stats["n_studies_excluded"] = len(
                    excluded_data["original_doi"].unique()
                )
                error_stats["n_observations_excluded"] = len(excluded_data)
                if len(excluded_data) > 0:
                    error_stats["excluded_study_effect_mean"] = excluded_data[
                        effect_type
                    ].mean()
            except Exception as e:
                logger.error(f"❌ Failed to get basic info about excluded study: {e}")
                pass

            results.append(error_stats)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    if verbose:
        success_rate = (
            results_df["successful_fit"].sum() / len(results_df) * 100
            if len(results_df) > 0
            else 0
        )
        print("\n✅ Leave-one-out analysis complete!")
        print(
            f"📈 Success rate: {success_rate:.1f}% ({n_successful}/{n_successful + n_failed} models)"
        )
        print(f"📊 Results shape: {results_df.shape}")

        # Summary statistics for successful models
        successful_models = results_df[results_df["successful_fit"]]
        if len(successful_models) > 0:
            if "LogLik" in successful_models.columns:
                loglik_range = f"{successful_models['LogLik'].min():.3f} to {successful_models['LogLik'].max():.3f}"
                print(f"📉 LogLik range: {loglik_range}")
            if "AICc" in successful_models.columns:
                aicc_range = f"{successful_models['AICc'].min():.3f} to {successful_models['AICc'].max():.3f}"
                print(f"📊 AICc range: {aicc_range}")
            if "residual_heterogeneity_QE" in successful_models.columns:
                qe_range = f"{successful_models['residual_heterogeneity_QE'].min():.3f} to {successful_models['residual_heterogeneity_QE'].max():.3f}"
                print(f"🔀 QE range: {qe_range}")

        if n_failed > 0:
            print(f"⚠️ {n_failed} models failed to fit")

    return results_df


def summarize_leave_one_out_results(
    results_df: pd.DataFrame, verbose: bool = True
) -> dict:
    """
    Summarize the results of leave-one-out cross-validation.

    Args:
        results_df (pd.DataFrame): Results from leave_one_out function
        verbose (bool): Whether to print summary

    Returns:
        dict: Summary statistics
    """
    successful_models = results_df[results_df["successful_fit"]]
    n_successful = len(successful_models)
    n_total = len(results_df)

    summary = {
        "n_total_studies": n_total,
        "n_successful_models": n_successful,
        "success_rate": n_successful / n_total if n_total > 0 else 0,
        "n_failed_models": n_total - n_successful,
    }

    if n_successful > 0:
        # Statistical summaries for model fit statistics
        fit_stats = [
            "LogLik",
            "AIC",
            "AICc",
            "BIC",
            "residual_heterogeneity_QE",
            "model_test_QM",
        ]
        for stat in fit_stats:
            if stat in successful_models.columns:
                stat_data = successful_models[stat].dropna()
                if len(stat_data) > 0:
                    summary[f"{stat}_mean"] = stat_data.mean()
                    summary[f"{stat}_std"] = stat_data.std()
                    summary[f"{stat}_min"] = stat_data.min()
                    summary[f"{stat}_max"] = stat_data.max()

        # Coefficient stability analysis
        coef_cols = [
            col for col in successful_models.columns if col.startswith("coef_")
        ]
        for coef_col in coef_cols:
            coef_data = successful_models[coef_col].dropna()
            if len(coef_data) > 1:
                summary[f"{coef_col}_mean"] = coef_data.mean()
                summary[f"{coef_col}_std"] = coef_data.std()
                summary[f"{coef_col}_cv"] = (
                    coef_data.std() / abs(coef_data.mean())
                    if coef_data.mean() != 0
                    else np.inf
                )

    if verbose:
        print("📋 Leave-One-Out Cross-Validation Summary")
        print("=" * 50)
        print(f"Total studies analyzed: {summary['n_total_studies']}")
        print(f"Successful model fits: {summary['n_successful_models']}")
        print(f"Success rate: {summary['success_rate']:.1%}")

        if n_successful > 0:
            print("\n📊 Model Fit Statistics:")
            for stat in ["LogLik", "AICc", "residual_heterogeneity_QE"]:
                if f"{stat}_mean" in summary:
                    print(
                        f"  {stat}: {summary[f'{stat}_mean']:.3f} ± {summary[f'{stat}_std']:.3f}"
                    )

            print("\n🎯 Coefficient Stability:")
            coef_summary = {
                k: v
                for k, v in summary.items()
                if k.startswith("coef_") and k.endswith("_cv")
            }
            for coef, cv in coef_summary.items():
                coef_name = coef.replace("coef_", "").replace("_cv", "")
                print(f"  {coef_name}: CV = {cv:.3f}")

    return summary


def identify_influential_studies(
    results_df: pd.DataFrame, metric: str = "AICc", threshold_percentile: float = 90
) -> pd.DataFrame:
    """
    Identify influential studies based on changes in model fit when excluded.

    Args:
        results_df (pd.DataFrame): Results from leave_one_out function
        metric (str): Metric to use for influence detection ('AICc', 'LogLik', 'residual_heterogeneity_QE')
        threshold_percentile (float): Percentile threshold for identifying influential studies

    Returns:
        pd.DataFrame: Studies ranked by influence
    """
    successful_models = results_df[results_df["successful_fit"]]

    if metric not in successful_models.columns:
        raise ValueError(
            f"Metric '{metric}' not found in results. Available metrics: {successful_models.columns.tolist()}"
        )

    metric_data = successful_models[["excluded_study", metric]].dropna()

    # Calculate influence score (how much the metric changes when study is excluded)
    if metric in ["AICc", "AIC", "BIC"]:
        # For information criteria, lower is better, so high values when study excluded = influential
        metric_data["influence_score"] = metric_data[metric]
        ascending = False
    elif metric == "LogLik":
        # For log-likelihood, higher is better, so low values when study excluded = influential
        metric_data["influence_score"] = -metric_data[metric]
        ascending = False
    else:
        # For other metrics, use absolute deviation from median
        median_val = metric_data[metric].median()
        metric_data["influence_score"] = abs(metric_data[metric] - median_val)
        ascending = False

    # Rank studies by influence
    metric_data = metric_data.sort_values("influence_score", ascending=ascending)
    metric_data["influence_rank"] = range(1, len(metric_data) + 1)
    metric_data["influence_percentile"] = (
        metric_data["influence_rank"] / len(metric_data)
    ) * 100

    # Identify influential studies
    influential_mask = metric_data["influence_percentile"] >= threshold_percentile
    metric_data["is_influential"] = influential_mask

    print(
        f"🔍 Identified {influential_mask.sum()} influential studies (top {100 - threshold_percentile}% by {metric})"
    )

    return metric_data[
        [
            "excluded_study",
            metric,
            "influence_score",
            "influence_rank",
            "influence_percentile",
            "is_influential",
        ]
    ].sort_values("influence_percentile", ascending=False)


if __name__ == "__main__":
    # Example usage (requires actual data)
    print("📚 Leave-One-Out Cross-Validation for Meta-Analysis")
    print("This script provides functions for leave-one-out analysis.")
    print("Import and use: from app.leave_one_out import leave_one_out")
