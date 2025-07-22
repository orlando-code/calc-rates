import pandas as pd
from rpy2.robjects import default_converter, r
from rpy2.robjects.conversion import localconverter
from scipy.interpolate import make_interp_spline


def summarize_metafor_models(model_summaries, model_names=None):
    """
    Extracts model diagnostics from a list of metafor model summaries (as ListVectors via rpy2).

    Parameters:
        model_summaries (list): List of <rpy2.robjects.vectors.ListVector> metafor summaries.

    Returns:
        pd.DataFrame: Summary table of key diagnostics.
    """
    summary_data = []

    for i, (summary, name) in enumerate(zip(model_summaries, model_names), start=1):
        with localconverter(default_converter) as _:
            # keys = [str(k) for k in list(summary.names)]
            topline_summary = list(r.fitstats(summary))
        # Convert R ListVector to Python dictionary
        s = {str(k): summary.rx2(str(k)) for k in summary.names}

        # Extract values safely
        loglik, deviance, AIC, BIC, AICc = [
            topline_summary[i] for i in range(len(topline_summary))
        ]

        QE_stat = s.get("QE", [None])[0]
        # QE_df = s.get("QE", [None])[1] if len(s.get("QE", [])) > 1 else None
        QE_pval = s.get("QEp", [None])[0]

        QM_stat = s.get("QM", [None])[0]
        # QM_df = s.get("QM", [None])[1] if len(s.get("QM", [])) > 1 else None
        QM_pval = s.get("QMp", [None])[0]

        var_comp = s.get("sigma2", [None, None])
        sigma2_study = var_comp[0]
        sigma2_within = var_comp[1] if len(var_comp) > 1 else None

        summary_data.append(
            {
                "Model": name,
                "Log-likelihood": f"{loglik:.0f}" if loglik is not None else None,
                "Deviance": f"{deviance:.0f}" if deviance is not None else None,
                "AIC": f"{AIC:.0f}" if AIC is not None else None,
                "AICc": f"{AICc:.0f}" if AICc is not None else None,
                # "QE (df)": f"{int(QE_df)}" if QE_df else None,
                "QE stat": f"{QE_stat:.0f}" if QE_stat is not None else None,
                "QE p-val": f"{QE_pval:.4f}" if QE_pval is not None else None,
                # "QM (df)": f"{int(QM_df)}" if QM_df else None,
                "QM stat": f"{QM_stat:.0f}" if QM_stat is not None else None,
                "QM p-val": f"{QM_pval:.4f}" if QM_pval is not None else None,
                "σ² (Study)": f"{sigma2_study:.0f}"
                if sigma2_study is not None
                else None,
                "σ² (Within)": f"{sigma2_within:.0f}"
                if sigma2_within is not None
                else None,
            }
        )

    return pd.DataFrame(summary_data).reset_index(drop=True)


def extrapolate_predictions(df, year=2100):
    grouping_cols = ["scenario", "percentile", "core_grouping", "time_frame"]
    value_cols = [col for col in df.columns if col not in grouping_cols]

    new_rows = []

    for (scenario, percentile, core_grouping), group_df in df.groupby(
        ["scenario", "percentile", "core_grouping"]
    ):
        group_df = group_df[group_df["time_frame"] > 1995]

        if group_df.empty:
            continue

        interp_xs = group_df["time_frame"].values

        # Prepare a dictionary for the new row (constant fields first)
        new_row = {
            "scenario": scenario,
            "percentile": percentile,
            "core_grouping": core_grouping,
            "time_frame": year,
        }

        for value_col in value_cols:
            inter_ys = group_df[value_col].values

            # Need at least 2 points to interpolate/extrapolate
            if len(interp_xs) < 2:
                continue

            spline = make_interp_spline(
                interp_xs, inter_ys, k=min(2, len(interp_xs) - 1)
            )
            value_at_year = float(spline(year))  # returns as array

            new_row[value_col] = value_at_year

        new_rows.append(new_row)

    # Add the new rows to the original dataframe
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return df
