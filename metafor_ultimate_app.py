#!/usr/bin/env python3
"""
Ultimate Metafor Analysis App
Combines robust data cleaning + force R initialization for maximum reliability.
"""

# CRITICAL: Import R initialization BEFORE anything else
print("🔧 Importing forced R initialization...")
# Suppress warnings early
import warnings

import r_force_init

warnings.filterwarnings("ignore", message="R is not initialized by the main thread")
warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Metafor Analysis Dashboard (Ultimate)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    """Load and cache the calcification data with robust error handling."""
    data_paths = [
        "data/clean/analysis_ready_data.csv",
        # "processed_calcification_data.csv",  # Fallback commented out
    ]

    for data_path in data_paths:
        if Path(data_path).exists():
            try:
                df = pd.read_csv(data_path)
                st.info(f"Loaded data from: {data_path}")

                # Add original_doi column if missing
                if "original_doi" not in df.columns and "doi" in df.columns:
                    df["original_doi"] = df["doi"]

                # Add treatment column if missing
                if "treatment" not in df.columns:
                    st.warning("Treatment column not found - inferring from data")
                    df["treatment"] = "temp"

                return df

            except Exception as e:
                st.error(f"Error loading {data_path}: {e}")
                continue

    st.error(f"No data files found. Tried: {', '.join(data_paths)}")
    return None


def clean_dataframe_for_r(df: pd.DataFrame, effect_type: str) -> pd.DataFrame:
    """Clean DataFrame to ensure it can be converted to R safely."""
    print(f"[CLEAN] Starting data cleaning for effect_type: {effect_type}")

    cleaned_df = df.copy()

    # Required columns for meta-analysis
    var_col = f"{effect_type}_var"
    required_cols = [effect_type, var_col]

    # Check for required columns
    missing_cols = [col for col in required_cols if col not in cleaned_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"[CLEAN] Required columns present: {required_cols}")

    # 1. Clean numeric columns (effect size and variance)
    for col in required_cols:
        print(f"[CLEAN] Cleaning numeric column: {col}")
        # Convert to numeric, coercing errors to NaN
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

        # Remove infinite values
        cleaned_df = cleaned_df[~cleaned_df[col].isin([float("inf"), float("-inf")])]

        print(f"[CLEAN] Column {col}: {cleaned_df[col].notna().sum()} valid values")

    # 2. Handle string columns properly - don't be overly aggressive
    string_cols = cleaned_df.select_dtypes(include=["object"]).columns
    print(f"[CLEAN] Processing string columns: {list(string_cols)}")

    for col in string_cols:
        if col in cleaned_df.columns:
            # Only clean problematic strings, keep normal categorical data

            # Handle repeated units issue specifically (like mgCaCO3 g-1d-1mgCaCO3...)
            if "unit" in col.lower() or col in ["st_calcification_unit"]:
                # Clean up repeated unit strings
                before_cleaning = cleaned_df[col].astype(str).str.len().max()
                cleaned_df[col] = (
                    cleaned_df[col]
                    .astype(str)
                    .str.replace(r"(.+?)\1+", r"\1", regex=True)
                )
                after_cleaning = cleaned_df[col].astype(str).str.len().max()
                if before_cleaning != after_cleaning:
                    print(
                        f"[CLEAN] Fixed repeated patterns in {col}: {before_cleaning} -> {after_cleaning} chars"
                    )

            # Only truncate extremely long strings (>500 chars) that would break R
            very_long_mask = cleaned_df[col].astype(str).str.len() > 500
            if very_long_mask.any():
                print(
                    f"[CLEAN] Truncating {very_long_mask.sum()} extremely long strings in column {col}"
                )
                cleaned_df.loc[very_long_mask, col] = (
                    cleaned_df.loc[very_long_mask, col].astype(str).str[:500]
                )

    # 3. Remove rows with missing values in required columns only
    before_count = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(subset=required_cols)
    after_count = len(cleaned_df)

    if before_count != after_count:
        print(
            f"[CLEAN] Removed {before_count - after_count} rows with missing required values"
        )

    # 4. Ensure we have enough data
    if len(cleaned_df) < 3:
        raise ValueError(
            f"Insufficient data after cleaning: {len(cleaned_df)} rows (need at least 3)"
        )

    print(f"[CLEAN] Final cleaned data shape: {cleaned_df.shape}")
    return cleaned_df


def fit_metafor_model_ultimate(
    df: pd.DataFrame, effect_type: str, random_effects: str = None
):
    """Fit metafor model using pre-initialized R objects and robust data cleaning."""
    print(f"[METAFOR] Starting ultimate model fitting with effect_type={effect_type}")

    # Check R initialization
    if not r_force_init.is_r_initialized():
        raise RuntimeError("R not properly initialized")

    # Get R objects
    r_objects = r_force_init.get_r_objects()
    if not r_objects:
        raise RuntimeError("Failed to get R objects")

    metafor = r_objects["metafor"]
    ro = r_objects["ro"]
    pandas2ri = r_objects["pandas2ri"]
    localconverter = r_objects["localconverter"]

    # Clean the data first
    try:
        cleaned_df = clean_dataframe_for_r(df, effect_type)
        print(f"[METAFOR] Data cleaning successful: {len(cleaned_df)} rows")
    except Exception as e:
        raise RuntimeError(f"Data cleaning failed: {e}")

    # Convert to R with detailed error handling
    var_col = f"{effect_type}_var"

    try:
        print("[METAFOR] Converting cleaned DataFrame to R")
        # Start with essential numeric columns
        essential_cols = [effect_type, var_col]

        # Add random effects grouping variable if needed
        if random_effects and "original_doi" in cleaned_df.columns:
            essential_cols.append("original_doi")

        # Add important categorical variables for potential moderators
        # These are safe string columns that R can handle
        potential_moderators = ["st_calcification_unit", "treatment", "species", "taxa"]
        for mod_col in potential_moderators:
            if mod_col in cleaned_df.columns and mod_col not in essential_cols:
                essential_cols.append(mod_col)

        # Add any numeric moderator variables that might be useful
        numeric_moderators = [
            "delta_ph",
            "delta_t",
            "temp",
            "phtot",
            "delta_omega_arag",
            "delta_omega_calc",
        ]
        for num_col in numeric_moderators:
            if num_col in cleaned_df.columns and num_col not in essential_cols:
                essential_cols.append(num_col)

        # Create DataFrame for R with selected columns
        r_ready_df = cleaned_df[essential_cols].copy()

        # Ensure the main effect columns are numeric
        r_ready_df[effect_type] = r_ready_df[effect_type].astype(float)
        r_ready_df[var_col] = r_ready_df[var_col].astype(float)

        # Convert numeric moderators to proper types
        for num_col in numeric_moderators:
            if num_col in r_ready_df.columns:
                r_ready_df[num_col] = pd.to_numeric(
                    r_ready_df[num_col], errors="coerce"
                )

        print(f"[METAFOR] Prepared DataFrame shape: {r_ready_df.shape}")
        print(f"[METAFOR] Columns: {list(r_ready_df.columns)}")
        print(f"[METAFOR] Data types: {r_ready_df.dtypes.to_dict()}")

        # Convert to R (using pre-initialized context)
        pandas2ri.activate()  # Re-activate to be safe
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(r_ready_df)

        print("[METAFOR] DataFrame conversion to R successful")

    except Exception as conv_error:
        print(f"[METAFOR] DataFrame conversion failed: {conv_error}")
        raise RuntimeError(f"R conversion failed: {conv_error}")

    # Fit the model - CRITICAL: Wrap in conversion context
    try:
        print("[METAFOR] Fitting metafor model with proper context")

        # Ensure conversion context is active for ALL R operations
        with localconverter(ro.default_converter + pandas2ri.converter):
            if random_effects and "original_doi" in cleaned_df.columns:
                print(f"[METAFOR] Using random effects: {random_effects}")
                # Create R formula within the conversion context
                random_formula = ro.Formula(random_effects)
                model = metafor.rma_mv(
                    yi=effect_type,
                    V=var_col,
                    data=r_df,
                    random=random_formula,
                    method="REML",
                )
            else:
                print("[METAFOR] Using simple model")
                model = metafor.rma(
                    yi=effect_type, vi=var_col, data=r_df, method="REML"
                )

        print("[METAFOR] Model fitting successful")
        return model, cleaned_df

    except Exception as model_error:
        print(f"[METAFOR] Model fitting failed: {model_error}")
        raise RuntimeError(f"Model fitting failed: {model_error}")


def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """Get unique values from a column."""
    if column in df.columns:
        unique_vals = df[column].dropna().unique().tolist()
        return sorted([str(v) for v in unique_vals])
    return []


def create_simple_plot(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Create a simple scatter plot."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            marker=dict(size=8, color="blue", opacity=0.7),
            name="Studies",
        )
    )

    fig.update_layout(
        title=f"{y_col} vs {x_col}", xaxis_title=x_col, yaxis_title=y_col, height=500
    )

    return fig


def main():
    st.title("🚀 Metafor Analysis Dashboard (Ultimate)")
    st.markdown(
        "**Robust data cleaning + Force R initialization = Maximum reliability**"
    )

    # Check R initialization status
    if r_force_init.is_r_initialized():
        st.success("✅ R environment initialized successfully (Force Init)")
        r_available = True
    else:
        st.error("❌ Failed to initialize R environment")
        st.info("Basic data exploration is still available")
        r_available = False

    # Load data
    df = load_data()
    if df is None:
        st.stop()

    # Show data info
    st.subheader("📊 Data Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        if "original_doi" in df.columns:
            st.metric("Unique Studies", len(df["original_doi"].unique()))
        else:
            st.metric("Unique Studies", "N/A")

    # Show data issues
    with st.expander("🔍 Data Quality Check"):
        # Check for string columns with potential issues
        string_cols = df.select_dtypes(include=["object"]).columns
        for col in string_cols:
            if col in df.columns:
                max_len = df[col].astype(str).str.len().max()
                if max_len > 100:
                    st.warning(
                        f"Column '{col}' has very long strings (max: {max_len} chars)"
                    )

                # Check for repeated patterns
                sample_val = (
                    df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ""
                )
                if len(str(sample_val)) > 50:
                    st.info(f"Sample from '{col}': {str(sample_val)[:100]}...")

    # Sidebar controls
    st.sidebar.header("📊 Analysis Controls")

    # Effect type selection - only the main effect types, not all columns
    available_effect_types = []
    if (
        "st_relative_calcification" in df.columns
        and "st_relative_calcification_var" in df.columns
    ):
        available_effect_types.append("st_relative_calcification")
    if "hedges_g" in df.columns and "hedges_g_var" in df.columns:
        available_effect_types.append("hedges_g")

    if not available_effect_types:
        st.error(
            "No valid effect types found. Need 'st_relative_calcification' or 'hedges_g' with their variance columns."
        )
        st.stop()

    effect_type = st.sidebar.selectbox(
        "Effect Type", options=available_effect_types, index=0
    )

    # Check variance column
    var_col = f"{effect_type}_var"
    has_variance = var_col in df.columns

    if not has_variance:
        st.sidebar.error(f"Missing variance column: {var_col}")
        st.stop()
    else:
        st.sidebar.success(f"Found variance column: {var_col}")

    # Data filtering
    st.sidebar.subheader("🔬 Data Filtering")

    # Treatment filtering
    available_treatments = get_unique_values(df, "treatment")
    selected_treatments = st.sidebar.multiselect(
        "Select Treatments", options=available_treatments, default=available_treatments
    )

    # Moderator selection
    potential_moderators = [
        "delta_ph",
        "delta_t",
        "temp",
        "phtot",
        "delta_omega_arag",
        "delta_omega_calc",
        "st_calcification_unit",
    ]
    available_moderators = [col for col in potential_moderators if col in df.columns]

    moderator = st.sidebar.selectbox(
        "Moderator Variable",
        options=available_moderators,
        index=0 if available_moderators else None,
    )

    st.sidebar.info(f"Available moderators: {len(available_moderators)}")

    # Filter data
    filtered_df = df.copy()
    if selected_treatments:
        filtered_df = filtered_df[filtered_df["treatment"].isin(selected_treatments)]

    # Remove missing values for essential columns
    if moderator and effect_type:
        required_cols = [moderator, effect_type, var_col]
        before_filter = len(filtered_df)
        filtered_df = filtered_df.dropna(subset=required_cols)
        after_filter = len(filtered_df)

        if before_filter != after_filter:
            st.sidebar.info(
                f"Removed {before_filter - after_filter} rows with missing values"
            )

    st.sidebar.metric("Filtered Dataset", f"{len(filtered_df)} rows")

    # Main analysis
    tab1, tab2 = st.tabs(["📊 Data Explorer", "📈 Meta-Analysis"])

    with tab1:
        st.subheader("Data Exploration")

        # Data preview
        st.subheader("Data Preview")
        st.dataframe(filtered_df.head(10))

        # Simple scatter plot
        if moderator and effect_type:
            st.subheader(f"{effect_type} vs {moderator}")

            # Check data quality before plotting
            valid_data = filtered_df[[moderator, effect_type]].dropna()
            if len(valid_data) > 0:
                fig = create_simple_plot(valid_data, moderator, effect_type)
                st.plotly_chart(fig, use_container_width=True)

                # Basic statistics
                corr = valid_data[moderator].corr(valid_data[effect_type])
                st.metric("Correlation", f"{corr:.3f}")
            else:
                st.warning("No valid data for plotting")

    with tab2:
        st.subheader("Meta-Analysis with Ultimate Reliability")

        if not r_available:
            st.warning("R environment not available - cannot run meta-analysis")
        elif len(filtered_df) < 3:
            st.warning("Need at least 3 studies for meta-analysis")
        else:
            # Model options
            use_random_effects = st.checkbox("Use Random Effects", value=True)

            if st.button("Run Meta-Analysis", type="primary"):
                with st.spinner("Running meta-analysis with ultimate reliability..."):
                    try:
                        # Determine random effects
                        random_effects = None
                        if use_random_effects and "original_doi" in filtered_df.columns:
                            random_effects = "~ 1 | original_doi"

                        # Fit model with ultimate approach
                        model, cleaned_data = fit_metafor_model_ultimate(
                            filtered_df, effect_type, random_effects=random_effects
                        )

                        st.success("✅ Meta-analysis completed successfully!")

                        # Show results
                        st.subheader("Model Results")
                        st.info(f"Model fitted with {len(cleaned_data)} studies")

                        # Show cleaned data summary
                        with st.expander("📋 Cleaned Data Summary"):
                            st.write(f"Original data: {len(filtered_df)} rows")
                            st.write(f"Cleaned data: {len(cleaned_data)} rows")
                            st.dataframe(
                                cleaned_data[[effect_type, var_col]].describe()
                            )

                        # Visualization
                        if moderator:
                            st.subheader("Data Visualization")
                            fig = create_simple_plot(
                                cleaned_data, moderator, effect_type
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Meta-analysis failed: {str(e)}")

                        with st.expander("🔧 Debugging Information"):
                            st.code(f"Error: {e}")
                            import traceback

                            st.code(traceback.format_exc())

        # Export options
        st.subheader("Export Data")
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name=f"metafor_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
