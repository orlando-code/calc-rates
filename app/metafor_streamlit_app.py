#!/usr/bin/env python3
"""
Metafor Streamlit App
Uses existing calcification/analysis code with robust context management.
"""

import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add the project root to the path so we can import calcification modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Suppress R warnings early
warnings.filterwarnings("ignore", message="R is not initialized by the main thread")
warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")


@contextmanager
def safe_r_context():
    """
    Context manager that safely handles rpy2 operations.
    Ensures proper conversion context for all R operations.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter

        # Ensure pandas2ri is activated
        pandas2ri.activate()

        # Use the combined converter context
        with localconverter(ro.default_converter + pandas2ri.converter):
            yield {"ro": ro, "pandas2ri": pandas2ri, "localconverter": localconverter}
    except Exception as e:
        print(f"[R_CONTEXT] Error in R context: {e}")
        raise


# Page configuration
st.set_page_config(
    page_title="Metafor Meta-Analysis Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


class RobustMetaforModel:
    """
    Robust wrapper that uses the hybrid approach for thread-safe R operations.
    """

    def __init__(self, **kwargs):
        """Initialize with hybrid adapter."""
        try:
            # Import the hybrid adapter
            from app.hybrid_metafor_adapter import StreamlitMetaforAdapter

            # Create the adapter instance
            self.model = StreamlitMetaforAdapter(**kwargs)
            self._context_initialized = True

            print("✅ RobustMetaforModel initialized with hybrid adapter")

        except Exception as e:
            print(f"❌ RobustMetaforModel initialization failed: {e}")
            self._context_initialized = False
            raise

    def _setup_r_context(self):
        """Setup R context in a thread-safe way."""
        try:
            # Force R initialization in current thread
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects.packages import importr

            # Activate pandas conversion globally for this thread
            pandas2ri.activate()

            # Import required R packages
            metafor = importr("metafor")
            base = importr("base")

            # Store for later use
            self.ro = ro
            self.pandas2ri = pandas2ri
            self.localconverter = localconverter
            self.metafor = metafor
            self.base = base

            print("[R_SETUP] R context initialized successfully")
            return True

        except Exception as e:
            print(f"[R_SETUP] Failed to initialize R context: {e}")
            return False

    def fit_model_robust(self):
        """Fit the model using the hybrid adapter."""
        try:
            # The hybrid adapter handles all context management internally
            fitted_model = self.model.fit_model()
            print("✅ Model fitted successfully using hybrid adapter")
            return fitted_model

        except Exception as e:
            print(f"❌ Model fitting failed: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Model fitting failed: {e}")

    def get_model_summary(self):
        """Get model summary using hybrid adapter."""
        return self.model.get_model_summary_text()

    def get_coefficients(self):
        """Extract model coefficients using hybrid adapter."""
        return self.model.get_coefficients_dataframe()


@st.cache_data
def load_data():
    """Load and cache the calcification data."""
    data_paths = [
        "data/clean/analysis_ready_data.csv",
    ]

    for data_path in data_paths:
        if Path(data_path).exists():
            try:
                df = pd.read_csv(data_path)
                st.info(f"✅ Loaded data from: {data_path}")

                # Ensure required columns exist
                if "original_doi" not in df.columns and "doi" in df.columns:
                    df["original_doi"] = df["doi"]

                if "treatment" not in df.columns:
                    st.warning("Treatment column not found - adding default")
                    df["treatment"] = "temp"

                return df

            except Exception as e:
                st.error(f"Error loading {data_path}: {e}")
                continue

    st.error(f"No data files found. Tried: {', '.join(data_paths)}")
    return None


def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """Get unique values from a column."""
    if column in df.columns:
        unique_vals = df[column].dropna().unique().tolist()
        return sorted([str(v) for v in unique_vals])
    return []


def create_effect_plot(
    df: pd.DataFrame, effect_col: str, moderator_col: str = None, colorby: str = None
) -> go.Figure:
    """Create a plot of effect sizes."""
    fig = go.Figure()

    if moderator_col and moderator_col in df.columns:
        # Scatter plot with moderator
        fig.add_trace(
            go.Scatter(
                x=df[moderator_col],
                y=df[effect_col],
                mode="markers",
                marker=dict(
                    size=8,
                    color=df[effect_col],
                    colorscale="RdYlBu",
                    showscale=True,
                    colorbar=dict(title=effect_col),
                ),
                text=df.get("original_doi", ""),
                hovertemplate=f"<b>%{{text}}</b><br>{moderator_col}: %{{x}}<br>{effect_col}: %{{y}}<extra></extra>",
                name="Studies",
            )
        )

        fig.update_layout(
            title=f"{effect_col} vs {moderator_col}",
            xaxis_title=moderator_col,
            yaxis_title=effect_col,
            height=500,
        )
    else:
        # Simple effect size distribution
        fig.add_trace(
            go.Histogram(x=df[effect_col], nbinsx=20, name="Effect Size Distribution")
        )

        fig.update_layout(
            title=f"Distribution of {effect_col}",
            xaxis_title=effect_col,
            yaxis_title="Frequency",
            height=400,
        )

    return fig


def main():
    st.title("🔬 Metafor Meta-Analysis Dashboard")
    st.markdown(
        "**Using existing calcification analysis code with robust R integration**"
    )

    # Load data
    df = load_data()
    if df is None:
        st.stop()

    # Data summary
    st.subheader("📊 Data Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Observations", len(df))
    with col2:
        if "original_doi" in df.columns:
            st.metric("Unique Studies", len(df["original_doi"].unique()))
        else:
            st.metric("Unique Studies", "N/A")
    with col3:
        if "treatment" in df.columns:
            st.metric("Treatments", len(df["treatment"].unique()))
        else:
            st.metric("Treatments", "N/A")
    with col4:
        if "st_calcification_unit" in df.columns:
            st.metric("Units", len(df["st_calcification_unit"].unique()))
        else:
            st.metric("Units", "N/A")

    # Sidebar controls
    st.sidebar.header("🔧 Analysis Settings")

    # Effect type selection
    available_effect_types = []
    if (
        "st_relative_calcification" in df.columns
        and "st_relative_calcification_var" in df.columns
    ):
        available_effect_types.append("st_relative_calcification")
    if "hedges_g" in df.columns and "hedges_g_var" in df.columns:
        available_effect_types.append("hedges_g")

    if not available_effect_types:
        st.error("❌ No valid effect types found")
        st.stop()

    effect_type = st.sidebar.selectbox(
        "Effect Type", options=available_effect_types, index=0
    )

    var_col = f"{effect_type}_var"
    st.sidebar.success(f"✅ Using variance column: {var_col}")

    # Treatment filtering
    default_treatment = "temp"
    available_treatments = get_unique_values(df, "treatment")
    selected_treatments = st.sidebar.multiselect(
        "Select Treatments",
        options=available_treatments,
        default=default_treatment
        if default_treatment in available_treatments
        else available_treatments[-1:],
    )

    # Calcification unit filtering
    default_unit = ["mgCaCO3 g-1d-1"]
    if "st_calcification_unit" in df.columns:
        available_units = get_unique_values(df, "st_calcification_unit")
        selected_units = st.sidebar.multiselect(
            "Calcification Units",
            options=available_units,
            default=default_unit
            if default_unit in available_units
            else available_units[-1:],
        )
    else:
        selected_units = []

    # Note: Moderator selection is now handled in the flexible formula construction
    # and in the plotting interface, not in the sidebar
    moderator = None

    # Filter data
    filtered_df = df.copy()

    if selected_treatments:
        filtered_df = filtered_df[filtered_df["treatment"].isin(selected_treatments)]

    if selected_units and "st_calcification_unit" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["st_calcification_unit"].isin(selected_units)
        ]

    # Remove missing values for essential columns
    required_cols = [effect_type, var_col]

    before_filter = len(filtered_df)
    filtered_df = filtered_df.dropna(subset=required_cols)
    after_filter = len(filtered_df)

    if before_filter != after_filter:
        st.sidebar.info(
            f"ℹ️ Removed {before_filter - after_filter} rows with missing values"
        )

    st.sidebar.metric("Filtered Dataset", f"{len(filtered_df)} observations")

    if len(filtered_df) < 3:
        st.warning("⚠️ Need at least 3 observations for meta-analysis")
        st.stop()

    # Main tabs
    tab1, tab2 = st.tabs(["🔬 Meta-Analysis", "📈 Data Exploration"])

    with tab1:
        st.subheader("Meta-Analysis")

        # Model settings
        st.subheader("🔧 Model Configuration")

        col1, col2 = st.columns(2)

        with col1:
            use_random_effects = st.checkbox(
                "Use Random Effects",
                value=True,
                help="Include random effects for study-level variation",
            )

        with col2:
            if use_random_effects:
                random_structure = st.selectbox(
                    "Random Effects Structure",
                    ["~ 1 | original_doi", "~ 1 | original_doi/ID"],
                    index=1,
                )
            else:
                random_structure = None

        # # Treatment as moderator
        # include_treatment = len(selected_treatments) > 1 and st.checkbox(
        #     "Include Treatment as Moderator"
        # )

        # --- Flexible formula construction with term type selection ---

        st.markdown("#### 🔢 Model Moderators & Formula Construction")

        # List of candidate columns for moderators (excluding effect, var, and ID columns)
        exclude_cols = {
            effect_type,
            var_col,
            "ID",
            "doi",
            "original_doi",
            "st_calcification_unit",
            "treatment",
            "hedges_g",
            "hedges_g_var",
            "st_relative_calcification",
            "st_relative_calcification_var",
            "st_control_calcification",
            "st_treatment_calcification",
        }
        candidate_cols = [
            col
            for col in filtered_df.columns
            if col not in exclude_cols and filtered_df[col].dtype.kind in "fiO"
        ]

        # Intercept tickbox
        include_intercept = st.checkbox(
            "Include Intercept in Model",
            value=True,
            help="If unchecked, the model will be fit without an intercept (i.e., '-1' in the formula).",
        )

        # New flexible moderator selection system
        st.markdown("#### 📊 Add Model Terms")
        st.markdown(
            "Select moderator variables and their term types. You can add the same variable multiple times with different term types (e.g., linear + quadratic)."
        )

        # Helpful examples
        with st.expander("💡 Examples"):
            st.markdown("""
            **Common model formulas you can build:**
            
            **Linear relationship:** `effect ~ delta_t`
            - Add: delta_t (linear)
            
            **Quadratic relationship:** `effect ~ delta_t + I(delta_t^2)`
            - Add: delta_t (linear)
            - Add: delta_t (quadratic)
            
            **Multiple variables:** `effect ~ delta_t + delta_ph + factor(treatment)`
            - Add: delta_t (linear)
            - Add: delta_ph (linear) 
            - Add: treatment (factor)
            
            **Complex model:** `effect ~ delta_t + I(delta_t^2) + delta_ph + I(delta_ph^2) - 1`
            - Add: delta_t (linear)
            - Add: delta_t (quadratic)
            - Add: delta_ph (linear)
            - Add: delta_ph (quadratic)
            - Uncheck "Include Intercept"
            """)

        # Initialize session state for moderator terms if not exists
        if "moderator_terms" not in st.session_state:
            st.session_state.moderator_terms = []

        # Add new moderator term
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            new_moderator = st.selectbox(
                "Select moderator variable:",
                options=[""] + candidate_cols,
                key="new_moderator_select",
            )

        with col2:
            if new_moderator:
                col_dtype = filtered_df[new_moderator].dtype
                # Guess default: factor for object/categorical, linear for numeric
                if col_dtype == "O" or str(col_dtype).startswith("category"):
                    default_type = "factor"
                else:
                    default_type = "linear"

                new_term_type = st.selectbox(
                    "Term type:",
                    options=["linear", "quadratic", "factor"],
                    index=["linear", "quadratic", "factor"].index(default_type),
                    key="new_term_type_select",
                )
            else:
                new_term_type = "linear"

        with col3:
            if st.button("➕ Add Term", disabled=not new_moderator):
                # Create the term string
                if new_term_type == "linear":
                    term = new_moderator
                elif new_term_type == "quadratic":
                    term = f"I({new_moderator}^2)"
                elif new_term_type == "factor":
                    term = f"factor({new_moderator})"

                # Add to session state
                st.session_state.moderator_terms.append(
                    {"variable": new_moderator, "type": new_term_type, "term": term}
                )
                st.rerun()

        # Display current moderator terms
        if st.session_state.moderator_terms:
            st.markdown("**Current model terms:**")

            # Clear all button
            if st.button("🗑️ Clear All Terms", type="secondary"):
                st.session_state.moderator_terms = []
                st.rerun()

            for i, term_info in enumerate(st.session_state.moderator_terms):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(
                        f"• {term_info['variable']} ({term_info['type']}) → `{term_info['term']}`"
                    )
                with col2:
                    if st.button("🗑️", key=f"remove_{i}"):
                        st.session_state.moderator_terms.pop(i)
                        st.rerun()
                with col3:
                    st.write("")  # Spacer

        # Build moderator terms list, dropping duplicates
        moderator_terms = list(
            set([term_info["term"] for term_info in st.session_state.moderator_terms])
        )

        # Build formula string with intercept option
        if moderator_terms:
            formula = " + ".join(moderator_terms)
            if not include_intercept:
                formula = formula + " - 1"
        else:
            formula = "1" if include_intercept else "0"

        formula = f"{effect_type} ~ {formula}"

        st.info(f"📝 Model formula: `{formula}`")
        if random_structure:
            st.info(f"🎲 Random effects: `{random_structure}`")

        # Debug info
        with st.expander("🔧 Debug Info"):
            st.write(f"**Filtered data shape:** {filtered_df.shape}")
            st.write(f"**Effect type:** {effect_type}")
            st.write(f"**Formula:** {formula}")
            st.write(f"**Random structure:** {random_structure}")
            st.write(
                f"**Required columns:** {[effect_type, var_col] + ([moderator] if moderator else [])}"
            )
            st.write(
                f"**Dropped due to NaN values or Cook's distance:** {len(filtered_df) - len(df)}"
            )

        # Fit model button
        if st.button("🚀 Fit Meta-Analysis Model", type="primary"):
            with st.spinner("Fitting meta-analysis model..."):
                try:
                    # Prepare model arguments
                    model_kwargs = {
                        "df": filtered_df,
                        "effect_type": effect_type,
                        "treatment": selected_treatments[0]
                        if len(selected_treatments) == 1
                        else selected_treatments,
                        "formula": formula if formula != "1" else None,
                        "random": random_structure
                        if use_random_effects
                        else "~ 1 | original_doi",
                        "verbose": True,
                    }

                    # Create and fit robust model
                    st.info("🔧 Creating model instance...")
                    robust_model = RobustMetaforModel(**model_kwargs)

                    st.info("🔬 Fitting model...")
                    _ = robust_model.fit_model_robust()

                    st.success("✅ Model fitted successfully!")

                    # Store the fitted model in session state for dynamic plotting
                    st.session_state.fitted_model = robust_model.model
                    st.session_state.model_fitted = True

                    # Display results
                    st.subheader("📋 Model Results")

                    # Model summary
                    with st.expander("📊 Model Summary", expanded=True):
                        summary_text = robust_model.get_model_summary()
                        st.code(summary_text, language="r")

                    # Coefficients table
                    with st.expander("📈 Coefficients"):
                        coef_df = robust_model.get_coefficients()
                        if coef_df is not None:
                            st.dataframe(coef_df)
                        else:
                            st.info("Could not extract coefficients table")

                    # Model info
                    with st.expander("ℹ️ Model Information"):
                        st.write(f"**Effect Type:** {effect_type}")
                        st.write(
                            f"**Number of Studies:** {len(filtered_df['original_doi'].unique()) if 'original_doi' in filtered_df.columns else 'N/A'}"
                        )
                        st.write(f"**Number of Observations:** {len(filtered_df)}")
                        st.write(f"**Formula:** {formula}")
                        if random_structure:
                            st.write(f"**Random Effects:** {random_structure}")

                except Exception as e:
                    st.error(f"❌ Model fitting failed: {str(e)}")

                    with st.expander("🔧 Error Details"):
                        import traceback

                        st.code(traceback.format_exc())

        # Meta-regression plotting (available after model is fitted)
        if st.session_state.get("model_fitted", False):
            st.subheader("📈 Meta-Regression Plotting")
            try:
                from app.streamlit_plotter import create_plotting_interface

                create_plotting_interface(st.session_state.fitted_model)
            except Exception as plot_error:
                st.error(f"Error loading plotting interface: {plot_error}")
                st.info("Plotting functionality is not available for this model.")

        # Export options
        st.subheader("💾 Export Data")
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name=f"metafor_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

    with tab2:
        st.subheader("Data Exploration")

        # Effect size plot - show distribution by default
        st.subheader(f"Distribution of {effect_type}")
        fig = create_effect_plot(filtered_df, effect_type)
        st.plotly_chart(fig, use_container_width=True)

        # Data preview
        with st.expander("📋 Data Preview"):
            st.dataframe(filtered_df[[effect_type, var_col]].head(10))

        # Summary statistics
        with st.expander("📊 Summary Statistics"):
            st.dataframe(filtered_df[[effect_type, var_col]].describe())


if __name__ == "__main__":
    main()
