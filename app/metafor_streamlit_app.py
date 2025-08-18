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

    st.subheader("Debug mode")
    debug_mode = st.checkbox(
        "Enable debug mode",
        value=False,
        help="Limit data to first 50 samples for rapid testing",
    )

    if debug_mode:
        df = df.head(50)

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
            f"ℹ️ Will remove {before_filter - after_filter} rows with missing values"
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
            
            **Interaction model:** `effect ~ delta_t + delta_ph + delta_t:delta_ph`
            - Add: delta_t (linear)
            - Add: delta_ph (linear)
            - Add: delta_t:delta_ph (interaction)
            
            **Full factorial:** `effect ~ delta_t * delta_ph` (equivalent to: delta_t + delta_ph + delta_t:delta_ph)
            - Add: delta_t (linear)
            - Add: delta_ph (linear) 
            - Add: delta_t:delta_ph (interaction)
            - Or use: delta_t*delta_ph (full factorial)
            
            **Three-way interaction:** `effect ~ delta_t:delta_ph:temp`
            - Add: delta_t (linear)
            - Add: delta_ph (linear)
            - Add: temp (linear)
            - Add: delta_t:delta_ph:temp (three-way interaction)
            
            **Quadratic interactions:** `effect ~ I(delta_t^2):I(delta_ph^2)`
            - Add: delta_t:delta_ph (interaction)
            - Set both variables to "second" order
            
            **Mixed-order interactions:** `effect ~ delta_t:I(delta_ph^2)`
            - Add: delta_t:delta_ph (interaction)
            - Set delta_t to "first" order, delta_ph to "second" order
            """)

        # Initialize session state for moderator terms if not exists
        if "moderator_terms" not in st.session_state:
            st.session_state.moderator_terms = []

        # Add new moderator term
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

        with col1:
            new_term_type = st.selectbox(
                "Term type:",
                options=[
                    "linear",
                    "quadratic",
                    "factor",
                    "interaction",
                    "full_factorial",
                    "three_way",
                ],
                format_func=lambda x: {
                    "linear": "Linear (x)",
                    "quadratic": "Quadratic (x²)",
                    "factor": "Factor (categorical)",
                    "interaction": "Interaction (x:y)",
                    "full_factorial": "Full Factorial (x*y)",
                    "three_way": "Three-way (x:y:z)",
                }[x],
                key="new_term_type_select",
                help="Choose the type of term to add to the model",
            )

            # Add helpful descriptions
            descriptions = {
                "linear": "Simple linear relationship: effect ~ x",
                "quadratic": "Quadratic relationship: effect ~ I(x²)",
                "factor": "Categorical variable: effect ~ factor(x)",
                "interaction": "Interaction only: effect ~ x:y",
                "full_factorial": "Full factorial: effect ~ x*y (= x + y + x:y)",
                "three_way": "Three-way interaction: effect ~ x:y:z",
            }
            st.caption(descriptions[new_term_type])

        with col2:
            if new_term_type in ["interaction", "full_factorial", "three_way"]:
                if new_term_type == "three_way":
                    # For three-way interactions, allow selection of three variables
                    st.markdown(
                        "**Select variables for three-way interaction (A:B:C):**"
                    )
                    interaction_var1 = st.selectbox(
                        "First variable:",
                        options=[""] + candidate_cols,
                        key="interaction_var1_select",
                    )
                    interaction_var2 = st.selectbox(
                        "Second variable:",
                        options=[""]
                        + [col for col in candidate_cols if col != interaction_var1],
                        key="interaction_var2_select",
                    )
                    interaction_var3 = st.selectbox(
                        "Third variable:",
                        options=[""]
                        + [
                            col
                            for col in candidate_cols
                            if col not in [interaction_var1, interaction_var2]
                        ],
                        key="interaction_var3_select",
                    )
                    new_moderator = (
                        f"{interaction_var1}:{interaction_var2}:{interaction_var3}"
                        if all([interaction_var1, interaction_var2, interaction_var3])
                        else ""
                    )
                else:
                    # For interactions and full factorial, allow selection of two variables
                    interaction_label = (
                        "full factorial (A*B)"
                        if new_term_type == "full_factorial"
                        else "interaction (A:B)"
                    )
                    st.markdown(f"**Select variables for {interaction_label}:**")
                    interaction_var1 = st.selectbox(
                        "First variable:",
                        options=[""] + candidate_cols,
                        key="interaction_var1_select",
                    )
                    interaction_var2 = st.selectbox(
                        "Second variable:",
                        options=[""]
                        + [col for col in candidate_cols if col != interaction_var1],
                        key="interaction_var2_select",
                    )

                    if new_term_type == "full_factorial":
                        new_moderator = (
                            f"{interaction_var1}*{interaction_var2}"
                            if interaction_var1 and interaction_var2
                            else ""
                        )
                        if interaction_var1 and interaction_var2:
                            st.info(
                                f"💡 This will add: {interaction_var1} + {interaction_var2} + {interaction_var1}:{interaction_var2}"
                            )
                    else:
                        new_moderator = (
                            f"{interaction_var1}:{interaction_var2}"
                            if interaction_var1 and interaction_var2
                            else ""
                        )
            else:
                # For non-interaction terms, single variable selection
                new_moderator = st.selectbox(
                    "Select moderator variable:",
                    options=[""] + candidate_cols,
                    key="new_moderator_select",
                )

                # Auto-detect appropriate default type for the variable
                if new_moderator and new_term_type == "linear":
                    col_dtype = filtered_df[new_moderator].dtype
                    if col_dtype == "O" or str(col_dtype).startswith("category"):
                        st.info(
                            "💡 Consider using 'factor' type for categorical variables"
                        )

        with col3:
            # Initialize default orders
            var1_order = var2_order = var3_order = term_order = "first"

            # Term order specification for interactions
            if new_term_type in ["interaction", "full_factorial", "three_way"]:
                st.markdown("**Term orders:**")
                if new_term_type == "three_way":
                    if interaction_var1:
                        var1_order = st.selectbox(
                            f"{interaction_var1[:8]}... order:",
                            options=["first", "second"],
                            format_func=lambda x: f"{x} (x)"
                            if x == "first"
                            else f"{x} (x²)",
                            key="var1_order_select",
                        )
                    if interaction_var2:
                        var2_order = st.selectbox(
                            f"{interaction_var2[:8]}... order:",
                            options=["first", "second"],
                            format_func=lambda x: f"{x} (x)"
                            if x == "first"
                            else f"{x} (x²)",
                            key="var2_order_select",
                        )
                    if interaction_var3:
                        var3_order = st.selectbox(
                            f"{interaction_var3[:8]}... order:",
                            options=["first", "second"],
                            format_func=lambda x: f"{x} (x)"
                            if x == "first"
                            else f"{x} (x²)",
                            key="var3_order_select",
                        )
                else:
                    # Two-way interactions
                    if interaction_var1:
                        var1_order = st.selectbox(
                            f"{interaction_var1[:8]}... order:",
                            options=["first", "second"],
                            format_func=lambda x: f"{x} (x)"
                            if x == "first"
                            else f"{x} (x²)",
                            key="var1_order_select",
                        )
                    if interaction_var2:
                        var2_order = st.selectbox(
                            f"{interaction_var2[:8]}... order:",
                            options=["first", "second"],
                            format_func=lambda x: f"{x} (x)"
                            if x == "first"
                            else f"{x} (x²)",
                            key="var2_order_select",
                        )
            else:
                # Single term order
                if new_moderator and new_term_type != "factor":
                    term_order = st.selectbox(
                        "Term order:",
                        options=["first", "second"],
                        format_func=lambda x: f"{x} (x)"
                        if x == "first"
                        else f"{x} (x²)",
                        key="term_order_select",
                        help="Specify the polynomial order for this term",
                    )

        with col4:
            # Enable button based on term type requirements
            if new_term_type == "three_way":
                button_disabled = not all(
                    [interaction_var1, interaction_var2, interaction_var3]
                )
            elif new_term_type in ["interaction", "full_factorial"]:
                button_disabled = not (interaction_var1 and interaction_var2)
            else:
                button_disabled = not new_moderator

            if st.button("➕ Add", disabled=button_disabled):
                terms = []

                def _format_term_with_order(variable, order):
                    """Format a variable term with the specified order."""
                    if order == "first":
                        return variable
                    elif order == "second":
                        return f"I({variable}^2)"
                    return variable

                # Create the term string based on type and orders
                if new_term_type == "linear":
                    formatted_term = _format_term_with_order(new_moderator, term_order)
                    terms.append(formatted_term)
                    display_var = f"{new_moderator} ({term_order} order)"

                elif new_term_type == "quadratic":
                    # Quadratic always uses second order by definition
                    terms.append(f"I({new_moderator}^2)")
                    display_var = f"{new_moderator} (quadratic)"

                elif new_term_type == "factor":
                    terms.append(f"factor({new_moderator})")
                    display_var = f"factor({new_moderator})"

                elif new_term_type == "interaction":
                    # Create interaction with specified orders
                    var1_formatted = _format_term_with_order(
                        interaction_var1, var1_order
                    )
                    var2_formatted = _format_term_with_order(
                        interaction_var2, var2_order
                    )
                    interaction_term = f"{var1_formatted}:{var2_formatted}"
                    terms.append(interaction_term)
                    display_var = f"{interaction_var1}({var1_order}):{interaction_var2}({var2_order})"

                elif new_term_type == "full_factorial":
                    # Full factorial with specified orders
                    var1_formatted = _format_term_with_order(
                        interaction_var1, var1_order
                    )
                    var2_formatted = _format_term_with_order(
                        interaction_var2, var2_order
                    )

                    # Add main effects with orders
                    terms.extend([var1_formatted, var2_formatted])
                    # Add interaction
                    interaction_term = f"{var1_formatted}:{var2_formatted}"
                    terms.append(interaction_term)
                    display_var = f"{interaction_var1}({var1_order})*{interaction_var2}({var2_order})"

                elif new_term_type == "three_way":
                    # Three-way interaction with specified orders
                    var1_formatted = _format_term_with_order(
                        interaction_var1, var1_order
                    )
                    var2_formatted = _format_term_with_order(
                        interaction_var2, var2_order
                    )
                    var3_formatted = _format_term_with_order(
                        interaction_var3, var3_order
                    )

                    # Add main effects
                    terms.extend([var1_formatted, var2_formatted, var3_formatted])
                    # Add three-way interaction
                    three_way_term = (
                        f"{var1_formatted}:{var2_formatted}:{var3_formatted}"
                    )
                    terms.append(three_way_term)
                    display_var = f"{interaction_var1}({var1_order}):{interaction_var2}({var2_order}):{interaction_var3}({var3_order})"

                # Check for duplicates
                existing_terms = []
                for existing_entry in st.session_state.moderator_terms:
                    existing_terms.extend(existing_entry["term"])

                duplicate_found = False
                for term in terms:
                    if term in existing_terms:
                        st.warning(f"⚠️ Term '{term}' already exists in the model!")
                        duplicate_found = True

                if not duplicate_found:
                    # Add to session state
                    st.session_state.moderator_terms.append(
                        {
                            "variable": display_var,
                            "type": new_term_type,
                            "term": terms,
                        }
                    )
                    st.rerun()

        # Display current moderator terms
        if st.session_state.moderator_terms:
            # Clear all button
            if st.button("🗑️ Clear All Terms", type="secondary"):
                st.session_state.moderator_terms = []
                st.rerun()

            st.markdown("**Current model terms:**")

            for i, term_info in enumerate(st.session_state.moderator_terms):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(
                        f"{term_info['variable']} ({term_info['type']}) → "
                        + " + ".join([f"`{term}`" for term in term_info["term"]])
                    )
                with col2:
                    if st.button("🗑️", key=f"remove_{i}"):
                        st.session_state.moderator_terms.pop(i)
                        st.rerun()
                with col3:
                    st.write("")  # Spacer

        # Build moderator terms list, dropping duplicates
        # Flatten all terms (which may be lists), then deduplicate
        all_terms = []
        for term_info in st.session_state.moderator_terms:
            # term_info["term"] is always a list of terms
            all_terms.extend(term_info["term"])
        moderator_terms = list(
            dict.fromkeys(all_terms)
        )  # preserves order, removes duplicates

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
            st.write(
                f"**Filtered data:** {len(filtered_df)} samples, {len(filtered_df.columns)} columns"
            )
            st.write(f"**Effect type:** {effect_type}")
            st.write(f"**Formula:** {formula}")
            st.write(f"**Random structure:** {random_structure}")
            st.write(
                f"**Required columns:** {[effect_type, var_col] + ([moderator] if moderator else [])}"
            )
            st.write(
                f"**Dropped due to irrelevant treatment or units / NaN values / Cook's distance:** {abs(len(filtered_df) - len(df))}"
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
