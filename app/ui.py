# 🖥️ UI Layer (Streamlit Interface)
# Pages: DataExplorer, ModelBuilder, PlotViewer, ResultsExport
# Components: Reusable DataFilters, PlotControls, ModelControls
# Utils: Session state management, layout helpers

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import helpers, infrastructure, metafor, plot  # noqa
from calcification.utils import utils  # noqa

# from notebooks.meta_analysis import formula_components  # Unused import


# """
# UI Structure Guidance:

# For a modular, maintainable Streamlit app with multiple pages and reusable components,
# the best practice is to use a function-based approach for each page/section,
# and (optionally) classes for reusable UI components or stateful widgets.

# Recommended structure:

# - One function per page (e.g., data_explorer_page(), model_builder_page(), etc.)
# - Optionally, helper functions for reusable UI elements (e.g., data_filters(), plot_controls())
# - Use Streamlit's session state for cross-page state management
# - If you have complex, stateful UI widgets, you can encapsulate them in classes, but for most Streamlit apps, functions suffice
# """

# Page configuration
st.set_page_config(
    page_title="Metafor Meta-Analysis Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# @st.cache_data
def check_effect_type(df: pd.DataFrame, effect_type: str) -> bool:
    """Check if effect type and its variance column exist in dataframe"""
    return effect_type in df.columns and f"{effect_type}_var" in df.columns


def sidebar(df: pd.DataFrame):
    st.sidebar.header("🔧 Analysis Settings")

    # check for effect types
    available_effect_types = []

    for effect_type in [
        "st_relative_calcification",
        "hedges_g",
        "absolute_calcification",
    ]:
        if check_effect_type(df, effect_type):
            available_effect_types.append(effect_type)

    if not available_effect_types:
        st.error("❌ No valid effect types found")
        st.stop()

    # select effect type
    effect_type = st.sidebar.selectbox(
        "Effect Type", options=available_effect_types, index=0
    )

    var_col = f"{effect_type}_var"
    st.sidebar.success(f"✅ Using variance column: {var_col}")

    # select treatment(s)
    default_treatment = "temp"
    available_treatments_raw = utils.get_unique_values(df, "treatment")

    # Create display mapping for treatments
    treatment_display_map = {}
    treatment_reverse_map = {}

    for treatment in available_treatments_raw:
        display_name = helpers.VAR_NAME_MAP.get(treatment, treatment)
        treatment_display_map[treatment] = display_name
        treatment_reverse_map[display_name] = treatment

    available_treatments_display = list(treatment_display_map.values())
    default_treatment_display = treatment_display_map.get(
        default_treatment, default_treatment
    )

    selected_treatments_display = st.sidebar.multiselect(
        "Select Treatments",
        options=available_treatments_display,
        default=default_treatment_display
        if default_treatment_display in available_treatments_display
        else available_treatments_display[-1:]
        if available_treatments_display
        else [],
    )

    # Convert back to original treatment names
    selected_treatments = [
        treatment_reverse_map.get(display, display)
        for display in selected_treatments_display
    ]

    # select calcification unit
    default_unit = ["mgCaCO3 g-1d-1"]
    if "st_calcification_unit" in df.columns:
        available_units = utils.get_unique_values(df, "st_calcification_unit")
        selected_units = st.sidebar.multiselect(
            "Calcification Units",
            options=available_units,
            default=default_unit
            if default_unit in available_units
            else available_units[-1:],
        )
    else:
        selected_units = []

    # optional debug mode
    debug_limit = 50  # Default debug limit
    debug_mode = st.sidebar.checkbox(
        "Enable debug mode",
        value=False,
        help=f"Limit data to first {debug_limit} samples for rapid testing",
    )
    if debug_mode:
        debug_limit = st.sidebar.slider(
            "Debug sample size (if enabled)",
            min_value=3,
            max_value=len(df),
            value=debug_limit,
            step=1,
            help=f"Number of samples to use in debug mode (default: {debug_limit})",
        )

    return df, effect_type, selected_treatments, selected_units, debug_mode, debug_limit


def data_overview(df: pd.DataFrame, title: str = "📊 Data Overview"):
    """Display headline figures of complete dataset"""
    # Data summary
    st.subheader(title)
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


def data_filter(
    df: pd.DataFrame,
    effect_type: str,
    selected_treatments: list[str],
    selected_units: list[str],
    debug_mode: bool,
    debug_limit: int,
):
    """Filter data based on user selections with validation"""

    # Input validation
    if df.empty:
        st.warning("⚠️ Empty dataframe provided to filter")
        return df

    if not effect_type:
        st.error("❌ No effect type specified")
        return df

    filtered_df = df.copy()

    # Apply treatment filter with validation
    if selected_treatments and "treatment" in filtered_df.columns:
        valid_treatments = [
            t for t in selected_treatments if t in filtered_df["treatment"].values
        ]
        if valid_treatments:
            filtered_df = filtered_df[filtered_df["treatment"].isin(valid_treatments)]
        else:
            st.warning("⚠️ No valid treatments found in data")

    # Apply unit filter with validation
    if selected_units and "st_calcification_unit" in filtered_df.columns:
        valid_units = [
            u
            for u in selected_units
            if u in filtered_df["st_calcification_unit"].values
        ]
        if valid_units:
            filtered_df = filtered_df[
                filtered_df["st_calcification_unit"].isin(valid_units)
            ]
        else:
            st.warning("⚠️ No valid calcification units found in data")

    # Validate required columns exist
    required_cols = [effect_type, f"{effect_type}_var"]
    missing_cols = [col for col in required_cols if col not in filtered_df.columns]

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        return pd.DataFrame()  # Return empty dataframe

    # Remove missing values and validate result
    nan_cols = [col for col in required_cols if filtered_df[col].isna().any()]
    initial_count = len(filtered_df)
    filtered_df = filtered_df.dropna(subset=required_cols)
    final_count = len(filtered_df)

    if final_count == 0:
        st.warning("⚠️ No data remaining after filtering and removing missing values")
    elif final_count < initial_count:
        missing_cols_str = (
            ", ".join(f"'{col}'" for col in nan_cols) + " columns"
            if len(nan_cols) > 1
            else f"'{nan_cols[0]}' column"
        )
        st.info(
            f"ℹ️ Removed {initial_count - final_count} rows with missing data in {missing_cols_str}"
        )

    if debug_mode:
        filtered_df = filtered_df.head(debug_limit)
    return filtered_df


def model_formula_construction_doc(st):
    st.markdown("""          
            **Linear relationship:** `effect ~ delta_t`
            - delta_t (linear)
            
            **Quadratic relationship:** `effect ~ delta_t + I(delta_t^2)`
            - delta_t (linear) & delta_t (quadratic)
            
            **Multiple variables:** `effect ~ delta_t + delta_ph + factor(treatment)`
            - delta_t (linear) & delta_ph (linear) & treatment (factor)
            
            **Complex model:** `effect ~ delta_t + I(delta_t^2) + delta_ph + I(delta_ph^2) - 1`
            - Uncheck 'Include Intercept' above
            - delta_t (linear) & delta_t (quadratic) & delta_ph (linear) & delta_ph (quadratic)
            
            **Full factorial:** `effect ~ delta_t * delta_ph` (equivalent to: delta_t + delta_ph + delta_t:delta_ph)
            - delta_t (linear) & delta_ph (linear) & delta_t:delta_ph (interaction)
            - Or use: delta_t*delta_ph (full factorial)
            
            **Interaction model:** `effect ~ delta_t:delta_ph`
            - delta_t:delta_ph (interaction)
            
            **Three-way interaction:** `effect ~ delta_t:delta_ph:temp`
            - delta_t (linear) & delta_ph (linear) & temp (linear) & delta_t:delta_ph:temp 
            
            **Quadratic interactions:** `effect ~ I(delta_t^2):I(delta_ph^2)`
            - delta_t:delta_ph (interaction)
            - Set both variables to "second" order
            
            **Mixed-order interactions:** `effect ~ delta_t:I(delta_ph^2)`
            - delta_t:delta_ph (interaction)
            - Set delta_t to "first" order, delta_ph to "second" order
            """)


def initialise_model_formula(st, df: pd.DataFrame, effect_type: str):
    """Handles the initialisation of the model formula and settings"""
    st.markdown("## 🔬 Meta-Analysis", unsafe_allow_html=True)

    # --- model settings ---
    st.markdown("### 🔧 Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        use_random_effects = st.checkbox(
            "Use Random Effects",
            value=True,
            help="Include random effects for study-level variation",
        )

        if use_random_effects:
            random_structure = st.selectbox(
                "Random Effects Structure",
                ["~ 1 | original_doi", "~ 1 | original_doi/ID"],
                index=1,
            )
        else:
            random_structure = None
    with col2:
        include_intercept = st.checkbox(
            "Include Intercept in Model",
            value=True,
            help="If unchecked, the model will be fit without an intercept (i.e., '-1' in the formula).",
        )

    # --- model moderator selection ---
    st.markdown("#### 🔢 Model Moderators & Formula Construction")

    st.markdown(
        "Select moderator variables and their term types. You can add the same variable multiple times with different term types (e.g., linear + quadratic)."
    )
    with st.expander("💡 Examples"):
        model_formula_construction_doc(st)

    # List of candidate columns for moderators (excluding effect, var, and ID columns)
    exclude_cols = {
        effect_type,
        f"{effect_type}_var",
        "ID",
        "doi",
        "original_doi",
        "treatment",
        "hedges_g",
        "hedges_g_var",
        "calcification",
        "calcification_unit",
        "st_relative_calcification",
        "st_relative_calcification_var",
        "st_control_calcification",
        "st_treatment_calcification",
        "dvar_phtot",
        "dvar_temp",
    }

    necessary_columns = [
        col
        for col in df.columns
        if col not in exclude_cols
        and df[col].dtype.kind in "fiO"  # include numeric and object columns
    ]

    formula = process_moderators_to_formula(
        st, df, necessary_columns, effect_type, include_intercept
    )

    st.info(f"📝 Model formula: `{formula}`")
    if random_structure:
        st.info(f"🎲 Random effects: `{random_structure}`")

    return formula, random_structure, use_random_effects


def select_term_type(st):
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
    return new_term_type


def select_moderator_variable(st, necessary_columns, new_term_type, df: pd.DataFrame):
    # Initialize variables to avoid UnboundLocalError
    interaction_var1 = interaction_var2 = interaction_var3 = ""
    new_moderator = ""

    # Create mapping for display: {display_name: original_name}
    display_to_original = {}
    original_to_display = {}

    for col in necessary_columns:
        display_name = helpers.VAR_NAME_MAP.get(col, col)

        display_to_original[display_name] = col
        original_to_display[col] = display_name

    display_options = list(display_to_original.keys())

    if new_term_type in ["interaction", "full_factorial", "three_way"]:
        if new_term_type == "three_way":
            # For three-way interactions, allow selection of three variables
            st.markdown("**Select variables for three-way interaction (A:B:C):**")
            interaction_var1_display = st.selectbox(
                "First variable:",
                options=[""] + display_options,
                key="interaction_var1_select",
            )
            interaction_var1 = display_to_original.get(interaction_var1_display, "")

            interaction_var2_display = st.selectbox(
                "Second variable:",
                options=[""]
                + [col for col in display_options if col != interaction_var1_display],
                key="interaction_var2_select",
            )
            interaction_var2 = display_to_original.get(interaction_var2_display, "")

            interaction_var3_display = st.selectbox(
                "Third variable:",
                options=[""]
                + [
                    col
                    for col in display_options
                    if col not in [interaction_var1_display, interaction_var2_display]
                ],
                key="interaction_var3_select",
            )
            interaction_var3 = display_to_original.get(interaction_var3_display, "")
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
            interaction_var1_display = st.selectbox(
                "First variable:",
                options=[""] + display_options,
                key="interaction_var1_select",
            )
            interaction_var1 = display_to_original.get(interaction_var1_display, "")

            interaction_var2_display = st.selectbox(
                "Second variable:",
                options=[""]
                + [col for col in display_options if col != interaction_var1_display],
                key="interaction_var2_select",
            )
            interaction_var2 = display_to_original.get(interaction_var2_display, "")

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
        new_moderator_display = st.selectbox(
            "Select moderator variable:",
            options=[""] + display_options,
            key="new_moderator_select",
        )
        new_moderator = display_to_original.get(new_moderator_display, "")

        # Auto-detect appropriate default type for the variable
        if new_moderator and new_term_type == "linear":
            col_dtype = df[new_moderator].dtype
            if col_dtype == "O" or str(col_dtype).startswith("category"):
                st.info("💡 Consider using 'factor' type for categorical variables")

    return new_moderator, interaction_var1, interaction_var2, interaction_var3


def select_order(
    st,
    new_term_type,
    interaction_var1,
    interaction_var2,
    interaction_var3,
    new_moderator,
):
    # default orders
    var1_order = var2_order = var3_order = term_order = "first"

    # Term order specification for interactions
    if new_term_type in ["interaction", "full_factorial", "three_way"]:
        st.markdown("**Term orders:**")
        if new_term_type == "three_way":
            if interaction_var1:
                var1_order = st.selectbox(
                    f"{interaction_var1[:8]}... order:",
                    options=["first", "second"],
                    format_func=lambda x: f"{x} (x)" if x == "first" else f"{x} (x²)",
                    key="var1_order_select",
                )
            if interaction_var2:
                var2_order = st.selectbox(
                    f"{interaction_var2[:8]}... order:",
                    options=["first", "second"],
                    format_func=lambda x: f"{x} (x)" if x == "first" else f"{x} (x²)",
                    key="var2_order_select",
                )
            if interaction_var3:
                var3_order = st.selectbox(
                    f"{interaction_var3[:8]}... order:",
                    options=["first", "second"],
                    format_func=lambda x: f"{x} (x)" if x == "first" else f"{x} (x²)",
                    key="var3_order_select",
                )
        else:
            # Two-way interactions
            if interaction_var1:
                var1_order = st.selectbox(
                    f"{interaction_var1[:8]}... order:",
                    options=["first", "second"],
                    format_func=lambda x: f"{x} (x)" if x == "first" else f"{x} (x²)",
                    key="var1_order_select",
                )
            if interaction_var2:
                var2_order = st.selectbox(
                    f"{interaction_var2[:8]}... order:",
                    options=["first", "second"],
                    format_func=lambda x: f"{x} (x)" if x == "first" else f"{x} (x²)",
                    key="var2_order_select",
                )
    else:
        # Single term order
        if new_moderator and new_term_type != "factor":
            term_order = st.selectbox(
                "Term order:",
                options=["first", "second"],
                format_func=lambda x: f"{x} (x)" if x == "first" else f"{x} (x²)",
                key="term_order_select",
                help="Specify the polynomial order for this term",
            )

    return var1_order, var2_order, var3_order, term_order


def conditional_moderator_button(
    st,
    new_term_type,
    interaction_var1,
    interaction_var2,
    interaction_var3,
    new_moderator,
    var1_order="first",
    var2_order="first",
    var3_order="first",
    term_order="first",
):
    # enable button based on term type requirements
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

        def _format_term_with_order(variable: str, order: str) -> str | None:
            """Format a variable term with the specified order."""
            order_mapping = {
                "first": 1,
                "second": 2,
                "third": 3,
                "fourth": 4,
                "fifth": 5,
            }
            try:
                if order == "first":
                    return variable
                else:
                    return f"I({variable}^{order_mapping[order]})"
            except KeyError:
                print(f"❌ Invalid order: {order}")
                return None

        # Create the term string based on type and orders
        if new_term_type == "linear":
            formatted_term = _format_term_with_order(new_moderator, term_order)
            if formatted_term is None:
                st.error(
                    f"❌ Failed to format term for {new_moderator} with order {term_order}"
                )
                return
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
            var1_formatted = _format_term_with_order(interaction_var1, var1_order)
            var2_formatted = _format_term_with_order(interaction_var2, var2_order)
            if var1_formatted is None or var2_formatted is None:
                st.error("❌ Failed to format interaction terms")
                return
            interaction_term = f"{var1_formatted}:{var2_formatted}"
            terms.append(interaction_term)
            display_var = (
                f"{interaction_var1}({var1_order}):{interaction_var2}({var2_order})"
            )

        elif new_term_type == "full_factorial":
            # Full factorial with specified orders
            var1_formatted = _format_term_with_order(interaction_var1, var1_order)
            var2_formatted = _format_term_with_order(interaction_var2, var2_order)
            if var1_formatted is None or var2_formatted is None:
                st.error("❌ Failed to format full factorial terms")
                return

            # Add main effects with orders
            terms.extend([var1_formatted, var2_formatted])
            # Add interaction
            interaction_term = f"{var1_formatted}:{var2_formatted}"
            terms.append(interaction_term)
            display_var = (
                f"{interaction_var1}({var1_order})*{interaction_var2}({var2_order})"
            )

        elif new_term_type == "three_way":
            # Three-way interaction with specified orders
            var1_formatted = _format_term_with_order(interaction_var1, var1_order)
            var2_formatted = _format_term_with_order(interaction_var2, var2_order)
            var3_formatted = _format_term_with_order(interaction_var3, var3_order)
            if (
                var1_formatted is None
                or var2_formatted is None
                or var3_formatted is None
            ):
                st.error("❌ Failed to format three-way interaction terms")
                return

            # Add main effects
            terms.extend([var1_formatted, var2_formatted, var3_formatted])
            # Add three-way interaction
            three_way_term = f"{var1_formatted}:{var2_formatted}:{var3_formatted}"
            terms.append(three_way_term)
            display_var = f"{interaction_var1}({var1_order}):{interaction_var2}({var2_order}):{interaction_var3}({var3_order})"

        else:
            # Handle unexpected term types
            st.error(f"❌ Unknown term type: {new_term_type}")
            return  # Exit early to avoid the UnboundLocalError

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


def process_moderators_to_formula(
    st,
    df: pd.DataFrame,
    necessary_columns: list[str],
    effect_type: str,
    include_intercept: bool,
) -> str:
    # initialize session state for moderator terms if not exists
    if "moderator_terms" not in st.session_state:
        st.session_state.moderator_terms = []

    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

    # --- select term type ---
    with col1:
        new_term_type = select_term_type(st)

    with col2:
        new_moderator, interaction_var1, interaction_var2, interaction_var3 = (
            select_moderator_variable(st, necessary_columns, new_term_type, df)
        )

    with col3:
        var1_order, var2_order, var3_order, term_order = select_order(
            st,
            new_term_type,
            interaction_var1,
            interaction_var2,
            interaction_var3,
            new_moderator,
        )

    with col4:
        conditional_moderator_button(
            st,
            new_term_type,
            interaction_var1,
            interaction_var2,
            interaction_var3,
            new_moderator,
            var1_order,
            var2_order,
            var3_order,
            term_order,
        )

    # --- display current moderator terms ---
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

    return f"{effect_type} ~ {formula}"


def meta_analysis(
    st,
    df: pd.DataFrame,
    effect_type: str,
    selected_treatments: list[str],
    formula: str,
    random_structure: str,
    use_random_effects: bool,
):
    if st.button(
        "🚀 Fit Meta-Analysis Model",
        type="primary",
        disabled=df.empty,
        help="No data available to fit model",
    ):
        with st.spinner("Fitting meta-analysis model..."):
            try:
                # Prepare model arguments
                model_kwargs = {
                    "df": df,
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
                with st.spinner("🔧 Creating model instance..."):
                    model = metafor.MetaforModel(**model_kwargs)

                with st.spinner("🔬 Fitting model..."):
                    _ = model.fit_model()

                st.success("✅ Model fitted successfully!")

                # Store the fitted model in session state for dynamic plotting
                st.session_state.fitted_model = model
                st.session_state.model_fitted = True

                return model

            except Exception as e:
                st.error(f"❌ Model fitting failed: {str(e)}")

                with st.expander("🔧 Error Details"):
                    import traceback

                    st.code(traceback.format_exc())


def create_metaregression_interface(fitted_model: metafor.MetaforModel):
    """Create the Streamlit plotting interface."""
    xaxis_moderators = ["delta_t", "delta_ph", "temp", "phtot"]

    # Set default moderator to delta_t if available, otherwise first available
    default_moderator = "delta_t"
    default_index = xaxis_moderators.index(default_moderator)

    # Moderator selection
    col1, col2 = st.columns([2, 1])
    # reformat x axis moderator names

    xaxis_moderator_display_map, xaxis_moderator_reverse_map = map_display_name_to_dict(
        xaxis_moderators
    )

    with col1:
        selected_xaxis_moderator_value = st.selectbox(
            "Select moderator for X-axis:",
            list(xaxis_moderator_display_map.values()),
            index=default_index,
            help="Choose a numeric variable to plot against the effect size. Common options: delta_t (temperature), delta_ph (pH)",
        )
        selected_xaxis_moderator = xaxis_moderator_reverse_map[
            selected_xaxis_moderator_value
        ]
        # Get available color variables from both processed and original dataframes
        color_options = ["core_grouping"]  # Always available

        # Add variables from processed dataframe
        processed_columns = set(fitted_model.processed_df.columns)

        # Add variables from original dataframe if available
        original_columns = set()
        if hasattr(fitted_model, "original_df"):
            original_columns = set(fitted_model.original_df.columns)

        # Combine all available columns
        all_columns = processed_columns.union(original_columns)
        # exclude columns from colour options
        excluded_cols = {
            fitted_model.effect_type,
            fitted_model.effect_type_var,
            "ID",
            "cooks_d",
            "leverage",
            "residuals",
            "fitted",
            "calcification_unit",
            "doi",
            "treatment",
            "dvar_phtot",
            "dvar_temp",
            "st_control_calcification",
            "st_treatment_calcification",
        }

        # Add interesting variables in priority order
        priority_vars = [
            "delta_ph",
            "delta_t",
            "temp",
            "phtot",
            "DOI",
            "treatment",
            "family",
            "original_doi",
        ]
        for var in priority_vars:
            if var in all_columns and var not in excluded_cols:
                color_options.append(var)

        # Add remaining numeric and categorical variables
        for col in sorted(all_columns):
            if col not in color_options and col not in excluded_cols:
                # Check if it's a reasonable variable to color by
                if not col.startswith("_") and not col.endswith("_var"):
                    color_options.append(col)

        colour_option_display_map, colour_option_reverse_map = map_display_name_to_dict(
            color_options
        )
        colorby_display_value = st.selectbox(
            "Colour by:",
            list(colour_option_display_map.values()),
            index=0,
            help="Choose a variable to color the points by.",
        )
        colorby_value = colour_option_reverse_map[colorby_display_value]

    with col2:
        plot_width = st.slider("Plot width", 600, 1200, 800, step=50)
        plot_height = st.slider("Plot height", 400, 800, 600, step=50)

        # Calculate y-limits efficiently without creating full plotter instance
        def _calculate_auto_y_limits():
            # Extract effect sizes directly from adapter
            effect_data = fitted_model.processed_df[fitted_model.effect_type].dropna()
            min_yi, max_yi = effect_data.min(), effect_data.max()
            range_y = max_yi - min_yi
            return min_yi - range_y * 0.1, max_yi + range_y * 0.1

        auto_y_min, auto_y_max = _calculate_auto_y_limits()
        # if use_custom_limits:
        y_range = auto_y_max - auto_y_min
        y_buffer = y_range * 0.2  # 20% buffer for slider limits

        st.markdown("**Axis limits:**")

        # Y-axis limits
        col2a, col2b = st.columns(2)
        with col2a:
            y_min_slider = st.slider(
                "Y min",
                float(auto_y_min - y_buffer),
                float(auto_y_max + y_buffer),
                float(auto_y_min),
                step=float(y_range / 100),
                format="%.2f",
            )
        with col2b:
            y_max_slider = st.slider(
                "Y max",
                float(auto_y_min - y_buffer),
                float(auto_y_max + y_buffer),
                float(auto_y_max),
                step=float(y_range / 100),
                format="%.2f",
            )

        if y_min_slider >= y_max_slider:
            st.warning("⚠️ Y min should be less than Y max")

    if selected_xaxis_moderator:
        col_regression, col_residuals = st.columns(2)

        plotter = plot.MetaRegressionPlotter(
            fitted_model,
            selected_xaxis_moderator,
            colorby=colorby_value,
        )
        with col_regression:
            if plotter.determine_whether_regression_possible():
                show_regression = st.checkbox(
                    "Show regression line",
                    value=True,
                    help="Show the regression line on the plot.",
                    key="regression_checkbox",
                )
            else:
                show_regression = False
        with col_residuals:
            # Meta-regression plot controls
            show_partial_residuals = st.checkbox(
                "Show partial residuals",
                value=False,
                help="Display partial residuals instead of raw data points. Partial residuals show the relationship between the moderator and outcome while controlling for other variables in the model.",
                key="partial_residuals_checkbox",
            )

        # check whether regression line should be plotted

        fig = plotter.plot_plotly_meta_regression(
            custom_y_limits=(y_min_slider, y_max_slider),
            width=plot_width,
            height=plot_height,
            show_partial_residuals=show_partial_residuals,
            show_regression=show_regression,
        )

        st.plotly_chart(fig, use_container_width=True)
        # Display summary statistics
        with st.expander("📊 Plot Statistics"):
            stats = plotter.create_summary_stats()
            for key, value in stats.items():
                st.write(f"**{key}:** {value}")

        export_options(fig, plotter, fitted_model)


def export_options(fig, plotter, fitted_model):
    """Export options for plotly figures"""
    with st.expander("💾 Download Options"):
        col1, col2 = st.columns(2)
        with col1:
            html_str = None
            if not html_str:
                html_str = fig.to_html()
                formula = (
                    fitted_model.formula.replace("~", "_")
                    .replace("+", "_")
                    .replace(" ", "")
                )
                st.download_button(
                    label="💾 Download HTML",
                    data=html_str,
                    file_name=f"metaregression_{formula}.html",
                    mime="text/html",
                )
        with col2:
            if st.button("📈 Download Plot Data as CSV"):
                plot_data = pd.DataFrame(
                    {
                        plotter.moderator_name: plotter.xi,
                        fitted_model.effect_type: plotter.yi,
                        f"{fitted_model.effect_type}_var": plotter.vi,
                        "study_weight": plotter.seinv,
                        "doi": plotter.dois,
                    }
                )
                csv = plot_data.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"metaregression_data_{formula}.csv",
                    mime="text/csv",
                )


def map_display_name_to_dict(display_names: list[str]) -> dict:
    display_map = {}
    reverse_map = {}
    for display_name in display_names:
        display_map[display_name] = helpers.VAR_NAME_MAP.get(display_name, display_name)
        reverse_map[display_map[display_name]] = display_name
    return display_map, reverse_map


def create_contour_interface(fitted_model: metafor.MetaforModel):
    """Create the Streamlit contour interface."""
    st.subheader("🔍 Contour Plot")
    col_moderators, col_plot_controls = st.columns([2, 1])
    with col_moderators:
        st.markdown("**Select two moderators for contour plot:**")
        moderatorx = st.selectbox(
            "$x$-axis moderator:",
            options=fitted_model.coefficient_names,
            key="moderatorx_select",
        )
        moderatorx = fitted_model.coefficient_names[
            fitted_model.coefficient_names.index(moderatorx)
        ]

        moderatory = st.selectbox(
            "$y$-axis moderator:",
            options=fitted_model.coefficient_names,
            key="moderatory_select",
        )
        moderatory = fitted_model.coefficient_names[
            fitted_model.coefficient_names.index(moderatory)
        ]

    with col_plot_controls:
        modx_range = st.slider(
            "$x$-axis moderator range:",
            min_value=0.0,
            max_value=10.0,
            value=(0.0, 10.0),
            step=0.1,
        )
        mody_range = st.slider(
            "$y$-axis moderator range:",
            min_value=-1.0,
            max_value=0.0,
            value=(-1.0, 0.0),
            step=0.1,
        )
    fig = plot.plot_contour(
        fitted_model,
        moderatorx=moderatorx,
        moderatory=moderatory,
        modx_range=modx_range,
        mody_range=mody_range,
        # n_points=resolution,
    )
    st.plotly_chart(fig, use_container_width=True)


def create_influence_interface(fitted_model: metafor.MetaforModel):
    """Create the Streamlit influence plot interface."""
    st.subheader("🔍 Influence Plot")
    fig = plot.plot_interactive_influence(
        fitted_model.original_df,
        fitted_model.effect_type,
        fitted_model.n_params,
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    """
    Main function to run streamlit app.
    """
    st.title("🔬 Metafor Meta-Analysis Dashboard")
    st.markdown("**Facilitating custom exploration of data and meta-analysis models**")

    # load data
    data_loader = infrastructure.DataLoader(
        data_path="data/clean/analysis_ready_data.csv"
    )
    df = data_loader.load_data()

    if df is None:
        st.error("❌ No data found")
        st.stop()

    # --- sidebar ---
    df, effect_type, selected_treatments, selected_units, debug_mode, debug_limit = (
        sidebar(df)
    )

    # --- top level main body ---
    data_overview(df, title="📊 Data Overview")

    # filter data
    filtered_df = data_filter(
        df, effect_type, selected_treatments, selected_units, debug_mode, debug_limit
    )

    # display filtered data
    data_overview(filtered_df, title="📊 Filtered Data Overview")

    # --- main body ---
    meta_analysis_tab, data_tab, metadata_tab = st.tabs(
        ["🔬 Meta-Analysis", "🔍 Data Explorer", "🌏 Metadata Explorer"]
    )

    with meta_analysis_tab:
        # get formula
        formula, random_structure, use_random_effects = initialise_model_formula(
            st, filtered_df, effect_type
        )
        _ = meta_analysis(
            st,
            filtered_df,
            effect_type,
            selected_treatments,
            formula,
            random_structure,
            use_random_effects,
        )

        # Use session state model for plotting - this persists across UI interactions
        if "fitted_model" in st.session_state and st.session_state.fitted_model.fitted:
            # Display Model Results section - now persistent across interactions
            st.subheader("📋 Model Results")

            # Model summary
            with st.expander("📊 Model Summary", expanded=True):
                summary_text = st.session_state.fitted_model.get_model_summary_text()
                st.code(summary_text, language="r")

            # Coefficients table
            with st.expander("📈 Coefficients"):
                coef_df = st.session_state.fitted_model.get_coefficients_dataframe()
                if coef_df is not None:
                    st.dataframe(coef_df)
                else:
                    st.info("Could not extract coefficients table")

            # Model info
            with st.expander("ℹ️ Model Information"):
                model = st.session_state.fitted_model
                st.write(f"**Effect Type:** {model.effect_type}")
                st.write(
                    f"**Number of Studies:** {len(model.df['original_doi'].unique()) if 'original_doi' in model.df.columns else 'N/A'}"
                )
                st.write(f"**Number of Observations:** {len(model.df)}")
                st.write(f"**Formula:** {model.formula}")
                if model.random:
                    st.write(f"**Random Effects:** {model.random}")

            create_metaregression_interface(st.session_state.fitted_model)
            create_contour_interface(st.session_state.fitted_model)
            create_influence_interface(st.session_state.fitted_model)

    with data_tab:
        st.subheader("📊 Data Explorer")
        # data_explorer(df, effect_type, selected_treatments, selected_units)  # TODO

    with metadata_tab:
        st.subheader("🌏 Metadata Explorer")
        # data_explorer(df, effect_type, selected_treatments, selected_units)  # TODO


if __name__ == "__main__":
    main()
