import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import itertools

st.set_page_config(page_title="Loan Profitability Dashboard", layout="wide")

# -----------------------
# Load and preprocess data with bucketing
# -----------------------
@st.cache_data
def load_and_bucket_data():
    """
    Loads data from 'final_loan_simulation.csv', preprocesses it, and creates buckets for selected features.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    try:
        df = pd.read_csv("final_loan_simulation.csv")
    except FileNotFoundError:
        st.error(
            "Error: \'final_loan_simulation.csv\' not found. Please ensure the file is in the correct directory.")
        return None

    # Basic data type conversions
    df['YEAR'] = df['YEAR'].astype(int)
    df['Origination_year'] = df['Origination_year'].astype(int)
    df['Months_elapsed'] = df['Months_elapsed'].astype(int)

    # Create buckets for numerical features
    def create_buckets(series, low_q, high_q, labels):
        """Creates buckets based on quantiles."""
        quantiles = series.quantile([0, low_q, high_q, 1]).unique().tolist()
        if len(quantiles) > 1:
            # Ensure that the number of labels is one less than the number of bins
            if len(quantiles) - 1 < len(labels):
                labels = labels[:len(quantiles) - 1]  # Truncate labels to match
            return pd.cut(series, bins=quantiles, labels=labels,
                          include_lowest=True, duplicates='drop').astype(str)
        return pd.Series(['NA'] * len(series), dtype=str)

    df['OPB_BUCKET'] = create_buckets(df['OPB'], 0.33, 0.67, ['Low', 'Medium', 'High'])
    df['PAYMENT_AMOUNT_BUCKET'] = create_buckets(df['PAYMENT_AMOUNT'], 0.33, 0.67,
                                                ['Low', 'Medium', 'High'])
    df['AUTO_PTI_TOTAL_BUCKET'] = create_buckets(df['AUTO_PTI_TOTAL'], 0.33, 0.67,
                                                ['Low', 'Medium', 'High'])

    # Create custom buckets for 'CREDIT_SCORE_AVG_CALC'
    df['CREDIT_SCORE_BUCKET'] = pd.cut(df['CREDIT_SCORE_AVG_CALC'], bins=[0, 600, 660, float('inf')],
                                       labels=['<600', '600-660', '660+'], include_lowest=True).astype(str)

    # Create buckets for categorical features by converting to string.
    df['TERM_OF_LOAN_BUCKET'] = df['TERM_OF_LOAN'].astype(str)
    df['Origination_year_BUCKET'] = df['Origination_year'].astype(str)
    return df


# Ensure data is loaded
data = load_and_bucket_data()

if data is None:
    st.stop()  # Stop if data loading fails

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("Navigation")
st.sidebar.markdown("Currently on: **Results Page**")

# -----------------------
# Section 1: Default Overview Chart
# -----------------------
st.title("📊 Loan Profitability Overview")

# Get available origination years
orig_years_available = sorted(data['Origination_year'].unique())

# Set default value only if it exists in the available options
default_orig_year = 2018 if 2018 in orig_years_available else None

selected_orig_years = st.multiselect(
    "Select Origination Year(s) to Show (Default: 2018)",
    orig_years_available,
    default=[default_orig_year] if default_orig_year else []
)

thresholds = {2018: float('inf'), 2019: 72, 2020: 60, 2021: 48, 2022: 36, 2023: 24, 2024: 12}

overview_fig = go.Figure()

for year in selected_orig_years:
    df_year = data[data['Origination_year'] == year].copy()  # Filter by selected year
    if df_year.empty:
        continue

    grouped = df_year.groupby(["Months_elapsed", "Origination_year"])[
        "Profitability_Cal"].sum().reset_index()

    for orig_year in grouped["Origination_year"].unique():
        threshold = thresholds.get(orig_year, 0)
        df_solid = grouped[(grouped["Origination_year"] == orig_year) & (
            grouped["Months_elapsed"] <= threshold)]
        df_dash = grouped[(grouped["Origination_year"] == orig_year) & (
            grouped["Months_elapsed"] > threshold)]

        overview_fig.add_trace(go.Scatter(
            x=df_solid["Months_elapsed"],
            y=df_solid["Profitability_Cal"],
            mode="lines+markers",
            name=f"{orig_year} (Actual)",
            line=dict(shape="linear", dash="solid"),
        ))
        if not df_dash.empty:
            overview_fig.add_trace(go.Scatter(
                x=df_dash["Months_elapsed"],
                y=df_dash["Profitability_Cal"],
                mode="lines+markers",
                name=f"{orig_year} (Predicted)",
                line=dict(shape="linear", dash="dash"),
            ))

overview_fig.update_layout(
    title="Loan Profitability Overview",
    xaxis_title="Months Elapsed",
    yaxis_title="Total Profitability",
    showlegend=True
)
st.plotly_chart(overview_fig, use_container_width=True)

# -----------------------
# Section 2: Feature-based Profitability
# -----------------------
st.subheader("📌 Analyze Profitability by Segments")

available_bucket_features = [col for col in data.columns if col.endswith('_BUCKET')]
feature_display_names = {col: col.replace("_BUCKET", "") for col in available_bucket_features}

available_bucket_features.extend(['Origination_year_BUCKET', 'TERM_OF_LOAN_BUCKET'])
feature_display_names['Origination_year_BUCKET'] = 'Origination_year'
feature_display_names['TERM_OF_LOAN_BUCKET'] = 'TERM_OF_LOAN'

with st.expander("ℹ️ Feature Descriptions"):
    st.markdown("""
    - **OPB**: Remaining unpaid loan balance.
    - **PAYMENT_AMOUNT**: Monthly payment made by the borrower.
    - **AUTO_PTI_TOTAL**: Payment-to-Income ratio for auto loans.
    - **CREDIT_SCORE_BUCKET**: Categorized credit score range.
    - **TERM_OF_LOAN**: Duration of the loan in months.
    - **Origination_year**: Year in which the loan originated.
    """)

selected_features = st.multiselect(
    "Select any number of features for segmentation:",
    options=available_bucket_features,
    format_func=lambda x: feature_display_names[x]
)

selected_buckets = {}
for feat in selected_features:
    unique_vals = sorted(data[feat].dropna().unique())
    if unique_vals:
        selected = st.multiselect(f"Select values for **{feature_display_names[feat]}**:",
                                  unique_vals, key=feat)
        if selected:
            selected_buckets[feat] = selected

# -----------------------
# Section 3: Simulator
# -----------------------
st.subheader("⚙️ Loan Parameter Simulator")

new_columns = ['Charge_Off_Bal', 'Interest_Amount', 'Recovery_Amount', 'Fees']

if 'simulation_rows' not in st.session_state:
    st.session_state.simulation_rows = [0]
    st.session_state.selected_cols = []
    st.session_state.modified_data = None  # Initialize modified_data in session state
    st.session_state.simulated_once = False  # Add a flag to track if simulation has run
    st.session_state.selected_adjustments = {}


def add_simulation_row():
    if len(st.session_state.simulation_rows) < 4:
        st.session_state.simulation_rows.append(st.session_state.simulation_rows[-1] + 1)
    else:
        st.warning("Maximum of 4 transformations allowed.")
    st.rerun()


def remove_simulation_row(index):
    if index < len(st.session_state.simulation_rows):
        removed_col = st.session_state.get(f'selected_col_{index}')
        if removed_col:
            st.session_state.selected_cols = [
                col for col in st.session_state.selected_cols if col not in removed_col
            ]
        st.session_state.simulation_rows.pop(index)
        # Remove associated session state keys
        st.session_state.pop(f'col_{index}', None)
        st.session_state.pop(f'adj_{index}', None)
        st.session_state.pop(f'selected_col_{index}', None)
        st.session_state.pop(f'adjustment_{index}', None)
    st.rerun()


modified_data = data.copy()
available_cols = [col for col in new_columns]  # Initialize available_cols

for index in st.session_state.simulation_rows:
    st.markdown(f"**Transformation #{index + 1}**")
    col1, col2, col3 = st.columns([2, 2, 1])  # Add a third column for the "Remove" button

    with col1:
        if available_cols:
            selected_col = st.selectbox("Select Feature", options=available_cols, key=f"col_{index}")
        else:
            selected_col = None

    with col2:
        adjustment = st.selectbox(
            "Select Adjustment Factor",
            options=[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            key=f"adj_{index}",
        )

    with col3:
        if st.button("❌ Remove", key=f"remove_{index}"):
            remove_simulation_row(index)

    if selected_col:
        st.session_state.selected_cols.append(selected_col)
        st.session_state[f'selected_col_{index}'] = [selected_col]  # Store selected column
        st.session_state[f'adjustment_{index}'] = adjustment
        st.session_state.selected_adjustments[selected_col] = adjustment
        if selected_col in available_cols:
            available_cols.remove(selected_col)

if st.button("➕ Add Transformation"):
    add_simulation_row()

proceed_button = st.button("✅ Apply All Transformations")
# -----------------------
# Section 4: Display Results
# -----------------------
st.subheader("📊 Profitability Trends")


def calculate_and_plot_profitability(df_to_plot, title_prefix=""):
    """
    Calculates and plots profitability based on the given DataFrame.

    Args:
        df_to_plot (pd.DataFrame): DataFrame containing the data to plot.
        title_prefix (str, optional): Prefix for the plot title. Defaults to "".
    """
    fig = go.Figure()
    has_data = False
    if selected_buckets and all(selected_buckets.values()):  # added this check
        bucket_keys = list(selected_buckets.keys())
        combinations = list(itertools.product(*[selected_buckets[k] for k in bucket_keys]))
        for combo in combinations:
            label = " | ".join([f"{feature_display_names[k]}: {v}" for k, v in zip(bucket_keys, combo)])
            combined_df = pd.DataFrame()

            filters = [(df_to_plot[bucket_keys[i]] == combo[i]) for i in range(len(combo))]
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter &= f
            filtered_df = df_to_plot[combined_filter].copy()

            if not filtered_df.empty:
                has_data = True
                grouped = filtered_df.groupby(["Months_elapsed", "Origination_year"])[
                    "Profitability_Cal"].sum().reset_index()

                for orig_year in grouped["Origination_year"].unique():
                    threshold = thresholds.get(orig_year, 0)
                    df_solid = grouped[(grouped["Origination_year"] == orig_year) & (
                        grouped["Months_elapsed"] <= threshold)]
                    df_dash = grouped[(grouped["Origination_year"] == orig_year) & (
                        grouped["Months_elapsed"] > threshold)]

                    fig.add_trace(go.Scatter(
                        x=df_solid["Months_elapsed"],
                        y=df_solid["Profitability_Cal"],
                        mode="lines+markers",
                        name=f"{label} | {orig_year} (Actual)",
                        line=dict(shape="linear", dash="solid"),
                    ))
                    if not df_dash.empty:
                        fig.add_trace(go.Scatter(
                            x=df_dash["Months_elapsed"],
                            y=df_dash["Profitability_Cal"],
                            mode="lines+markers",
                            name=f"{label} | {orig_year} (Predicted)",
                            line=dict(shape="linear", dash="dash"),
                        ))

        if has_data:
            fig.update_layout(
                title=f"{title_prefix}Profitability vs Months Elapsed for Selected Feature Combinations",
                xaxis_title="Months Elapsed",
                yaxis_title="Total Profitability",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.4,
                    xanchor="center",
                    x=0.5
                ),
                height=700,
                width=1200
            )
            st.plotly_chart(fig, use_container_width=False)
        else:
            st.error(
                "No data available for the selected feature combinations. Please adjust your selections.")
    else:
        st.warning(
            "Please select at least one feature and corresponding bucket to proceed.")  # added warning
        st.stop()  # added stop()


if selected_buckets and all(selected_buckets.values()):
    calculate_and_plot_profitability(data)  # function call

    if proceed_button:
        # Always start with the original data
        modified_data = data.copy()  # Create a fresh copy of the original data

        # Apply adjustments within the Profitability_Cal calculation
        interest_adjustment = st.session_state.selected_adjustments.get(
            'Interest_Amount', 1)  # Default to 1 if not found
        fees_adjustment = st.session_state.selected_adjustments.get('Fees', 1)
        recovery_adjustment = st.session_state.selected_adjustments.get('Recovery_Amount', 1)
        charge_off_adjustment = st.session_state.selected_adjustments.get('Charge_Off_Bal', 1)

        # Recalculate Profitability_Cal based on the current adjustments
        modified_data['Profitability_Cal'] = (
            modified_data['Interest_Amount'] * interest_adjustment +
            modified_data['Fees'] * fees_adjustment +
            modified_data['Recovery_Amount'] * recovery_adjustment -
            modified_data['Charge_Off_Bal'] * charge_off_adjustment
        )

        # Store the modified data in session state
        st.session_state.modified_data = modified_data
        st.session_state.simulated_once = True

        # Recalculate and plot profitability
        calculate_and_plot_profitability(st.session_state.modified_data,
                                         title_prefix="Simulated ")  # Use the modified data
        st.rerun()

    elif st.session_state.simulated_once:  # Check the flag
        calculate_and_plot_profitability(st.session_state.modified_data,
                                         title_prefix="Simulated ")

else:
    st.warning("Please select at least one feature and corresponding bucket to proceed.")
