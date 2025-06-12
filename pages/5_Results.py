import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import itertools
import pickle  # Import pickle
from datetime import datetime, timedelta

st.set_page_config(page_title="Loan Profitability Dashboard", layout="wide")


# -----------------------
# Load and preprocess data with bucketing
# -----------------------
@st.cache_data
def load_and_bucket_data():
    """
    Loads data from 'combined_df.pkl', preprocesses it, and creates buckets for selected features.

    Returns:
        pd.DataFrame: The processed DataFrame. Returns None on error.
    """
    try:
        with open("combined_df.pkl", "rb") as f:
            df = pickle.load(f)  # Deserialize the DataFrame
    except FileNotFoundError:
        st.error(
            "Error: 'combined_df.pkl' not found. Please run Valid.py first to generate the data.")
        return None  # Explicitly return None on error
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None
   # Convert Timestamp_x to datetime format
    if 'Timestamp_x' in df.columns:
        df['Timestamp_x'] = pd.to_datetime(df['Timestamp_x'], errors='coerce')  # Convert to datetime
    
    # Basic data type conversions. Check if columns exist before converting.
    for col in ['Origination_Year', 'TERM_OF_LOAN', 'Month']:
        if col in df.columns:
            # Replace infinite values with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # Fill NaN values with a default value (e.g., 0)
            df[col] = df[col].fillna(0)
            # Convert to integer
            df[col] = df[col].astype(int)

    # Calculate Profitability_Cal
    df['Profitability_Cal'] = (
        df['Interest_Amount'] + df['Fees'] + df['Recovery_Amount'] - df['Charge_Off_Bal']
    )

    # Create buckets for numerical features
    def create_buckets(series, labels):
        """Creates buckets for the given Series."""
        unique_vals = sorted(series.unique())
        if len(unique_vals) > 1:
            return pd.cut(series, bins=unique_vals, labels=labels, include_lowest=True, duplicates='drop').astype(str)
        return pd.Series(['NA'] * len(series), dtype=str)

    # Origination_Year and TERM_OF_LOAN are now categorical.
    if 'Origination_Year' in df.columns:
        df['Origination_Year_BUCKET'] = df['Origination_Year'].astype(str)
        # Filter out 0 from Origination_Year_BUCKET
        df = df[df['Origination_Year'] != 0]

    if 'TERM_OF_LOAN' in df.columns:
        df['TERM_OF_LOAN_BUCKET'] = df['TERM_OF_LOAN'].astype(str)
        # Filter out 0 from TERM_OF_LOAN_BUCKET
        df = df[df['TERM_OF_LOAN'] != 0]

    # Create OPB_BUCKET based on percentiles
    if 'OPB' in df.columns:
        percentiles = np.percentile(df['OPB'], [33, 67])
        df['OPB_BUCKET'] = pd.cut(
            df['OPB'],
            bins=[-np.inf, percentiles[0], percentiles[1], np.inf],
            labels=['Low', 'Medium', 'High']
        )

    return df



data = load_and_bucket_data()


if data is None:
    st.stop()  # Stop if data loading fails

# -----------------------
# Sidebar
# -----------------------
# st.sidebar.title("Navigation")
# st.sidebar.markdown("Currently on: **Results Page**")
col1, col2, col3 = st.columns([1, 20, 5])

with col1:
    st.image("cropped-Sigmoid_logo_3x.png", width=100)

with col2:
    st.markdown(
        """
        <style>
        .dynamic-title {
            text-align: center;
            margin-left: auto;
            margin-right: auto;
            font-size: 32px; /* Reduced font size */
            font-weight: bold;
        }
        </style>
        <h1 class="dynamic-title"> Loan Profitability AI engine</h1>
        """,
        unsafe_allow_html=True,
    )

with col3:
    # Dropdown-style contact bubble
    st.markdown("""
    <style>
    .dropdown {
        position: relative;
        display: inline-block;
        margin-top: 10px;
        float: right;
    }

    .dropdown-button {
        background-color: #E9F5FE;
        color: #0A2540;
        padding: 10px 16px;
        font-size: 14px;
        border: none;
        cursor: pointer;
        border-radius: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .dropdown-content {
        display: none;
        position: absolute;
        right: 0;
        background-color: #F9FAFB;
        min-width: 180px;
        padding: 10px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        z-index: 1;
    }

    .dropdown:hover .dropdown-content {
        display: block;
    }

    .dropdown-content a {
        color: #0A2540;
        text-decoration: none;
        display: block;
        font-size: 14px;
        margin-top: 5px;
    }
    </style>

    <div class="dropdown">
      <button class="dropdown-button">📞 Contact Us</button>
      <div class="dropdown-content">
        <b>Ravi Bajagur</b><br>
        <a href="tel:8959896843">8959896843</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Adjust spacing dynamically based on sidebar visibility
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] + div {
        margin-left: 150px; /* Adjust this value to control spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
    <style>
    .intro-text {
        font-size: 21px;
        font-weight: 600;
        color: #000000;
        line-height: 1.7;
        background-color: #f5f7fa;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)
# -----------------------
# Section 1: Default Overview Chart
# -----------------------
st.title("📊 Loan Profitability Overview")

orig_years_available = sorted(data['Origination_Year'].unique())
selected_orig_years = st.multiselect(
    "Select Origination Year(s) to Show ", orig_years_available, default=[2019, 2021]
)

thresholds = {2018: float('inf'), 2019: 72, 2020: 60, 2021: 48, 2022: 36, 2023: 24, 2024: 12}

overview_fig = go.Figure()

for year in selected_orig_years:
    df_year = data[data['Origination_Year'] == year].copy()  # Filter by selected year
    if df_year.empty:
        continue

    grouped = df_year.groupby(["Month", "Origination_Year"])[
        "Profitability_Cal"].sum().reset_index()

    for orig_year in grouped["Origination_Year"].unique():
        threshold = thresholds.get(orig_year, 0)
        df_solid = grouped[(grouped["Origination_Year"] == orig_year) & (
            grouped["Month"] <= threshold)]
        df_dash = grouped[(grouped["Origination_Year"] == orig_year) & (
            grouped["Month"] > threshold)]

        overview_fig.add_trace(go.Scatter(
            x=df_solid["Month"],
            y=df_solid["Profitability_Cal"],
            mode="lines+markers",
            name=f"{orig_year} (Actual)",
            line=dict(shape="linear", dash="solid"),
        ))
        if not df_dash.empty:
            overview_fig.add_trace(go.Scatter(
                x=df_dash["Month"],
                y=df_dash["Profitability_Cal"],
                mode="lines+markers",
                name=f"{orig_year} (Predicted)",
                line=dict(shape="linear", dash="dash"),
            ))

overview_fig.update_layout(
    title="Loan Profitability Overview",
    xaxis_title="Month",
    yaxis_title="Total Profitability",
    showlegend=True
)
st.plotly_chart(overview_fig, use_container_width=True)
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)

# -----------------------
# Section: Vintage Analysis
# -----------------------
st.subheader("📊 Vintage Analysis & Historical Comparison")

today = datetime.now().date()
yesterday_date_obj = today - timedelta(days=1)

# Filter data for yesterday based on Timestamp_x
if 'Timestamp_x' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Timestamp_x']):
    yesterday_data = data[data['Timestamp_x'].dt.date == yesterday_date_obj]
else:
    yesterday_data = pd.DataFrame()  # Empty if no valid timestamp
    st.warning("Cannot calculate yesterday's insights due to missing or invalid Timestamp_x column.")

# Calculate total profitability for yesterday
total_profitability_yesterday = 0
if 'Profitability_Cal' in yesterday_data.columns and not yesterday_data.empty:
    total_profitability_yesterday = yesterday_data.groupby('LOAN_ID')['Profitability_Cal'].sum().mean()

# Select comparison period
comparison_period = st.selectbox(
    "Compare Profitability Against:",
    ['Last Week', 'Last Month', 'Last Year'],
    key="comparison_period_select"
)

compare_data_start = None
compare_data_end = today - timedelta(days=2)  # Common end for comparison periods

if comparison_period == 'Yesterday':
    compare_data_start = today - timedelta(days=2)
elif comparison_period == 'Last Week':
    compare_data_start = today - timedelta(days=8)
elif comparison_period == 'Last Month':
    compare_data_start = today - timedelta(days=31)
else:  # Last Year
    compare_data_start = today - timedelta(days=366)

compare_data = pd.DataFrame()  # Initialize as empty
if compare_data_start and compare_data_end >= compare_data_start:
    compare_data = data[(data['Timestamp_x'].dt.date >= compare_data_start) &
                        (data['Timestamp_x'].dt.date <= compare_data_end)]
else:
    st.warning("Invalid comparison period generated.")

# Calculate profitability for comparison period
profitability_comparison = 0
if 'Profitability_Cal' in compare_data.columns and not compare_data.empty:
    # Group by LOAN_ID using sum, then calculate the mean of these values
    profitability_comparison = compare_data.groupby('LOAN_ID')['Profitability_Cal'].sum().mean()

profitability_change = total_profitability_yesterday - profitability_comparison
def format_to_millions(value):
    return f"${value / 1_000_000:.2f}M"
def format_to_thousands(value):
    return f"${value / 1_00_000:.2f}k"
# Display metrics side by side
with st.container():
    col1, col2 = st.columns(2)

    # Set delta text and color manually
    if profitability_change >= 0:
        delta_text = f"▲ {format_to_thousands(abs(profitability_change))}"
        delta_color = "green"
    else:
        delta_text = f"▼ {format_to_thousands(abs(profitability_change))}"
        delta_color = "red"

    # Column 1: Yesterday
    col1.markdown(
        f"""
        <div style="font-size:18px; font-weight:600;">Average Profitability per Loan (Yesterday)</div>
        <div style="font-size:28px; font-weight:bold;">{format_to_thousands(total_profitability_yesterday)}</div>
        <div style="font-size:16px; color:{delta_color}; font-weight:600;">{delta_text}</div>
        """,
        unsafe_allow_html=True
    )

    # Column 2: Comparison Period
    col2.markdown(
        f"""
        <div style="font-size:18px; font-weight:600;">Average Profitability per Loan ({comparison_period})</div>
        <div style="font-size:28px; font-weight:bold;">{format_to_thousands(profitability_comparison)}</div>
        """,
        unsafe_allow_html=True
    )


# -----------------------
# Section 2: Feature-based Profitability
# -----------------------
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
st.subheader("📌 Analyze Profitability by Segments")

# Feature selection
available_bucket_features = [col for col in data.columns if col.endswith('_BUCKET')] + ['LOAN_ID']
feature_display_names = {col: col.replace("_BUCKET", "") for col in available_bucket_features}
feature_display_names['Origination_Year_BUCKET'] = 'Origination_Year'
feature_display_names['TERM_OF_LOAN_BUCKET'] = 'TERM_OF_LOAN'
feature_display_names['OPB_BUCKET'] = 'Original Principal'
feature_display_names['LOAN_ID'] = 'Loan ID'

with st.expander("ℹ️ Feature Descriptions"):
    st.markdown("""
    - **TERM_OF_LOAN**: Duration of the loan in months.
    - **Origination_Year**: Year in which the loan originated.
    - **OPB**: Original Principal Balance.
    - **LOAN_ID**: Unique identifier for each loan.
    """)

# Default selections
default_selected_features = ['Origination_Year_BUCKET', 'TERM_OF_LOAN_BUCKET']
selected_features = st.multiselect(
    "Select any number of features for segmentation:",
    options=available_bucket_features,
    default=default_selected_features,
    format_func=lambda x: feature_display_names[x]
)

selected_buckets = {}
for feat in selected_features:
    unique_vals = sorted(data[feat].dropna().unique())
    if feat == 'Origination_Year_BUCKET':
        default_vals = [str(val) for val in ['2022', '2023'] if str(val) in unique_vals]
    elif feat == 'TERM_OF_LOAN_BUCKET':
        default_vals = [str(60)] if '60' in unique_vals else []
    else:
        default_vals = []
    selected = st.multiselect(
        f"Select values for **{feature_display_names[feat]}**:",
        unique_vals,
        default=default_vals,
        key=feat
    )
    if selected:
        selected_buckets[feat] = selected

# Profitability Forecast
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
st.subheader("📊 Profitability Forecast")

def calculate_and_plot_profitability(df_to_plot, title_prefix=""):
    fig = go.Figure()
    has_data = False
    if selected_buckets and all(selected_buckets.values()):
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
                grouped = filtered_df.groupby(["Month", "Origination_Year"])[
                    "Profitability_Cal"].sum().reset_index()

                for orig_year in grouped["Origination_Year"].unique():
                    threshold = thresholds.get(orig_year, 0)
                    df_solid = grouped[(grouped["Origination_Year"] == orig_year) &
                                       (grouped["Month"] <= threshold)]
                    df_dash = grouped[(grouped["Origination_Year"] == orig_year) &
                                      (grouped["Month"] > threshold)]

                    fig.add_trace(go.Scatter(
                        x=df_solid["Month"],
                        y=df_solid["Profitability_Cal"],
                        mode="lines+markers",
                        name=f"{label} | {orig_year} (Actual)",
                        line=dict(shape="linear", dash="solid"),
                    ))
                    if not df_dash.empty:
                        fig.add_trace(go.Scatter(
                            x=df_dash["Month"],
                            y=df_dash["Profitability_Cal"],
                            mode="lines+markers",
                            name=f"{label} | {orig_year} (Predicted)",
                            line=dict(shape="linear", dash="dash"),
                        ))

        if has_data:
            fig.update_layout(
                title=f"{title_prefix}Profitability vs Months for Selected Feature Combinations",
                xaxis_title="Month",
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
            st.error("No data available for the selected feature combinations. Please adjust your selections.")
    else:
        st.warning("Please select at least one feature and corresponding bucket to proceed.")
        st.stop()

# Run visualization
if selected_buckets and all(selected_buckets.values()):
    calculate_and_plot_profitability(data)
else:
    st.warning("Please select at least one feature and corresponding bucket to proceed.")


if st.button("Proceed to Simulator Page"):
        st.switch_page("pages/6_Loan_Simulator.py")
