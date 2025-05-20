import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import os
import shutil  # Import shutil for directory cleanup

# Path to the default data directory
DEFAULT_DATA_DIR = "default_data"
DATA_REGISTRY_DIR = "data_registry"
# Function to load default data if session state is empty
def load_default_data():
    default_files = {
        "on_us_data": os.path.join(DEFAULT_DATA_DIR, "default_on_us_data.parquet"),
    }

    for key, file_path in default_files.items():
        if os.path.exists(file_path):
            try:
                st.session_state[key] = pd.read_parquet(file_path)
                
            except Exception as e:
                st.error(f"Error loading default data for '{key}' from {file_path}: {e}")
        else:
            st.warning(f"Default file for '{key}' not found at {file_path}. Please check the default_data directory.")

# Fixed list of categorical features
CATEGORICAL_FEATURES = [
    "RECENT_OPEN_ACCT_CUR_BAL_OPEN_TOTAL",
    "COLLECTIONS_CUR_BAL_TOTAL",
    "BANK_CARD_CREDIT_LIMIT_TOTAL",
    "REVOLVING_CUR_BAL_TOTAL",
    "DEROG_CUR_BAL_TOTAL",
    "DEROG_MO_PMT_TOTAL",
    "TOTAL_DOWN_CONTRACT_PERCENT",
    "PREPAYMENT_EVENT_LABEL"
]

# Clear the data_registry directory at the start of a new session
def clear_data_registry():
    if os.path.exists(DATA_REGISTRY_DIR):
        # Iterate through all files and subdirectories in the directory
        for root, dirs, files in os.walk(DATA_REGISTRY_DIR, topdown=False):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    os.unlink(file_path)  # Remove the file
                    st.info(f"Deleted file: {file_path}")
                except Exception as e:
                    st.error(f"Failed to delete file {file_path}: {e}")
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    os.rmdir(dir_path)  # Remove the directory (if empty)
                    st.info(f"Deleted directory: {dir_path}")
                except Exception as e:
                    st.error(f"Failed to delete directory {dir_path}: {e}")
        try:
            os.rmdir(DATA_REGISTRY_DIR)  # Remove the main directory (if empty)
            st.info(f"Deleted main directory: {DATA_REGISTRY_DIR}")
        except Exception as e:
            st.error(f"Failed to delete directory {DATA_REGISTRY_DIR}: {e}")
    else:
        os.makedirs(DATA_REGISTRY_DIR, exist_ok=True)  # Create the directory if it doesn't exist
        st.info(f"Created directory: {DATA_REGISTRY_DIR}")  # Create the directory if it doesn't exist

# Clear the `data_registry` directory only once per session
if "data_registry_cleared" not in st.session_state:
    clear_data_registry()
    st.session_state["data_registry_cleared"] = True  # Mark as cleared for this session

# --- Functions ---
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

    # Ensure the 'Timestamp' column is in datetime format
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
        if data['Timestamp'].isnull().all():
            st.error("The 'Timestamp' column could not be converted to datetime. Please check your data.")
            return None
    else:
        st.error("The dataset does not contain a 'Timestamp' column.")
        return None

    return data

def filter_data_by_date(df, start_date, end_date):
    mask = (df['Timestamp'] >= pd.to_datetime(start_date)) & (df['Timestamp'] <= pd.to_datetime(end_date))
    return df.loc[mask]

def calculate_change(current, previous):
    if previous == 0:
        return 0
    return round(((current - previous) / previous) * 100, 2)

def today_rates(df):
    total = len(df)
    if total == 0:
        return 0.0, 0.0
    approved = df[df['Loan_Status'] == 1].shape[0]
    approved_pct = (approved / total) * 100
    rejected_pct = 100 - approved_pct
    return approved_pct, rejected_pct

def average_approval_rate(df):
    if df.empty:
        return 0.0, 0.0

    grouped = df.groupby(df['Timestamp'].dt.date)
    summary = grouped.apply(lambda g: pd.Series({
        'total': len(g),
        'approved': g[g['Loan_Status'] == 1].shape[0]
    })).reset_index()

    summary = summary[summary['total'] > 0]
    if summary.empty:
        return 0.0, 0.0

    summary['approved_pct'] = summary['approved'] / summary['total'] * 100
    summary['rejected_pct'] = 100 - summary['approved_pct']

    return summary['approved_pct'].mean(), summary['rejected_pct'].mean()

def display_insights(data, dataset_name):
    """Display insights for a given dataset."""
    st.header(f"{dataset_name} Data Feature Summary")
    with st.expander(f"View {dataset_name} Data Feature Overview"):
        data_no_timestamp = data.drop(columns=[col for col in data.columns if 'Timestamp' in col])

        def get_most_repeating_value(series):
            try:
                val_counts = series.value_counts(dropna=True)
                if not val_counts.empty:
                    most_common = val_counts.idxmax()
                    return f"{most_common} ({val_counts.max()})"
            except:
                return ''
            return ''

        data_summary = pd.DataFrame({
            "Feature": data_no_timestamp.columns,
            "Data Type": data_no_timestamp.dtypes.values,
            "% Missing": data_no_timestamp.isnull().mean().values * 100,
            "# Distinct Values": data_no_timestamp.nunique().values,
            "Most Repeating Value": [
                get_most_repeating_value(data_no_timestamp[col]) for col in data_no_timestamp.columns
            ],
            "Min": [
                data_no_timestamp[col].min() if np.issubdtype(data_no_timestamp[col].dtype, np.number) else '' 
                for col in data_no_timestamp.columns
            ],
            "Max": [
                data_no_timestamp[col].max() if np.issubdtype(data_no_timestamp[col].dtype, np.number) else '' 
                for col in data_no_timestamp.columns
            ],
            "Avg": [
                data_no_timestamp[col].mean() if np.issubdtype(data_no_timestamp[col].dtype, np.number) else '' 
                for col in data_no_timestamp.columns
            ],
        })

        st.dataframe(data_summary, height=400)

    # Yesterday’s Insights
    st.subheader(f"Yesterday’s Loan Insights & Historical Comparison ({dataset_name} Data)")
    today = datetime.now().date()
    yesterday_data = data[data['Timestamp'].dt.date == today - timedelta(days=1)]

    with st.container():
        col1, col2, col3 = st.columns(3)

        total_applications = len(yesterday_data)
        col1.metric("Total Loan Applications Yesterday", total_applications)

        comparison_period = st.selectbox(f"Compare Approved/Rejected % Against ({dataset_name} Data)", ['Yesterday', 'Last Week', 'Last Month', 'Last Year'])
        if comparison_period == 'Yesterday':
            compare_data = data[data['Timestamp'].dt.date == today - timedelta(days=2)]
        elif comparison_period == 'Last Week':
            compare_data = filter_data_by_date(data, today - timedelta(days=8), today - timedelta(days=2))
        elif comparison_period == 'Last Month':
            compare_data = filter_data_by_date(data, today - timedelta(days=31), today - timedelta(days=2))
        else:
            compare_data = filter_data_by_date(data, today - timedelta(days=366), today - timedelta(days=2))

        today_approved_pct, today_rejected_pct = today_rates(yesterday_data)
        compare_approved_pct, compare_rejected_pct = average_approval_rate(compare_data)

        approved_change = calculate_change(today_approved_pct, compare_approved_pct)
        rejected_change = calculate_change(today_rejected_pct, compare_rejected_pct)

        col2.metric(
            label="Approved Loan %",
            value=f"{today_approved_pct:.2f}%",
            delta=f"{compare_approved_pct:.2f}% ({abs(approved_change):.2f}%)",
            delta_color="normal" if approved_change >= 0 else "inverse"
        )

        col3.metric(
            label="Rejected Loan %",
            value=f"{today_rejected_pct:.2f}%",
            delta=f"{compare_rejected_pct:.2f}% ({abs(rejected_change):.2f}%)",
            delta_color="normal" if rejected_change >= 0 else "inverse"
        )

    # Historical Insights
    display_historical_insights(data, dataset_name)

def display_bureau_insights(data, dataset_name):
    """Display insights for Bureau dataset."""
    st.header(f"{dataset_name} Data Feature Summary")
    with st.expander(f"View {dataset_name} Data Feature Overview"):
        data_no_timestamp = data.drop(columns=[col for col in data.columns if 'Timestamp' in col])

        # Data Summary Logic (unchanged)
        data_summary = pd.DataFrame({
            "Feature": data_no_timestamp.columns,
            "Data Type": data_no_timestamp.dtypes.values,
            "% Missing": data_no_timestamp.isnull().mean().values * 100,
            "# Distinct Values": data_no_timestamp.nunique().values,
        })
        st.dataframe(data_summary, height=400)

    # Filters for Start and End Date
    st.subheader(f"Loan Insights & Metrics ({dataset_name} Data)")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_date = st.date_input(f"Start Date ({dataset_name} Data)", datetime.now().date() - timedelta(days=30))
    with col2:
        end_date = st.date_input(f"End Date ({dataset_name} Data)", datetime.now().date())

    # Filter data based on selected dates
    filtered_data = filter_data_by_date(data, start_date, end_date)

    if not filtered_data.empty:
        # Calculate percentage of high-risk customers
        total_customers = len(filtered_data)
        high_risk_customers = filtered_data[
            (filtered_data['DEROG_TRDLN_TOTAL'] > 1) &  # More than 1 derogatory account
            (filtered_data['COLLECTIONS_CUR_BAL_TOTAL'] > 0) &  # Active collections
            (filtered_data['CREDIT_SCORE_AVG_CALC'] < 550) &  # Credit score under 550
            (filtered_data['DELINQ_CNT_30_DAY_TOTAL'] > 0)  # History of missed payments
        ]
        high_risk_pct = (len(high_risk_customers) / total_customers) * 100

        # Calculate total unpaid balance on derogatory accounts
        total_unpaid_balance = filtered_data['DEROG_CUR_BAL_TOTAL'].sum()

        # Display metrics as cards next to filters
        with col3:
            st.metric(
                label="High-Risk Customers (%)",
                value=f"{high_risk_pct:.2f}%"
            )
            st.metric(
                label="Total Unpaid Balance on Derogatory Accounts",
                value=f"£{total_unpaid_balance / 1_000_000:.1f}M"
            )

        # Historical Insights Section (unchanged)
        display_historical_insights(data, dataset_name)

def display_historical_insights(data, dataset_name):
    """Display historical insights for a given dataset."""
    st.subheader(f"Historical Insights ({dataset_name} Data)")

    # Layout: Filters on the left, feature selection on the right
    col1, col2 = st.columns([1, 2])

    # Filters for Start and End Date
    with col1:
        hist_start_date = st.date_input(
            f"Start Date ({dataset_name} Data)", 
            datetime.now().date() - timedelta(days=30), 
            key=f"hist_start_date_{dataset_name}"
        )
        hist_end_date = st.date_input(
            f"End Date ({dataset_name} Data)", 
            datetime.now().date(), 
            key=f"hist_end_date_{dataset_name}"
        )

    # Filter data based on selected dates
    historical_data = filter_data_by_date(data, hist_start_date, hist_end_date)

    if not historical_data.empty:
        # Feature selection stacked to the right
        with col2:
            chart_type = st.selectbox(
                f"Select Feature Type ({dataset_name} Data)", 
                ["Categorical", "Continuous"], 
                key=f"chart_type_{dataset_name}"
            )
            features = CATEGORICAL_FEATURES if chart_type == "Categorical" else [
                col for col in data.columns if col not in CATEGORICAL_FEATURES and col not in ['Timestamp', 'Loan_Status']
            ]
            selected_feature = st.selectbox(
                f"Select Feature ({dataset_name} Data)", 
                features, 
                key=f"selected_feature_{dataset_name}"
            )

        # Display insights based on feature type
        col1, col2 = st.columns([3, 2])

        if chart_type == "Continuous":
            # Plot the line chart
            fig = px.line(historical_data, x='Timestamp', y=selected_feature, title=f"Trend of {selected_feature} ({dataset_name} Data)")
            col1.plotly_chart(fig, use_container_width=True)

            # Tabulate statistics
            stats = historical_data[selected_feature].describe(percentiles=[0.25, 0.5, 0.75])
            null_pct = historical_data[selected_feature].isnull().mean() * 100

            stats_table = pd.DataFrame({
                "Statistic": ["Min", "Max", "Mean", "Null Values %", "25th Percentile", "50th Percentile", "75th Percentile"],
                "Value": [
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}",
                    f"{stats['mean']:.2f}",
                    f"{null_pct:.2f}%",
                    f"{stats['25%']:.2f}",
                    f"{stats['50%']:.2f}",
                    f"{stats['75%']:.2f}"
                ]
            })

            with col2:
                st.markdown("### Statistics")
                st.table(stats_table)

        else:
            # Plot the bar chart for top 5 categories
            top_counts = historical_data[selected_feature].value_counts(normalize=True).nlargest(5) * 100
            fig = px.bar(
                x=top_counts.index,
                y=top_counts.values,
                labels={'x': selected_feature, 'y': 'Percentage'},
                title=f"Top 5 {selected_feature} Frequencies ({dataset_name} Data)"
            )
            col1.plotly_chart(fig, use_container_width=True)

            # Tabulate top 5 counts
            top_counts_table = pd.DataFrame({
                "Category": top_counts.index,
                "Percentage": [f"{value:.2f}%" for value in top_counts.values]
            })

            with col2:
                st.markdown("### Top 5 Counts")
                st.table(top_counts_table)

# --- Main App ---
st.sidebar.header("Upload Files")
on_us_file = st.sidebar.file_uploader("Upload On-Us File", type=["csv", "xlsx"])
bureau_file = st.sidebar.file_uploader("Upload Bureau File", type=["csv", "xlsx"])

if on_us_file is not None:
    on_us_data = load_data(on_us_file)
    if on_us_data is not None:
        display_insights(on_us_data, "On-Us")
    if on_us_file is not None and on_us_data is not None:
        print("ENTERED DATA REGISTRY PARQUET")
        save_dir = "data_registry"
        os.makedirs(save_dir, exist_ok=True)
        data_path = os.path.join(save_dir, "on_us_data.parquet")
        on_us_data.to_parquet(data_path, index=False)

        # Store the path in session state
        st.session_state["on_us_data_path"] = data_path

if bureau_file is not None:
    bureau_data = load_data(bureau_file)
    if bureau_data is not None:
        display_bureau_insights(bureau_data, "Bureau")

# At the end of your data engineering logic, before moving to the next page:



    
# --- Main Logic ---


# Check if 'on_us_data' is in session state or load default data
