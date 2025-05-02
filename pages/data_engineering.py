import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Loan Applications Dashboard", layout="wide")

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

# --- Functions ---
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

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

# --- UI Layout ---
st.title("Loan Applications Dashboard")

# File uploaders for Honors and Bureau data
honors_file = st.file_uploader("Upload Honors Data (CSV or Excel)", type=["csv", "xlsx"])
bureau_file = st.file_uploader("Upload Bureau Data (CSV or Excel)", type=["csv", "xlsx"])

if honors_file is not None and bureau_file is not None:
    try:
        # Load Honors data
        honors_data = load_data(honors_file)
        if honors_data is None:
            st.error("Error loading the Honors data.")
            st.stop()

        # Load Bureau data
        bureau_data = load_data(bureau_file)
        if bureau_data is None:
            st.error("Error loading the Bureau data.")
            st.stop()

        # Handle multiple Timestamp columns for both datasets
        for dataset, name in [(honors_data, "Honors"), (bureau_data, "Bureau")]:
            timestamp_columns = [col for col in dataset.columns if 'Timestamp' in col]
            if timestamp_columns:
                dataset['Timestamp'] = pd.to_datetime(dataset[timestamp_columns[0]], errors='coerce')  # Use first Timestamp
                dataset.dropna(subset=['Timestamp'], inplace=True)

        # --- Honors Data Summary ---
        st.header("Honors Data Summary")
        with st.expander("View Honors Data Feature Overview"):
            honors_no_timestamp = honors_data.drop(columns=[col for col in honors_data.columns if 'Timestamp' in col])

            def get_most_repeating_value(series):
                try:
                    val_counts = series.value_counts(dropna=True)
                    if not val_counts.empty:
                        most_common = val_counts.idxmax()
                        return f"{most_common} ({val_counts.max()})"
                except:
                    return ''
                return ''

            honors_summary = pd.DataFrame({
                "Feature": honors_no_timestamp.columns,
                "Data Type": honors_no_timestamp.dtypes.values,
                "% Missing": honors_no_timestamp.isnull().mean().values * 100,
                "# Distinct Values": honors_no_timestamp.nunique().values,
                "Most Repeating Value": [
                    get_most_repeating_value(honors_no_timestamp[col]) for col in honors_no_timestamp.columns
                ],
                "Min": [
                    honors_no_timestamp[col].min() if np.issubdtype(honors_no_timestamp[col].dtype, np.number) else '' 
                    for col in honors_no_timestamp.columns
                ],
                "Max": [
                    honors_no_timestamp[col].max() if np.issubdtype(honors_no_timestamp[col].dtype, np.number) else '' 
                    for col in honors_no_timestamp.columns
                ],
                "Avg": [
                    honors_no_timestamp[col].mean() if np.issubdtype(honors_no_timestamp[col].dtype, np.number) else '' 
                    for col in honors_no_timestamp.columns
                ],
            })

            st.dataframe(honors_summary, height=400)

        # --- Bureau Data Insights ---
        st.header("Bureau Data Insights")
        today = datetime.now().date()
        bureau_today_data = bureau_data[bureau_data['Timestamp'].dt.date == today - timedelta(days=1)]

        # Yesterday’s Insights for Bureau Data
        st.subheader("Yesterday’s Loan Insights & Historical Comparison (Bureau Data)")
        with st.container():
            col1, col2, col3 = st.columns(3)

            total_applications = len(bureau_today_data)
            col1.metric("Total Loan Applications Yesterday", total_applications)

            comparison_period = st.selectbox("Compare Approved/Rejected % Against (Bureau Data)", ['Yesterday', 'Last Week', 'Last Month', 'Last Year'])
            if comparison_period == 'Yesterday':
                compare_data = bureau_data[bureau_data['Timestamp'].dt.date == today - timedelta(days=2)]
            elif comparison_period == 'Last Week':
                compare_data = filter_data_by_date(bureau_data, today - timedelta(days=8), today - timedelta(days=2))
            elif comparison_period == 'Last Month':
                compare_data = filter_data_by_date(bureau_data, today - timedelta(days=31), today - timedelta(days=2))
            else:
                compare_data = filter_data_by_date(bureau_data, today - timedelta(days=366), today - timedelta(days=2))

            today_approved_pct, today_rejected_pct = today_rates(bureau_today_data)
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

        # Historical Insights for Bureau Data
        st.subheader("Historical Insights (Bureau Data)")

        hist_start_date = st.date_input("Start Date (Bureau Data)", today - timedelta(days=30))
        hist_end_date = st.date_input("End Date (Bureau Data)", today)
        historical_data = filter_data_by_date(bureau_data, hist_start_date, hist_end_date)

        if not historical_data.empty:
            chart_type = st.selectbox("Select Feature Type", ["Categorical", "Continuous"])

            features = CATEGORICAL_FEATURES if chart_type == "Categorical" else [
                col for col in bureau_data.columns if col not in CATEGORICAL_FEATURES and col not in timestamp_columns and col != 'Loan_Status'
            ]

            selected_feature = st.selectbox("Select Feature", features)
            col1, col2 = st.columns([3, 2])

            if chart_type == "Continuous":
                # Plot the line chart
                fig = px.line(historical_data, x='Timestamp', y=selected_feature, title=f"Trend of {selected_feature}")
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
                    title=f"Top 5 {selected_feature} Frequencies"
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

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    pass  # No function named 'show_page' is defined, so this is replaced with a placeholder.
