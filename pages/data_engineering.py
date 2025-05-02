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
    st.subheader(f"Historical Insights ({dataset_name} Data)")

    hist_start_date = st.date_input(f"Start Date ({dataset_name} Data)", today - timedelta(days=30))
    hist_end_date = st.date_input(f"End Date ({dataset_name} Data)", today)
    historical_data = filter_data_by_date(data, hist_start_date, hist_end_date)

    if not historical_data.empty:
        chart_type = st.selectbox(f"Select Feature Type ({dataset_name} Data)", ["Categorical", "Continuous"])

        features = CATEGORICAL_FEATURES if chart_type == "Categorical" else [
            col for col in data.columns if col not in CATEGORICAL_FEATURES and col not in ['Timestamp', 'Loan_Status']
        ]

        selected_feature = st.selectbox(f"Select Feature ({dataset_name} Data)", features)
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
honors_file = st.sidebar.file_uploader("Upload Honors File", type=["csv", "xlsx"])
bureau_file = st.sidebar.file_uploader("Upload Bureau File", type=["csv", "xlsx"])

if honors_file is not None:
    honors_data = load_data(honors_file)
    if honors_data is not None:
        display_insights(honors_data, "Honors")

if bureau_file is not None:
    bureau_data = load_data(bureau_file)
    if bureau_data is not None:
        display_insights(bureau_data, "Bureau")
