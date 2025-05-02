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

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        data = load_data(uploaded_file)
        if data is None:
            st.error("Error loading the data.")
            st.stop()

        # Handle multiple Timestamp columns
        timestamp_columns = [col for col in data.columns if 'Timestamp' in col]
        if timestamp_columns:
            data['Timestamp'] = pd.to_datetime(data[timestamp_columns[0]], errors='coerce')  # Use first Timestamp
            data = data.dropna(subset=['Timestamp'])

        # --- Feature Summary ---
        st.header("Feature Summary")
        with st.expander("View Feature Overview"):
            data_no_timestamp = data.drop(columns=timestamp_columns)

            def get_most_repeating_value(series):
                try:
                    val_counts = series.value_counts(dropna=True)
                    if not val_counts.empty:
                        most_common = val_counts.idxmax()
                        return f"{most_common} ({val_counts.max()})"
                except:
                    return ''
                return ''

            feature_summary = pd.DataFrame({
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

            st.dataframe(feature_summary, height=400)

        today = datetime.now().date()
        today_data = data[data['Timestamp'].dt.date == today - timedelta(days=1)]

        # --- Yesterday’s Insights ---
        st.header("Yesterday’s Loan Insights & Historical Comparison")
        with st.container():
            col1, col2, col3 = st.columns(3)

            total_applications = len(today_data)
            col1.metric("Total Loan Applications Yesterday", total_applications)

            comparison_period = st.selectbox("Compare Approved/Rejected % Against", ['Yesterday', 'Last Week', 'Last Month', 'Last Year'])
            if comparison_period == 'Yesterday':
                compare_data = data[data['Timestamp'].dt.date == today - timedelta(days=2)]
            elif comparison_period == 'Last Week':
                compare_data = filter_data_by_date(data, today - timedelta(days=8), today - timedelta(days=2))
            elif comparison_period == 'Last Month':
                compare_data = filter_data_by_date(data, today - timedelta(days=31), today - timedelta(days=2))
            else:
                compare_data = filter_data_by_date(data, today - timedelta(days=366), today - timedelta(days=2))

            today_approved_pct, today_rejected_pct = today_rates(today_data)
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

        # --- Historical Insights ---
        st.header("Historical Insights")

        hist_start_date = st.date_input("Start Date", today - timedelta(days=30))
        hist_end_date = st.date_input("End Date", today)
        historical_data = filter_data_by_date(data, hist_start_date, hist_end_date)

        if not historical_data.empty:
            chart_type = st.selectbox("Select Feature Type", ["Categorical", "Continuous"])

            features = CATEGORICAL_FEATURES if chart_type == "Categorical" else [
                col for col in data.columns if col not in CATEGORICAL_FEATURES and col not in timestamp_columns and col != 'Loan_Status'
            ]

            selected_feature = st.selectbox("Select Feature", features)
            col1, col2 = st.columns([3, 2])

            if chart_type == "Continuous":
                fig = px.line(historical_data, x='Timestamp', y=selected_feature, title=f"Trend of {selected_feature}")
                col1.plotly_chart(fig, use_container_width=True)

                stats = historical_data[selected_feature].describe(percentiles=[0.25, 0.5, 0.75])
                null_pct = historical_data[selected_feature].isnull().mean() * 100

                with col2:
                    st.markdown("### Statistics")
                    st.markdown(f"- **Min**: {stats['min']:.2f}")
                    st.markdown(f"- **Max**: {stats['max']:.2f}")
                    st.markdown(f"- **Mean**: {stats['mean']:.2f}")
                    st.markdown(f"- **Null Values %**: {null_pct:.2f}%")
                    st.markdown(f"- **25th Percentile**: {stats['25%']:.2f}")
                    st.markdown(f"- **50th Percentile**: {stats['50%']:.2f}")
                    st.markdown(f"- **75th Percentile**: {stats['75%']:.2f}")

            else:
                top_counts = historical_data[selected_feature].value_counts().nlargest(5)
                fig = px.bar(x=top_counts.index, y=top_counts.values, labels={'x': selected_feature, 'y': 'Count'},
                             title=f"Top 5 {selected_feature} Frequencies")
                col1.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("### Top 5 Counts")
                    for cat, count in top_counts.items():
                        st.markdown(f"- **{cat}**: {count}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    pass  # No function named 'show_page' is defined, so this is replaced with a placeholder.
