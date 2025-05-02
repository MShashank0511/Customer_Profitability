import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import itertools

st.set_page_config(page_title="Loan Profitability Dashboard", layout="wide")

# -----------------------
# Load and preprocess data with bucketing
# -----------------------
@st.cache_data
def load_and_bucket_data():
    data = {}
    all_opb_values = pd.Series(dtype='float64')
    all_payment_values = pd.Series(dtype='float64')
    all_pti_values = pd.Series(dtype='float64')
    all_credit_values = pd.Series(dtype='float64')

    for year in range(2018, 2025):
        file = f"df_{year}.csv"
        if os.path.exists(file):
            df = pd.read_csv(file)
            df['YEAR'] = df['YEAR'].astype(int)
            df['Origination_year'] = df['Origination_year'].astype(int)
            df['Months_elapsed'] = df['Months_elapsed'].astype(int)

            if 'OPB' in df.columns:
                all_opb_values = pd.concat([all_opb_values, df['OPB'].dropna()])
            if 'PAYMENT_AMOUNT' in df.columns:
                all_payment_values = pd.concat([all_payment_values, df['PAYMENT_AMOUNT'].dropna()])
            if 'AUTO_PTI_TOTAL' in df.columns:
                all_pti_values = pd.concat([all_pti_values, df['AUTO_PTI_TOTAL'].dropna()])
            if 'CREDIT_SCORE_AVG_CALC' in df.columns:
                all_credit_values = pd.concat([all_credit_values, df['CREDIT_SCORE_AVG_CALC'].dropna()])
            data[year] = df

    def get_bucket_labels(series, low_q, high_q, labels):
        quantiles = series.quantile([0, low_q, high_q, 1]).unique().tolist()
        if len(quantiles) > 1:
            return [f"{labels[i]} ({quantiles[i]:.2f} to {quantiles[i+1]:.2f})" for i in range(min(len(labels), len(quantiles)-1))]
        return []

    opb_labels = get_bucket_labels(all_opb_values, 0.33, 0.67, ['Low', 'High', 'Medium'])
    payment_labels = get_bucket_labels(all_payment_values, 0.33, 0.67, ['Low', 'Medium', 'High'])
    pti_labels = get_bucket_labels(all_pti_values, 0.33, 0.67, ['Low', 'Medium', 'High'])
    credit_labels = get_bucket_labels(all_credit_values, 0.33, 0.67, ['Low', 'Medium', 'High'])

    for year, df in data.items():
        if 'OPB' in df.columns and opb_labels:
            df['OPB_BUCKET'] = pd.qcut(df['OPB'], q=[0, 0.33, 0.67, 1], labels=opb_labels, duplicates='drop').astype(str)
        else:
            df['OPB_BUCKET'] = 'NA'

        if 'PAYMENT_AMOUNT' in df.columns and payment_labels:
            df['PAYMENT_AMOUNT_BUCKET'] = pd.qcut(df['PAYMENT_AMOUNT'], q=[0, 0.33, 0.67, 1], labels=payment_labels, duplicates='drop').astype(str)
        else:
            df['PAYMENT_AMOUNT_BUCKET'] = 'NA'

        if 'AUTO_PTI_TOTAL' in df.columns and pti_labels:
            df['AUTO_PTI_TOTAL_BUCKET'] = pd.qcut(df['AUTO_PTI_TOTAL'], q=[0, 0.33, 0.67, 1], labels=pti_labels, duplicates='drop').astype(str)
        else:
            df['AUTO_PTI_TOTAL_BUCKET'] = 'NA'

        if 'TERM_OF_LOAN' in df.columns:
            df['TERM_OF_LOAN_BUCKET'] = df['TERM_OF_LOAN'].astype(str)

        if 'Origination_year' in df.columns:
            df['Origination_year_BUCKET'] = df['Origination_year'].astype(str)

        if 'CREDIT_SCORE_BUCKET' not in df.columns and 'CREDIT_SCORE_AVG_CALC' in df.columns and credit_labels:
            df['CREDIT_SCORE_BUCKET'] = pd.qcut(df['CREDIT_SCORE_AVG_CALC'], q=[0, 0.33, 0.67, 1], labels=credit_labels, duplicates='drop').astype(str)
        elif 'CREDIT_SCORE_BUCKET' not in df.columns:
            df['CREDIT_SCORE_BUCKET'] = 'NA'

    return data

data_by_year = load_and_bucket_data()

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("Navigation")
st.sidebar.markdown("Currently on: **Results Page**")

# -----------------------
# Section 1: Default Overview Chart
# -----------------------
st.title("📊 Loan Profitability Overview")

orig_years_available = sorted(data_by_year.keys())
selected_orig_years = st.multiselect(
    "Select Origination Year(s) to Show (Default: 2018)", orig_years_available, default=[2018]
)

thresholds = {2018: float('inf'), 2019: 72, 2020: 60, 2021: 48, 2022: 36, 2023: 24, 2024: 12}

overview_fig = go.Figure()

for year in selected_orig_years:
    df = data_by_year[year].copy()
    if df.empty:
        continue

    grouped = df.groupby(["Months_elapsed", "Origination_year"])["Profitability"].sum().reset_index()

    for orig_year in grouped["Origination_year"].unique():
        threshold = thresholds.get(orig_year, 0)
        df_solid = grouped[(grouped["Origination_year"] == orig_year) & (grouped["Months_elapsed"] <= threshold)]
        df_dash = grouped[(grouped["Origination_year"] == orig_year) & (grouped["Months_elapsed"] > threshold)]

        overview_fig.add_trace(go.Scatter(
            x=df_solid["Months_elapsed"],
            y=df_solid["Profitability"],
            mode="lines+markers",
            name=f"{orig_year} (Actual)",
            line=dict(shape="linear", dash="solid"),
        ))
        if not df_dash.empty:
            overview_fig.add_trace(go.Scatter(
                x=df_dash["Months_elapsed"],
                y=df_dash["Profitability"],
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


sample_df = next(iter(data_by_year.values()))
available_bucket_features = [col for col in sample_df.columns if col.endswith('_BUCKET')]
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
    unique_vals = set()
    for df in data_by_year.values():
        if feat in df.columns:
            unique_vals.update(df[feat].dropna().unique())
    if unique_vals:
        selected = st.multiselect(f"Select values for **{feature_display_names[feat]}**:", sorted(unique_vals), key=feat)
        if selected:
            selected_buckets[feat] = selected

if selected_buckets and all(selected_buckets.values()):
    st.subheader("📈 Profitability Trends for Selected Combinations")

    bucket_keys = list(selected_buckets.keys())
    combinations = list(itertools.product(*[selected_buckets[k] for k in bucket_keys]))

    all_combinations_fig = go.Figure()
    has_data = False
    error_message = ""

    for combo in combinations:
        label = " | ".join([f"{feature_display_names[k]}: {v}" for k, v in zip(bucket_keys, combo)])

        combined_df = pd.DataFrame()
        for year, df in data_by_year.items():
            filters = [(df[bucket_keys[i]] == combo[i]) for i in range(len(combo))]
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter &= f
            filtered = df[combined_filter].copy()

            if not filtered.empty:
                has_data = True
                grouped = filtered.groupby(["Months_elapsed", "Origination_year"])["Profitability"].sum().reset_index()
                combined_df = pd.concat([combined_df, grouped])

        if not combined_df.empty:
            for orig_year in combined_df['Origination_year'].unique():
                threshold = thresholds.get(orig_year, 0)
                df_combo_year = combined_df[combined_df['Origination_year'] == orig_year]
                df_solid = df_combo_year[df_combo_year["Months_elapsed"] <= threshold]
                df_dash = df_combo_year[df_combo_year["Months_elapsed"] > threshold]

                if not df_solid.empty:
                    all_combinations_fig.add_trace(go.Scatter(
                        x=df_solid["Months_elapsed"],
                        y=df_solid["Profitability"],
                        mode="lines+markers",
                        name=f"{label} | {orig_year} (Actual)",
                        line=dict(shape="linear", dash="solid"),
                    ))
                if not df_dash.empty:
                    all_combinations_fig.add_trace(go.Scatter(
                        x=df_dash["Months_elapsed"],
                        y=df_dash["Profitability"],
                        mode="lines+markers",
                        name=f"{label} | {orig_year} (Predicted)",
                        line=dict(shape="linear", dash="dash"),
                    ))
        else:
            error_message += f"No data available for combination: {label}.<br>"

    if has_data:
        all_combinations_fig.update_layout(
            title="Profitability vs Months Elapsed for Selected Feature Combinations",
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
        st.plotly_chart(all_combinations_fig, use_container_width=False)
    else:
        st.error("No data available for any of the selected feature combinations. Please adjust your selections.")
    if error_message:
        st.markdown(f"<span style='color:red'>{error_message}</span>", unsafe_allow_html=True)