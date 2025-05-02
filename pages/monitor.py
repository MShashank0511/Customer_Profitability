import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import itertools

st.set_page_config(page_title="Feature-Based Profitability Analysis", layout="wide")

# ------------------------
# Load Preprocessed Data
# ------------------------
@st.cache_data
def load_combined_data():
    df = pd.read_csv("combined_df.csv")

    # Ensure correct types
    df['Origination_year'] = df['Origination_year'].astype(str)
    df['TERM_OF_LOAN'] = df['TERM_OF_LOAN'].astype(str)

    def create_buckets(series, labels):
        quantiles = series.quantile([0, 0.33, 0.67, 1]).values
        quantiles = np.unique(quantiles)
        if len(quantiles) < 4:
            return pd.Series(["NA"] * len(series))
        bins = pd.cut(series, bins=quantiles, labels=[
            f"{labels[i]} ({quantiles[i]:.2f} to {quantiles[i+1]:.2f})"
            for i in range(len(labels))], include_lowest=True, duplicates='drop')
        return bins

    if 'OPB' in df.columns:
        df['OPB_BUCKET'] = create_buckets(df['OPB'], ['Low', 'Medium', 'High'])

    if 'PAYMENT_AMOUNT' in df.columns:
        df['PAYMENT_AMOUNT_BUCKET'] = create_buckets(df['PAYMENT_AMOUNT'], ['Low', 'Medium', 'High'])

    if 'AUTO_PTI_TOTAL' in df.columns:
        df['AUTO_PTI_TOTAL_BUCKET'] = create_buckets(df['AUTO_PTI_TOTAL'], ['Low', 'Medium', 'High'])

    if 'CREDIT_SCORE_BUCKET' not in df.columns and 'CREDIT_SCORE_AVG_CALC' in df.columns:
        df['CREDIT_SCORE_BUCKET'] = create_buckets(df['CREDIT_SCORE_AVG_CALC'], ['Low', 'Medium', 'High'])

    df = df[df['Origination_year'] != '2025']  # Exclude 2025

    df['TERM_OF_LOAN_BUCKET'] = df['TERM_OF_LOAN']
    df['Origination_year_BUCKET'] = df['Origination_year']

    return df

combined_df = load_combined_data()

# ------------------------
# Feature Selection
# ------------------------
st.title("🔍 Monitoring Profitability Analysis")

available_bucket_features = [col for col in combined_df.columns if col.endswith('_BUCKET')]
feature_display_names = {col: col.replace('_BUCKET', '') for col in available_bucket_features}

selected_features = st.multiselect(
    "Select any number of features for segmentation:",
    options=available_bucket_features,
    format_func=lambda x: feature_display_names[x]
)

selected_buckets = {}
if selected_features:
    for feat in selected_features:
        unique_vals = sorted(combined_df[feat].dropna().unique())
        selected = st.multiselect(f"Select values for **{feature_display_names[feat]}**:", unique_vals, key=feat)
        if selected:
            selected_buckets[feat] = selected

proceed = st.button("Proceed")

if proceed and selected_buckets and all(selected_buckets.values()):
    st.subheader("📈 Predicted Profitability vs Actual Profitability")
    combos = list(itertools.product(*[selected_buckets[k] for k in selected_buckets]))
    st.session_state.combo_dfs = {}

    for combo in combos:
        label = " | ".join([f"{feature_display_names[k]}: {v}" for k, v in zip(selected_buckets.keys(), combo)])
        filt = np.logical_and.reduce([
            combined_df[list(selected_buckets.keys())[i]] == combo[i]
            for i in range(len(combo))
        ])
        df_subset = combined_df[filt].copy()
        if df_subset.empty:
            st.warning(f"No data found for combination: {label}")
            continue

        grouped = df_subset.groupby("Months_elapsed")[["Profitability", "Actual_profitability"]].sum().reset_index()

        # Error only if values differ
        grouped["Error"] = np.where(
            grouped["Profitability"].round(2) == grouped["Actual_profitability"].round(2), 0,
            abs(grouped["Profitability"] - grouped["Actual_profitability"]) / grouped["Actual_profitability"] * 100
        )
        mean_error = grouped["Error"].mean()
        st.session_state.combo_dfs[tuple(combo)] = (df_subset, grouped, mean_error, label)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=grouped["Months_elapsed"], y=grouped["Profitability"],
            mode="lines+markers", name="Predicted Profitability", line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=grouped["Months_elapsed"], y=grouped["Actual_profitability"],
            mode="lines+markers", name="Actual Profitability", line=dict(color="green")
        ))

        fig.update_layout(
            title=f"{label}",
            xaxis_title="Months Elapsed",
            yaxis_title="Profitability",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Average Error (%)**: {mean_error:.2f}")

    st.markdown("---")
    st.subheader("🔧 Readjustment of Graphs")
    st.button("Readjust the graphs", key="readjust")

if st.session_state.get("readjust"):
    st.subheader("📉 Adjusted Profitability vs Actual Profitability")

    if 'combo_dfs' in st.session_state and st.session_state.combo_dfs:
        for combo, (df_subset, _, _, label) in st.session_state.combo_dfs.items():
            actual_sum = df_subset["Actual_profitability"].sum()
            predicted_sum = df_subset["Profitability"].sum()
            factor = actual_sum / predicted_sum if predicted_sum != 0 else 1.0

            df_subset["Adjusted_profitability"] = df_subset["Profitability"] * factor

            grouped = df_subset.groupby("Months_elapsed")[["Adjusted_profitability", "Actual_profitability"]].sum().reset_index()

            grouped["Error"] = np.where(
                grouped["Adjusted_profitability"].round(2) == grouped["Actual_profitability"].round(2), 0,
                abs(grouped["Adjusted_profitability"] - grouped["Actual_profitability"]) / grouped["Actual_profitability"] * 100
            )
            new_error = grouped["Error"].mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=grouped["Months_elapsed"], y=grouped["Adjusted_profitability"],
                mode="lines+markers", name="Adjusted Profitability", line=dict(color="orange")
            ))
            fig.add_trace(go.Scatter(
                x=grouped["Months_elapsed"], y=grouped["Actual_profitability"],
                mode="lines+markers", name="Actual Profitability", line=dict(color="green")
            ))

            fig.update_layout(
                title=f"{label} (Adjusted)",
                xaxis_title="Months Elapsed",
                yaxis_title="Profitability",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**New Average Error (%)**: {new_error:.2f}")
            st.markdown(f"**Adjustment Factor Applied**: {factor:.4f}")