import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from back import process_models_from_session  # <-- Use this function


st.title("Model Prediction Monitoring: Profitability, Charge-Off, Prepayment")

# --- Main App ---
def main():
    """
    Main function to run the Streamlit application.
    """
    processed_dataframes = process_models_from_session()  # <-- Use this function

    if processed_dataframes is None or not processed_dataframes:
        st.error("Failed to retrieve data from the processing step.")
        return

    # --- 1. Data Source Selection ---
    st.header("1. Data Source Selection")
    available_data_sources = {df.attrs.get('name', f'Unnamed DataFrame {i}'): df for i, df in enumerate(processed_dataframes)}

    if not available_data_sources:
        st.warning("No data available. Please ensure data has been processed.")
        return

    data_source_name = st.selectbox(
        "Select Data Source (Model)", options=list(available_data_sources.keys())
    )
    selected_df = available_data_sources[data_source_name]
    st.session_state.data_source = data_source_name

    # --- 2. Data Preprocessing (Origination_Year, Year, Months_elapsed) ---
    df_edited = selected_df.copy()

    # Threshold COF_EVENT_LABEL, PREPAYMENT_EVENT_LABEL, and conditionally Predicted_Target
    if 'COF_EVENT_LABEL' in df_edited.columns or 'PREPAYMENT_EVENT_LABEL' in df_edited.columns:
        if 'Predicted_Target' in df_edited.columns:
            df_edited['Predicted_Target'] = (df_edited['Predicted_Target'] >= 0.5).astype(int)
        for col in ['COF_EVENT_LABEL', 'PREPAYMENT_EVENT_LABEL']:
            if col in df_edited.columns:
                df_edited[col] = (df_edited[col] >= 0.5).astype(int)

    # 1. Calculate Origination Year
    if 'Timestamp' in df_edited.columns:
        print("Found 'Timestamp' column")
        print(f"Data type of 'Timestamp': {df_edited['Timestamp'].dtype}")

        # Check for null values in the 'Timestamp' column
        print(f"Number of null values in 'Timestamp': {df_edited['Timestamp'].isnull().sum()}")

        # Print some non-null values to inspect their format
        non_null_timestamps = df_edited['Timestamp'].dropna()
        if not non_null_timestamps.empty:
            print(f"First 5 non-null values of 'Timestamp': {non_null_timestamps.head().to_list()}")
            try:
                # Try converting only non-null values
                df_edited['Origination_Year'] = pd.to_datetime(df_edited['Timestamp'], errors='coerce').dt.year
                print("Calculated 'Origination_Year' from 'Timestamp'")
                print(f"First 5 values of 'Origination_Year': {df_edited['Origination_Year'].head().to_list()}")
            except Exception as e:
                print(f"Error calculating 'Origination_Year' from 'Timestamp': {e}")
                df_edited['Origination_Year'] = None
        else:
            print("'Timestamp' column contains only null values.")
            df_edited['Origination_Year'] = None

    elif 'date' in df_edited.columns:
        try:
            df_edited['Origination_Year'] = pd.to_datetime(df_edited['date'], errors='coerce').dt.year
        except Exception as e:
            print(f"Error calculating 'Origination_Year' from 'date': {e}")
            df_edited['Origination_Year'] = None
    elif 'loan_start_date' in df_edited.columns:
        try:
            df_edited['Origination_Year'] = pd.to_datetime(df_edited['loan_start_date'], errors='coerce').dt.year
        except Exception as e:
            print(f"Error calculating 'Origination_Year' from 'loan_start_date': {e}")
            df_edited['Origination_Year'] = None
    else:
        df_edited['Origination_Year'] = None
        print("Warning: No date/timestamp column found for 'Origination_Year' calculation.")

    # 2. Calculate 'Year' and subsequent rows, and modify 'Predicted_Target' and Target columns
    df_expanded = []
    if not df_edited.empty:
        for _, row in df_edited.iterrows():
            term_of_loan = int(row['TERM_OF_LOAN']) if 'TERM_OF_LOAN' in row and pd.notna(row['TERM_OF_LOAN']) else 0
            original_year = row.get('Origination_Year')
            original_predicted_target = row['Predicted_Target']  # Store original prediction
            original_profitability = row.get('Profitability_GBP', None)
            original_cof_event = row.get('COF_EVENT_LABEL', None)
            original_prepayment_event = row.get('PREPAYMENT_EVENT_LABEL', None)

            row_dict = row.to_dict()
            row_dict['Year'] = original_year
            df_expanded.append(row_dict)

            if original_year is not None and term_of_loan > 0:
                num_subsequent_rows = term_of_loan // 12
                percentages = np.linspace(0.2, 1.0, num_subsequent_rows)
                for i, percentage in enumerate(percentages):
                    subsequent_year = int(original_year + i + 1)
                    subsequent_row = row.to_dict()
                    subsequent_row['Year'] = subsequent_year
                    # Modify 'Predicted_Target' and Target columns for subsequent rows
                    if 'COF_EVENT_LABEL' in df_edited.columns or 'PREPAYMENT_EVENT_LABEL' in df_edited.columns:
                        subsequent_row['Predicted_Target'] = (original_predicted_target * percentage >= 0.5).astype(int) # threshold
                    else:
                        subsequent_row['Predicted_Target'] = original_predicted_target * percentage
                    if original_profitability is not None:
                        subsequent_row['Profitability_GBP'] = original_profitability * percentage
                    if original_cof_event is not None:
                        subsequent_row['COF_EVENT_LABEL'] = (original_cof_event * percentage >= 0.5).astype(int)
                    if original_prepayment_event is not None:
                        subsequent_row['PREPAYMENT_EVENT_LABEL'] = (original_prepayment_event * percentage >= 0.5).astype(int)
                    df_expanded.append(subsequent_row)
    df_expanded = pd.DataFrame(df_expanded)
    if not df_expanded.empty:
        df_edited = df_expanded.copy()

    # 3. Calculate Months_elapsed
    if 'Year' in df_edited.columns and 'Origination_Year' in df_edited.columns:
        df_edited['Months_elapsed'] = ((df_edited['Year'] - df_edited['Origination_Year']) * 12).fillna(0).astype(int)
    else:
        df_edited['Months_elapsed'] = 0

    # --- 2. Input Data Overview ---
    st.header("2. Input Data Overview")
    st.subheader("2.1. Key Columns")
    st.dataframe(df_edited.head(), use_container_width=True)

    # --- 3. Feature Selection ---
    st.header("3. Feature Selection")
    # Available features are now fixed.
    available_features = ['Origination_Year', 'TERM_OF_LOAN']
    selected_features = st.multiselect("Select feature(s) to filter by:", available_features, default=available_features)  # make default
 # --- 4. Bucket Selection ---
    bucket_filters = {}
    for feature in selected_features:
        if feature in df_edited.columns:
            values = df_edited[feature].dropna().unique().tolist()
            selected_values = st.multiselect(f"Select values for {feature}:", values, key=f"bucket_sel_{feature}")
            if selected_values:
                bucket_filters[feature] = selected_values

    # --- 5. Apply Filters ---
    df_filtered = df_edited.copy()
    if selected_features and bucket_filters:
        query_parts = []
        for feature, selected_values in bucket_filters.items():
            if feature in df_filtered.columns:
                query_parts.append(df_filtered[feature].isin(selected_values))
        if query_parts:
            query = np.logical_and.reduce(query_parts)
            df_filtered = df_filtered[query]

    if df_filtered.empty:
        st.error("No data matches the selected filters.")
        st.stop()

    # Process the selected dataframe
    df_edited = df_filtered.copy()

    # 2. Column Selection
    target_variable_present = None
    if 'Profitability_GBP' in df_edited.columns:
        target_variable_present = 'Profitability_GBP'
    elif 'COF_EVENT_LABEL' in df_edited.columns:
        target_variable_present = 'COF_EVENT_LABEL'
    elif 'PREPAYMENT_EVENT_LABEL' in df_edited.columns:
        target_variable_present = 'PREPAYMENT_EVENT_LABEL'

    required_columns = ['Timestamp', 'date', 'loan_start_date', 'TERM_OF_LOAN', target_variable_present, 'Predicted_Target', 'Origination_Year', 'Year', 'Months_elapsed']
    columns_to_keep = [col for col in required_columns if col in df_edited.columns]
    df_edited = df_edited[columns_to_keep].copy()

    # Ensure no None values before converting to int
    for col in ['Origination_Year', 'TERM_OF_LOAN', 'Year', 'Months_elapsed']:
        if col in df_edited.columns:
            df_edited[col] = df_edited[col].fillna(0)  # Replace None with 0, or another suitable default
            df_edited[col] = df_edited[col].astype(int)

    # --- 6. Model Evaluation ---
    st.header("4. Model Evaluation")

    if target_variable_present == 'Profitability_GBP':
        # --- 6.1 Profitability Analysis ---
        st.subheader("4.1 Profitability Analysis")
        actual_col = 'Profitability_GBP'

        if actual_col in df_edited.columns and 'Origination_Year' in df_edited.columns and 'Year' in df_edited.columns and 'Months_elapsed' in df_edited.columns:

            # Group by 'Months_elapsed' and sum 'Profitability_GBP' and 'Predicted_Target'
            grouped_df = df_edited.groupby('Months_elapsed').agg({
                'Profitability_GBP': 'sum',
                'Predicted_Target': 'sum'
            }).reset_index()

            if (grouped_df.shape[0] > 1):
                # Plotting
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=grouped_df['Months_elapsed'], y=grouped_df['Profitability_GBP'], mode='lines', name='Actual Profitability'))
                fig.add_trace(go.Scatter(x=grouped_df['Months_elapsed'], y=grouped_df['Predicted_Target'], mode='lines', name='Predicted Profitability'))
                fig.update_layout(title=f"Total Profitability and Prediction Over Time - {data_source_name}",
                                    xaxis_title="Months Elapsed",
                                    yaxis_title="Total Profitability GBP")
                st.plotly_chart(fig, use_container_width=True)

                # Calculate and display error metrics
                mse = mean_squared_error(grouped_df['Profitability_GBP'], grouped_df['Predicted_Target'])
                avg_error = np.mean(np.abs(grouped_df['Profitability_GBP'] - grouped_df['Predicted_Target']))
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"Average Error: {avg_error:.2f}")
            else:
                st.write("Not enough data to plot.")

        else:
            st.write("Profitability data is not available for the selected model.")
    elif target_variable_present in ['COF_EVENT_LABEL', 'PREPAYMENT_EVENT_LABEL']:
        # --- 6.2 Charge-Off and Prepayment Analysis ---
        st.subheader("4.2 Charge-Off and Prepayment Analysis")
        model_name = st.session_state.data_source
        df_model = df_edited

        # Use the thresholded values from df_model
        y_true = df_model[target_variable_present].astype(int)
        y_pred = df_model['Predicted_Target'].astype(int)

        # check if target variable has more than 2 unique values
        if len(y_true.unique()) <= 2:
            cm = confusion_matrix(y_true, y_pred)

            # Create heatmap for confusion matrix
            fig = go.Figure(data=go.Heatmap(z=cm,
                                           x=['Predicted 0', 'Predicted 1'],
                                           y=['Actual 0', 'Actual 1'],
                                           colorscale=[[0, 'rgb(255,255,255)'], [1, 'rgb(255,0,0)']], # White to Red
                                           colorbar=dict(title='Count'),
                                           texttemplate="%{z}",
                                           textfont={"color": "black"})) # Make text color black

            fig.update_layout(title=f"Confusion Matrix ({target_variable_present} vs. Predicted) - {model_name}",
                              xaxis_title='Predicted Label',
                              yaxis_title='Actual Label')
            st.plotly_chart(fig, use_container_width=True)

            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            st.text(f"Precision: {p:.4f}, Recall: {r:.4f}, F1 Score: {f1:.4f}")
        else:
            st.warning(f"Metrics (Confusion Matrix, Precision, Recall, F1) are not applicable for non-binary classification.")
    else:
        st.write("Charge-Off or Prepayment data is not available for the selected model.")


if __name__ == "__main__":
    main()

 
