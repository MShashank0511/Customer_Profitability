import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error, roc_curve, roc_auc_score
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend import process_models_from_session  # <-- Use this function
def generate_chargeoff_trend(term, skew_start=0.2, skew_middle=0.5, skew_end=0.8, weights=(1.0, 1.0, 1.0)):
    """
    Generates a custom skewed Gaussian trend across loan term.
    Args:
        term (int): Number of months.
        skew_* (float): Skew centers.
        weights (tuple): Weights for start, middle, and end phases.
    Returns:
        np.ndarray: Normalized trend values over months.
    """
    x = np.linspace(0, 1, term)
    w_start, w_middle, w_end = weights

    g_start = np.exp(-((x - skew_start) ** 2) / 0.02) * w_start
    g_middle = np.exp(-((x - skew_middle) ** 2) / 0.02) * w_middle
    g_end = np.exp(-((x - skew_end) ** 2) / 0.02) * w_end

    trend = g_start + g_middle + g_end
    return trend / trend.sum()

def split_and_distribute_with_normalization(df, prob_col='Predicted_Probability', term_col='TERM_OF_LOAN', max_value=0.3, scaling_factor=1.2):
    """
    Splits each row into monthly rows and distributes the predicted probability using a realistic trend.
    Ensures that the distributed probabilities are scaled up while maintaining the pattern.

    Args:
        df (pd.DataFrame): The input DataFrame.
        prob_col (str): The column containing the total probability.
        term_col (str): The column containing the loan term in months.
        max_value (float): The maximum allowed value for any monthly probability.
        scaling_factor (float): The factor by which to scale the probabilities.

    Returns:
        pd.DataFrame: The expanded DataFrame with distributed probabilities.
    """
    expanded_rows = []

    for _, row in df.iterrows():
        term = int(row[term_col])
        total_prob = row[prob_col]

        # Generate a skewed trend for distribution
        trend = generate_chargeoff_trend(term, skew_start=0.2, skew_middle=0.5, skew_end=0.8)
        distributed_probs = trend * total_prob

        # Apply the scaling factor to increase probabilities
        distributed_probs *= scaling_factor

        # Ensure the sum of distributed probabilities does not exceed the original total probability
        distributed_probs = distributed_probs * (total_prob / distributed_probs.sum())

        # Ensure no individual month's probability exceeds the max threshold
        distributed_probs = np.clip(distributed_probs, 0, max_value)

        for month in range(1, term + 1):
            new_row = row.to_dict()  # Convert the row to a dictionary
            new_row['Month'] = month
            new_row['Distributed_Probability'] = distributed_probs[month - 1]
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)

def split_and_distribute_with_credit_risk(df, prob_col='Predicted_Probability', term_col='TERM_OF_LOAN',
                                           max_value=0.3, scaling_factor=1.2,
                                           credit_col='CREDIT_SCORE_AVG_CALC' , delinq_col='DELINQ_CNT_30_DAY_TOTAL' ):
    """
    Splits and distributes probabilities per loan using risk-adjusted temporal patterns.

    Returns:
        pd.DataFrame: Expanded DataFrame with distributed values.
    """
    expanded_rows = []

    for _, row in df.iterrows():
        term = int(row[term_col])
        total_prob = row[prob_col]
        credit_score = row[credit_col]
        delinquency = row[delinq_col]

        # Risk-adjusted Gaussian weighting
        weights = map_credit_risk_params(credit_score, delinquency)
        trend = generate_chargeoff_trend(term, weights=weights)

        distributed_probs = trend * total_prob
        distributed_probs *= scaling_factor

        # Normalize to preserve total probability
        distributed_probs = distributed_probs * (total_prob / distributed_probs.sum())

        # Cap max probability per month
        distributed_probs = np.clip(distributed_probs, 0, max_value)

        for month in range(1, term + 1):
            new_row = row.to_dict()
            new_row['Month'] = month
            new_row['Distributed_Probability'] = distributed_probs[month - 1]
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)

st.set_page_config(layout="wide")
st.title("Model Prediction Monitoring: Profitability, Charge-Off, Prepayment")

# --- Backend Function (Now within this file) ---
def adjust_predicted_target(df: pd.DataFrame, avg_error: float) -> tuple[pd.DataFrame, float]:
    """
    Adjusts the 'Predicted_Target' column in a DataFrame to reduce the average error
    in predicting 'Profitability_GBP'.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'Predicted_Target' and 'Profitability_GBP'.
        avg_error (float): The average error.

    Returns:
        tuple[pd.DataFrame, float]: A tuple containing the adjusted DataFrame and the adjustment factor.
                         Returns (None, 0) if adjustment fails or input is invalid.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input 'df' must be a pandas DataFrame.")
        return None, 0.0

    if 'Predicted_Target' not in df.columns or 'Profitability_GBP' not in df.columns:
        print("Error: DataFrame must contain 'Predicted_Target' and 'Profitability_GBP' columns.")
        return None, 0.0

    if not np.isscalar(avg_error):
        print("Error: 'avg_error' must be a scalar value.")
        return None, 0.0

    if avg_error <= 0.2:
        print("Average error is already within the acceptable range (<= 0.2). No adjustment needed.")
        return df.copy(), 1.0  # Return a copy to avoid modifying the original DataFrame


    try:
        # Calculate the sum of 'Profitability_GBP' and 'Predicted_Target' for each 'Months_elapsed'
        grouped_df = df.groupby('Months_elapsed').agg({
            'Profitability_GBP': 'sum',
            'Predicted_Target': 'sum'
        }).reset_index()

        # Calculate the adjustment factor.  Cap it.
        factor = np.clip(np.mean(grouped_df['Profitability_GBP'] / grouped_df['Predicted_Target']), 0.1, 5) # 0.1 and 5 are arbitrary caps
        print(f"Calculated adjustment factor: {factor}")

        # Adjust 'Predicted_Target'
        adjusted_df = df.copy()  # Create a copy to avoid modifying the original DataFrame in place
        adjusted_df['Predicted_Target'] = adjusted_df['Predicted_Target'] * factor

        return adjusted_df, factor

    except Exception as e:
        print(f"An error occurred during adjustment: {e}")
        return None, 0.0

# --- Function to store data in session state ---
# Declare combined_df as a global variable
combined_df = None

def store_processed_data(data: pd.DataFrame = None, data_source_name: str = None) -> pd.DataFrame:
    """
    Stores or retrieves the processed DataFrame globally and serializes it using pickle.

    Args:
        data (pd.DataFrame, optional): The processed DataFrame to store. Defaults to None.
        data_source_name (str, optional): The name of the data source. Defaults to None.

    Returns:
        pd.DataFrame: The stored DataFrame if no data is provided.
    """
    global combined_df
    if data is not None:
        combined_df = data  # Store the DataFrame globally
        print(f"Stored processed data for: {data_source_name}")
        # Serialize the DataFrame to a pickle file
        with open("combined_df.pkl", "wb") as f:
            pickle.dump(combined_df, f)
        print("DataFrame serialized to 'combined_df.pkl'")
    elif combined_df is not None:
        return combined_df  # Retrieve the stored DataFrame
    else:
        raise ValueError("No data available. Please run Valid.py first.")

def get_combined_df() -> pd.DataFrame:
    """
    Returns the combined DataFrame.

    Returns:
        pd.DataFrame: The combined DataFrame.
    """
    global combined_df
    if combined_df is None:
        raise ValueError("The combined DataFrame is not available. Please run Valid.py first.")
    return combined_df

# --- Function to add OPB column ---
def add_opb_column(df, loan_data_path='loan_data.csv'):
    """
    Adds the OPB column to the DataFrame if it is not already present.
    The OPB values are mapped using the Timestamp_x column from loan_data.csv.

    Args:
        df (pd.DataFrame): The DataFrame to which the OPB column will be added.
        loan_data_path (str): Path to the loan_data.csv file.

    Returns:
        pd.DataFrame: The updated DataFrame with the OPB column.
    """
    if 'OPB' not in df.columns:
        try:
            # Load loan_data.csv
            loan_data = pd.read_csv(loan_data_path)

            # Ensure Timestamp_x column exists in both DataFrames
            if 'Timestamp_x' in df.columns and 'Timestamp_x' in loan_data.columns:
                # Convert Timestamp_x columns to datetime for alignment
                loan_data['Timestamp_x'] = pd.to_datetime(loan_data['Timestamp_x'], errors='coerce')
                df['Timestamp_x'] = pd.to_datetime(df['Timestamp_x'], errors='coerce')

                # Map OPB values from loan_data.csv to the DataFrame using the Timestamp_x column
                df = df.merge(loan_data[['Timestamp_x', 'OPB']], on='Timestamp_x', how='left')
                st.write("Added OPB column to the DataFrame using loan_data.csv.")
            else:
                st.warning("Timestamp_x column is missing in either the DataFrame or loan_data.csv. Cannot map OPB values.")
        except FileNotFoundError:
            st.error(f"Error: The file '{loan_data_path}' was not found. Please ensure it exists.")
        except Exception as e:
            st.error(f"Error adding OPB column: {e}")
    else:
        st.write("OPB column already exists in the DataFrame.")
    return df

# Function to add loan metrics
def add_loan_metrics_updated(df, interest_rate=0.012, fee_rate=0.005):
    """
    Adds loan metrics to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        interest_rate (float): The interest rate to calculate interest amount.
        fee_rate (float): The fee rate to calculate fees.

    Returns:
        pd.DataFrame: The updated DataFrame with new loan metrics.
    """
    r = 0.012  # Monthly interest rate (1.2% per month)
    # Step 1: Calculate Outstanding Principal
    df['Outstanding_Principal'] = df['OPB'] * (
    (1 + r) ** df['TERM_OF_LOAN'] - (1 + r) ** df['Month']
) / (
    (1 + r) ** df['TERM_OF_LOAN'] - 1
)
    df['Outstanding_Principal'] = df['Outstanding_Principal'].clip(lower=0)

    # Step 2: Calculate Remaining Balance based on Total_Probability
    df['Remaining_Balance'] = (1 - df['Total_Probability']) * df['Outstanding_Principal']

    # Step 3: Derived columns based on Remaining_Balance
    df['Charge_Off_Bal'] = df['Remaining_Balance'] * df['COF_EVENT_LABEL']
    df['Interest_Amount'] = df['Remaining_Balance'] * interest_rate
    df['Fees'] = df['Remaining_Balance'] * fee_rate
    df['Recovery_Amount'] = df['Charge_Off_Bal'] * 0.05

    return df

#Function to apply cumulative columns
def apply_cumulative_columns(df, loan_id_cols=['OPB', 'TERM_OF_LOAN']):
    """
    Overwrites existing columns with their cumulative values, except for 'Charge_Off_Bal',
    which retains its actual values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        loan_id_cols (list): Columns to group by for creating Loan_ID.

    Returns:
        pd.DataFrame: The updated DataFrame with cumulative values.
    """
    # Create a Loan_ID if not already available
    df = df.copy()
    if 'LOAN_ID' not in df.columns:
        df['LOAN_ID'] = df.groupby(loan_id_cols).ngroup()

    # Sort by LOAN_ID and Month to ensure proper cumsum order
    df = df.sort_values(by=['LOAN_ID', 'Month'])

    # Overwrite the selected columns with their cumulative sums, except 'Charge_Off_Bal'
    cumulative_cols = ['Charge_Off_Bal', 'Interest_Amount', 'Fees', 'Recovery_Amount']
    for col in cumulative_cols:
        if col != 'Charge_Off_Bal':  # Skip cumulative sum for 'Charge_Off_Bal'
            df[col] = df.groupby('LOAN_ID')[col].cumsum()

    return df



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
    
    # Add OPB column if not present
    df_edited = add_opb_column(df_edited)

    # Determine the target variable
    target_variable_present = None
    if 'Profitability_GBP' in df_edited.columns:
        target_variable_present = 'Profitability_GBP'
    elif 'COF_EVENT_LABEL' in df_edited.columns:
        target_variable_present = 'COF_EVENT_LABEL'
    elif 'PREPAYMENT_EVENT_LABEL' in df_edited.columns:
        target_variable_present = 'PREPAYMENT_EVENT_LABEL'
    
    # Convert TERM_OF_LOAN values to either 60 or 84 for non-Profitability_GBP target variables
    if target_variable_present in ['COF_EVENT_LABEL', 'PREPAYMENT_EVENT_LABEL']:
        if 'TERM_OF_LOAN' in df_edited.columns:
            df_edited['TERM_OF_LOAN'] = df_edited['TERM_OF_LOAN'].apply(lambda x: 60 if x <= 60 else 84)
            st.write("Converted TERM_OF_LOAN values to either 60 or 84.")
    
    # Calculate Origination Year for all DataFrames
    if 'Timestamp_x' in df_edited.columns:
        df_edited['Origination_Year'] = pd.to_datetime(df_edited['Timestamp_x'], errors='coerce').dt.year
    elif 'date' in df_edited.columns:
        df_edited['Origination_Year'] = pd.to_datetime(df_edited['date'], errors='coerce').dt.year
    elif 'loan_start_date' in df_edited.columns:
        df_edited['Origination_Year'] = pd.to_datetime(df_edited['loan_start_date'], errors='coerce').dt.year
    else:
        df_edited['Origination_Year'] = None

    # --- 2. Input Data Overview ---
    st.header("2. Input Data Overview")
    st.subheader("2.1. Key Columns")
    st.dataframe(df_edited.head(), use_container_width=True)

    # --- 3. Feature Selection ---
    st.header("3. Feature Selection")
    available_features = ['Origination_Year', 'TERM_OF_LOAN']
    selected_features = st.multiselect(
        "Select feature(s) to filter by:", 
        available_features, 
        default=available_features  # Make both features selected by default
    )

    # --- 4. Bucket Selection ---
    bucket_filters = {}
    for feature in selected_features:
        if feature in df_edited.columns:
            # Get unique values for the feature
            values = df_edited[feature].dropna().unique().tolist()
            # Sort the values for better user experience
            values = sorted(values)
            # Create a multiselect widget for the feature
            selected_values = st.multiselect(
                f"Select values for {feature}:", 
                values, 
                key=f"bucket_sel_{feature}"
            )
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

    # Ensure required columns are present
    required_columns = ['OPB', 'Predicted_Probability', 'Timestamp_x', 'TERM_OF_LOAN', 
                        'COF_EVENT_LABEL', 'PREPAYMENT_EVENT_LABEL', 
                        'CREDIT_SCORE_AVG_CALC', 'DELINQ_CNT_30_DAY_TOTAL','CREDIT_SCORE_AVG_CALC_x','DELINQ_CNT_30_DAY_TOTAL_x']
    for col in required_columns:
        if col not in df_filtered.columns and col in df_edited.columns:
            df_filtered[col] = df_edited[col]

    if df_filtered.empty:
        st.error("No data matches the selected filters.")
        st.stop()

    # Process the selected dataframe
    df_edited = df_filtered.copy()

    # Split rows for the current DataFrame using credit risk
    try:
        df_model_split = split_and_distribute_with_credit_risk(df_edited)
        st.write("Applied split_and_distribute_with_credit_risk to the data.")
    except Exception as e:
        st.error(f"Error applying split_and_distribute_with_credit_risk: {e}")
        return

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
        
        # Display ROC AUC Curve based on predictions before splitting rows
        if target_variable_present in df_model.columns and 'Predicted_Probability' in df_model.columns:
            y_true = df_model[target_variable_present]
            y_pred_proba = df_model['Predicted_Probability']

            if len(y_true.unique()) <= 2:  # Ensure binary classification
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                auc = roc_auc_score(y_true, y_pred_proba)

                # Plot ROC Curve
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc:.2f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash', color='gray')))
                fig.update_layout(
                    title=f"ROC Curve for {target_variable_present} - {model_name}",
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                )
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"ROC AUC Score: {auc:.4f}")
        
        
            # Convert TERM_OF_LOAN values to either 60 or 84
        if 'TERM_OF_LOAN' in df_model.columns:
            df_model['TERM_OF_LOAN'] = df_model['TERM_OF_LOAN'].apply(lambda x: 60 if x <= 60 else 84)

        # Ensure 'Predicted_Probability' exists
        if 'Predicted_Probability' not in df_model.columns:
            if 'Predicted_Target' in df_model.columns:
                df_model['Predicted_Probability'] = df_model['Predicted_Target']
                st.write("Mapped 'Predicted_Probability' from 'Predicted_Target'.")
            else:
                st.error("The required column 'Predicted_Probability' is missing. Please ensure the model outputs probabilities.")
                return


        # Split rows for the current DataFrame
        try:
            df_model_split = split_and_distribute_with_normalization(df_model)
            st.write("Applied split_and_distribute_with_normalization to the data.")
        except Exception as e:
            st.error(f"Error applying split_and_distribute_with_normalization: {e}")
            return

        # Store the processed data for the current target variable
        if target_variable_present == 'COF_EVENT_LABEL':
            st.session_state.cof_event_df = df_model_split
        elif target_variable_present == 'PREPAYMENT_EVENT_LABEL':
            st.session_state.prepayment_event_df = df_model_split
        
        # Combine DataFrames if both are available
        if 'cof_event_df' in st.session_state and 'prepayment_event_df' in st.session_state:
            cof_event_df = st.session_state.cof_event_df
            prepayment_event_df = st.session_state.prepayment_event_df

            # Merge the two DataFrames on Timestamp_x and Month
            combined_df = pd.merge(
                cof_event_df[['Timestamp_x', 'Month', 'Distributed_Probability']],
                prepayment_event_df[['Timestamp_x', 'Month', 'Distributed_Probability']],
                on=['Timestamp_x', 'Month'],
                how='outer',
                suffixes=('_COF', '_PREPAYMENT')
            )

            # Add OPB, Origination_Year, TERM_OF_LOAN, and target columns
            combined_df = combined_df.merge(
                df_model[['Timestamp_x', 'OPB', 'Origination_Year', 'TERM_OF_LOAN']].drop_duplicates(),
                on='Timestamp_x',
                how='left'
            )
            combined_df['COF_EVENT_LABEL'] = combined_df['Distributed_Probability_COF']
            combined_df['PREPAYMENT_EVENT_LABEL'] = combined_df['Distributed_Probability_PREPAYMENT']

            # Calculate Total_Probability
            combined_df['Total_Probability'] = combined_df[['Distributed_Probability_COF', 'Distributed_Probability_PREPAYMENT']].sum(axis=1)

            # Select relevant columns
            combined_df = combined_df[['Timestamp_x', 'OPB', 'COF_EVENT_LABEL', 'PREPAYMENT_EVENT_LABEL', 'Origination_Year', 'TERM_OF_LOAN', 'Month', 'Total_Probability']]

            # Add loan metrics to the combined DataFrame
            combined_df = add_loan_metrics_updated(combined_df)

            # Apply cumulative values to the existing columns
            combined_df = apply_cumulative_columns(combined_df)

            # Store the combined DataFrame with cumulative values
            store_processed_data(combined_df, "Combined_COFPREPAYMENT")

            

            # Send the DataFrame to the next page
            st.session_state.final_combined_df = combined_df
            st.write("DataFrame has been sent to the next page.")
        else:
            st.error("No predictions available for COF_EVENT_LABEL or PREPAYMENT_EVENT_LABEL.")
    else:
        st.write("Charge-Off or Prepayment data is not available for the selected model.")
        if target_variable_present is not None:
            final_df_to_store = df_edited.copy()
            try:
                final_df_to_store = split_and_distribute_with_normalization(final_df_to_store)
                st.write("Applied split_and_distribute_with_normalization to the data.")
            except Exception as e:
                st.error(f"Error applying split_and_distribute_with_normalization: {e}")

            store_processed_data(final_df_to_store, data_source_name)

    # --- Combine DataFrames for COF_EVENT_LABEL and PREPAYMENT_EVENT_LABEL ---
    st.header("4. Combined DataFrame for COF and Prepayment")

    # Retrieve processed DataFrames from session state
    cof_event_df = st.session_state.get('cof_event_df', None)
    prepayment_event_df = st.session_state.get('prepayment_event_df', None)

    # Ensure 'Timestamp_x' exists and is consistent in all DataFrames
    if cof_event_df is not None:
        cof_event_df['Timestamp_x'] = pd.to_datetime(cof_event_df['Timestamp_x'], errors='coerce')
    if prepayment_event_df is not None:
        prepayment_event_df['Timestamp_x'] = pd.to_datetime(prepayment_event_df['Timestamp_x'], errors='coerce')
    df_edited['Timestamp_x'] = pd.to_datetime(df_edited['Timestamp_x'], errors='coerce')

    # Merge the two DataFrames
    if cof_event_df is not None and prepayment_event_df is not None:
        combined_df = pd.merge(
            cof_event_df[['Timestamp_x', 'Month', 'Distributed_Probability']].rename(
                columns={'Distributed_Probability': 'Distributed_Probability_COF'}
            ),
            prepayment_event_df[['Timestamp_x', 'Month', 'Distributed_Probability']].rename(
                columns={'Distributed_Probability': 'Distributed_Probability_PREPAYMENT'}
            ),
            on=['Timestamp_x', 'Month'],
            how='outer'
        )
    elif cof_event_df is not None:
        combined_df = cof_event_df.rename(
            columns={'Distributed_Probability': 'Distributed_Probability_COF'}
        )
        combined_df['Distributed_Probability_PREPAYMENT'] = 0
    elif prepayment_event_df is not None:
        combined_df = prepayment_event_df.rename(
            columns={'Distributed_Probability': 'Distributed_Probability_PREPAYMENT'}
        )
        combined_df['Distributed_Probability_COF'] = 0
    else:
        st.error("No data available for COF_EVENT_LABEL or PREPAYMENT_EVENT_LABEL.")
        return

    # Add OPB, Origination_Year, TERM_OF_LOAN, and other columns
    combined_df = combined_df.merge(
        df_edited[['Timestamp_x', 'OPB', 'Origination_Year', 'TERM_OF_LOAN']].drop_duplicates(),
        on='Timestamp_x',
        how='left'
    )

    # Ensure no None values in Distributed_Probability columns
    combined_df['Distributed_Probability_COF'] = combined_df['Distributed_Probability_COF'].fillna(0)
    combined_df['Distributed_Probability_PREPAYMENT'] = combined_df['Distributed_Probability_PREPAYMENT'].fillna(0)

    # Map probabilities to target columns
    combined_df['COF_EVENT_LABEL'] = combined_df['Distributed_Probability_COF']
    combined_df['PREPAYMENT_EVENT_LABEL'] = combined_df['Distributed_Probability_PREPAYMENT']

    # Handle cases where only one DataFrame is selected
    if 'cof_event_df' in st.session_state and 'Distributed_Probability_COF' in combined_df.columns:
        combined_df['COF_EVENT_LABEL'] = combined_df['Distributed_Probability_COF']
    if 'prepayment_event_df' in st.session_state and 'Distributed_Probability_PREPAYMENT' in combined_df.columns:
        combined_df['PREPAYMENT_EVENT_LABEL'] = combined_df['Distributed_Probability_PREPAYMENT']

    # Calculate Total_Probability
    combined_df['Total_Probability'] = combined_df[['Distributed_Probability_COF', 'Distributed_Probability_PREPAYMENT']].sum(axis=1)

    # Add loan metrics to the combined DataFrame
    combined_df = add_loan_metrics_updated(combined_df) 

    # Apply cumulative values to the existing columns
    combined_df = apply_cumulative_columns(combined_df)

    # Display the combined DataFrame with cumulative values
    st.write("Combined DataFrame:")
    columns_to_exclude = ['Distributed_Probability_COF', 'Distributed_Probability_PREPAYMENT', 'COF_EVENT_LABEL', 'PREPAYMENT_EVENT_LABEL']
    columns_to_display = [col for col in combined_df.columns if col not in columns_to_exclude]
    st.dataframe(combined_df[columns_to_display])

    # Store the combined DataFrame globally
    store_processed_data(combined_df, "Combined_COFPREPAYMENT")

    # Send the DataFrame to the next page
    st.session_state.final_combined_df = combined_df
    st.write("DataFrame has been sent to the next page.")

def map_credit_risk_params(credit_score, delinquency_count):
    """
    Determines skew weights based on risk profile.
    Args:
        credit_score (float): CREDIT_SCORE_AVG_CALC.
        delinquency_count (float): DELINQ_CNT_30_DAY_TOTAL.
    Returns:
        Tuple: (start_wt, middle_wt, end_wt)
    """
    # Normalize credit score: 300–850 mapped to 0 (risky) to 1 (safe)
    score_risk = 1 - np.clip((credit_score - 300) / (850 - 300), 0, 1)
    
    # Normalize delinquency count: assume 0–10+ range
    delinquency_risk = np.clip(delinquency_count / 10, 0, 1)

    # Blend both risks
    combined_risk = 0.6 * score_risk + 0.4 * delinquency_risk

    # Map to weights: high risk → early peak
    start_wt = 1.0 + combined_risk * 1.5    # up to 2.5x
    middle_wt = 1.0 - combined_risk * 0.5   # down to 0.5x
    end_wt = 1.0 - combined_risk * 1.0      # down to 0x

    return (start_wt, middle_wt, end_wt)

if __name__ == "__main__":
    main()
    # Ensure combined_df is stored globally
    if combined_df is not None:
        print("Combined DataFrame is ready and stored globally.")

