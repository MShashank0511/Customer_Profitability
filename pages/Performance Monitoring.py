import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error, roc_curve, roc_auc_score
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend import process_models_from_session 
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
 # <-- Use this function
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
                                           credit_col='CREDIT_SCORE_AVG_CALC', delinq_col='DELINQ_CNT_30_DAY_TOTAL'):
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

# st.set_page_config(layout="wide")
st.title("Comprehensive Model Monitoring: Profitability, Charge-Off, Prepayment")

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
    Adds the OPB, CREDIT_SCORE_AVG_CALC, and DELINQ_CNT_30_DAY_TOTAL columns to the DataFrame if they are not already present.
    The values are mapped using the Timestamp_x column from loan_data.csv.

    Args:
        df (pd.DataFrame): The DataFrame to which the columns will be added.
        loan_data_path (str): Path to the loan_data.csv file.

    Returns:
        pd.DataFrame: The updated DataFrame with the required columns.
    """
    try:
        # Load loan_data.csv
        loan_data = pd.read_csv(loan_data_path)

        # Ensure Timestamp_x column exists in both DataFrames
        if 'Timestamp_x' in df.columns and 'Timestamp_x' in loan_data.columns:
            # Convert Timestamp_x columns to datetime for alignment
            loan_data['Timestamp_x'] = pd.to_datetime(loan_data['Timestamp_x'], errors='coerce')
            df['Timestamp_x'] = pd.to_datetime(df['Timestamp_x'], errors='coerce')

            # Map OPB values if not present
            if 'OPB' not in df.columns:
                df = df.merge(loan_data[['Timestamp_x', 'OPB']], on='Timestamp_x', how='left')
                
            
            # Map CREDIT_SCORE_AVG_CALC values if not present
            if 'CREDIT_SCORE_AVG_CALC' not in df.columns:
                df = df.merge(loan_data[['Timestamp_x', 'CREDIT_SCORE_AVG_CALC']], on='Timestamp_x', how='left')
                
            
            # Map DELINQ_CNT_30_DAY_TOTAL values if not present
            if 'DELINQ_CNT_30_DAY_TOTAL' not in df.columns:
                df = df.merge(loan_data[['Timestamp_x', 'DELINQ_CNT_30_DAY_TOTAL']], on='Timestamp_x', how='left')
                
            
    except FileNotFoundError:
        st.error(f"Error: The file '{loan_data_path}' was not found. Please ensure it exists.")
    except Exception as e:
        st.error(f"Error adding columns: {e}")

    return df

import numpy_financial as npf  # Ensure numpy_financial is installed

def add_loan_metrics_updated(df, APR=0.144, fee_rate=0.005):
    """
    Adds loan metrics to the DataFrame using amortization calculations.

    Args:
        df (pd.DataFrame): The input DataFrame.
        APR (float): Annual Percentage Rate (default is 14.4%).
        fee_rate (float): The fee rate to calculate fees.

    Returns:
        pd.DataFrame: The updated DataFrame with new loan metrics.
    """
    interest_rate = APR / 12  # Monthly interest rate

    # Calculate Outstanding Principal for existing rows
    df['Outstanding_Principal'] = df.apply(
        lambda row: row['OPB'] - np.sum(-npf.ppmt(interest_rate, np.arange(1, row['Month'] + 1), row['TERM_OF_LOAN'], row['OPB'])),
        axis=1
    )

    # Clip Outstanding Principal to ensure no negative values
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
    Overwrites existing columns with their cumulative values, except for 'Charge_Off_Bal' and 'Interest_Amount',
    which retain their actual values.

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

    # Overwrite the selected columns with their cumulative sums, except 'Charge_Off_Bal' and 'Interest_Amount'
    cumulative_cols = ['Charge_Off_Bal', 'Interest_Amount', 'Fees', 'Recovery_Amount']
    for col in cumulative_cols:
        if col not in ['Charge_Off_Bal', 'Interest_Amount']:  # Skip cumulative sum for these columns
            df[col] = df.groupby('LOAN_ID')[col].cumsum()

    return df



# --- Main App ---
def main():
    """
    Main function to run the Streamlit application.
    """
    global combined_df  # Ensure combined_df is globally accessible

    # --- Steps to Follow Section ---
    st.markdown("""
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 10px; background-color: #f9f9f9; color: black;">
        <h3 style="color: #4CAF50;">📊 Steps to Follow on This Page</h3>
        <ol>
            <li><b>Select a Model</b><br>
                Choose any one of the two models (as defined in the Model Development page) using the Model Selection section.</li>
            <li><b>Explore Evaluation Metrics</b><br>
                Once a model is selected, explore its evaluation metrics across different segments to understand its performance in detail.</li>
            <li><b>Repeat for All Models</b><br>
                After evaluating the first model, repeat the model selection and evaluation steps for the remaining two models.</li>
            <li><b>Review Profitability Metrics</b><br>
                Once all two models are selected, view the calculated metrics related to profitability predictions for the given dataset.</li>
            <li><b>Proceed to Results Page</b><br>
                After reviewing the evaluation outcomes, move to the Results page to explore profitability insights and simulation options.</li>
        </ol>
        <p style="color: red; font-weight: bold;">Note: Please select all two models in the Model Selection section to check the results generated for Profitability Calculation.</p>
    </div>
    """, unsafe_allow_html=True)
    # --- 1. Model Selection ---
    st.header("Step 1 : Select Model for Evaluation")

    # Attempt to retrieve processed dataframes
    try:
        processed_dataframes = process_models_from_session()  # <-- Use this function
    except Exception as e:
        st.warning(f"An issue occurred while processing data: {e}")
        processed_dataframes = None

    # Fallback to default CSV files if no data is available
    if processed_dataframes is None or not processed_dataframes:
        st.warning("No processed data available. Loading default data.")

        try:
            # Load default CSV files into DataFrames
            profitability_df = pd.read_csv("Linear_Regression_(Profitability_GBP)_preview.csv")
            cof_df = pd.read_csv("Logistic_Regression_(COF_EVENT_LABEL)_preview.csv")
            prepayment_df = pd.read_csv("Logistic_Regression_(PREPAYMENT_EVENT_LABEL)_preview.csv")

            # Assign names to the DataFrames for identification
            profitability_df.attrs['name'] = "Linear Regression (Profitability_GBP)"
            cof_df.attrs['name'] = "Logistic Regression (COF_EVENT_LABEL)"
            prepayment_df.attrs['name'] = "Logistic Regression (PREPAYMENT_EVENT_LABEL)"

            # Combine the DataFrames into a list
            processed_dataframes = [profitability_df, cof_df, prepayment_df]

        except FileNotFoundError as e:
            st.error(f"Error: Required default CSV file not found. {e}")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred while loading default data: {e}")
            st.stop()

    # Create a dictionary of available data sources
    # Build the mapping of available data sources
    available_data_sources = {
        df.attrs.get('name', f'Unnamed DataFrame {i}'): df 
        for i, df in enumerate(processed_dataframes)
    }

    if not available_data_sources:
        st.warning("No data available. Please ensure data has been processed.")
        st.stop()

    # Desired default option
    default_option = "Linear Regression (Profitability_GBP)"
    options = list(available_data_sources.keys())

    # Determine the index of the default option if it exists
    default_index = options.index(default_option) if default_option in options else 0

    # Show selectbox with default selection
    data_source_name = st.selectbox(
        "Selected Model (Target_Variable)",
        options=options,
        index=default_index
    )

    # Set the selected DataFrame
    selected_df = available_data_sources[data_source_name]
    st.session_state.data_source = data_source_name

    # --- Display Data Preview ---
    st.subheader("Data Preview")
    with st.expander("Click to view the Data for selected Model", expanded=False):
    
        st.write("Preview of the data for selected Model:")

        # Ensure Origination_Year is displayed correctly without commas
        if 'Origination_Year' in selected_df.columns:
            selected_df['Origination_Year'] = selected_df['Origination_Year'].astype(int)

        st.dataframe(selected_df.head())


    # --- Check for Missing Columns ---
    required_columns = ['OPB', 'TERM_OF_LOAN', 'Timestamp_x']
    missing_columns = [col for col in required_columns if col not in selected_df.columns]

    if missing_columns:
        st.warning(f"Warning: The selected model does not contain the required columns: {missing_columns}. Please select both models in the Data Source section to avoid errors.")
        return
    
    # --- 2. Data Preprocessing ---
    df_edited = selected_df.copy()
    df_edited = add_opb_column(df_edited)

    # Ensure `Origination_Year` exists
    if 'Origination_Year' not in df_edited.columns:
        if 'Timestamp_x' in df_edited.columns:
            df_edited['Origination_Year'] = pd.to_datetime(df_edited['Timestamp_x'], errors='coerce').dt.year
        else:
            st.error("The 'Origination_Year' column could not be created because 'Timestamp_x' is missing.")
            return

    # Convert TERM_OF_LOAN values to 60 or 84
    df_edited['TERM_OF_LOAN'] = df_edited['TERM_OF_LOAN'].apply(lambda x: 60 if x <= 60 else 84)

    # Display the preprocessed data
    st.subheader("Transformed Data")
    with st.expander("Click to view the preprocessed data", expanded=False):
        st.write("Preview of the transformed data:")
        st.dataframe(df_edited.head())
    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
    # --- 3. Explore Model Evaluation for Selected Segments ---
    st.header("Step 2 : Explore Model Evaluation for Selected Segments")

    # Extract unique values from the preprocessed data
    origination_years = sorted(df_edited['Origination_Year'].dropna().unique())
    term_of_loans = sorted(df_edited['TERM_OF_LOAN'].dropna().unique())

    # Create multiselect options for filtering Origination Year and TERM_OF_LOAN
    selected_years = st.multiselect("Filter by Origination Year", options=origination_years, default=origination_years)
    selected_terms = st.multiselect("Filter by TERM_OF_LOAN", options=term_of_loans, default=term_of_loans)

    # Add a multi-select filter for CREDIT_SCORE_AVG_CALC
    credit_score_ranges = {
        '660+': (660, float('inf')),
        '600-660': (600, 660),
        '<600': (float('-inf'), 600)
    }
    selected_credit_scores = st.multiselect(
        "Filter by Credit Score Range ",
        options=list(credit_score_ranges.keys()),
        default=list(credit_score_ranges.keys())
    )

    # Apply filters to the preprocessed data
    filtered_df = df_edited.copy()

    # Apply Origination Year filter
    if selected_years:
        filtered_df = filtered_df[filtered_df['Origination_Year'].isin(selected_years)]

    # Apply TERM_OF_LOAN filter
    if selected_terms:
        filtered_df = filtered_df[filtered_df['TERM_OF_LOAN'].isin(selected_terms)]

    # Apply CREDIT_SCORE_AVG_CALC filter
    if selected_credit_scores:
        credit_score_conditions = []
        for score_range in selected_credit_scores:
            lower, upper = credit_score_ranges[score_range]
            credit_score_conditions.append((filtered_df['CREDIT_SCORE_AVG_CALC'] >= lower) & (filtered_df['CREDIT_SCORE_AVG_CALC'] < upper))
        if credit_score_conditions:
            filtered_df = filtered_df[np.logical_or.reduce(credit_score_conditions)]

    # Display the filtered data
    

    # Determine the target variable
    target_variable_present = None
    if 'Profitability_GBP' in filtered_df.columns:
        target_variable_present = 'Profitability_GBP'
    elif 'COF_EVENT_LABEL' in filtered_df.columns:
        target_variable_present = 'COF_EVENT_LABEL'
    elif 'PREPAYMENT_EVENT_LABEL' in filtered_df.columns:
        target_variable_present = 'PREPAYMENT_EVENT_LABEL'

    # --- 4. Model Evaluation ---
    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
    st.header("Step 3 : Performance metrics and evaluation graphs based on selected filters.")

    if target_variable_present == 'Profitability_GBP':
        # Calculate evaluation metrics
        actual_values = filtered_df['Profitability_GBP']
        predicted_values = filtered_df['Predicted_Target']

        # Avoid division by zero by replacing zeros in actual values with a small number
        actual_values = actual_values.replace(0, np.finfo(float).eps)

        # Calculate metrics
        average_error = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100  # Average Error %
        mse = mean_squared_error(actual_values, predicted_values)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        mae = np.mean(np.abs(actual_values - predicted_values))  # Mean Absolute Error

        # Display metrics
        st.subheader("Evaluation Metrics for Profitability")
        st.write(f"**Average Error (%):** {average_error:.2f}%")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")

        # Group data by Origination_Year and calculate the sum for Predicted_Target and Profitability_GBP
        grouped_df = filtered_df.groupby('Origination_Year').agg({
            'Predicted_Target': 'sum',
            'Profitability_GBP': 'sum'
        }).reset_index()

        # Plot Predicted_Target and Profitability_GBP vs. Origination_Year
        st.subheader("Predicted Profitability vs. Actual Profitability (Grouped by Origination Year)")
        fig = go.Figure()

        # Add Predicted_Target line
        fig.add_trace(go.Scatter(
            x=grouped_df['Origination_Year'],
            y=grouped_df['Predicted_Target'],
            mode='lines+markers',
            name='Predicted Profitability'
        ))

        # Add Profitability_GBP line
        fig.add_trace(go.Scatter(
            x=grouped_df['Origination_Year'],
            y=grouped_df['Profitability_GBP'],
            mode='lines+markers',
            name='Actual Profitability'
        ))

        # Update layout
        fig.update_layout(
            title="Predicted Profitability vs. Actual Profitability (Grouped by Origination Year)",
            xaxis_title="Origination Year",
            yaxis_title="Total Profitability",
            template="plotly_white"
        )

        # Display the graph
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Add a button to show the ROC curve for other target variables
        if target_variable_present in ['COF_EVENT_LABEL', 'PREPAYMENT_EVENT_LABEL']:
                # Display ROC AUC Curve
            if target_variable_present in filtered_df.columns and 'Predicted_Probability' in filtered_df.columns:
                y_true = filtered_df[target_variable_present]
                y_pred_proba = filtered_df['Predicted_Probability']

                if len(y_true.unique()) <= 2:  # Ensure binary classification
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                    auc = roc_auc_score(y_true, y_pred_proba)

                    # Plot ROC Curve
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc:.2f})'))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash', color='gray')))
                    fig.update_layout(
                        title=f"ROC Curve for {target_variable_present} - {data_source_name}",
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(f"ROC AUC Score: {auc:.4f}")

        # Process and store combined_df for COF and Prepayment models
        if target_variable_present == 'COF_EVENT_LABEL':
            st.session_state.cof_event_df = split_and_distribute_with_normalization(filtered_df)
        elif target_variable_present == 'PREPAYMENT_EVENT_LABEL':
            st.session_state.prepayment_event_df = split_and_distribute_with_normalization(filtered_df)

        # --- Combine DataFrames for COF and Prepayment Models ---
        if 'cof_event_df' in st.session_state and 'prepayment_event_df' in st.session_state:
            cof_event_df = st.session_state.cof_event_df
            prepayment_event_df = st.session_state.prepayment_event_df

            # Merge COF and Prepayment DataFrames
            combined_df = pd.merge(
                cof_event_df[['Timestamp_x', 'Month', 'Distributed_Probability']],
                prepayment_event_df[['Timestamp_x', 'Month', 'Distributed_Probability']],
                on=['Timestamp_x', 'Month'],
                how='outer',
                suffixes=('_COF', '_PREPAYMENT')
            )

            # Merge with additional columns from df_edited
            combined_df = combined_df.merge(
                df_edited[['Timestamp_x', 'OPB', 'Origination_Year', 'TERM_OF_LOAN']].drop_duplicates(),
                on='Timestamp_x',
                how='left'
            )

            # Populate COF_EVENT_LABEL and PREPAYMENT_EVENT_LABEL
            combined_df['COF_EVENT_LABEL'] = combined_df['Distributed_Probability_COF'].fillna(0)
            combined_df['PREPAYMENT_EVENT_LABEL'] = combined_df['Distributed_Probability_PREPAYMENT'].fillna(0)

            # Calculate Total_Probability
            combined_df['Total_Probability'] = combined_df[['Distributed_Probability_COF', 'Distributed_Probability_PREPAYMENT']].sum(axis=1)

            # Add Loan Metrics
            combined_df = add_loan_metrics_updated(combined_df)

            # Apply Cumulative Columns
            combined_df = apply_cumulative_columns(combined_df)

            # Store the combined DataFrame globally
            store_processed_data(combined_df, "Combined_COFPREPAYMENT")

    # --- 5. Button to Show Results ---
    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
    # st.header("Step 4 : Results")

    # # Debugging: Check if cof_event_df and prepayment_event_df are present
    # if 'cof_event_df' not in st.session_state:
    #     st.warning("COF Event DataFrame is not available.")
    # if 'prepayment_event_df' not in st.session_state:
    #     st.warning("Prepayment Event DataFrame is not available.")

    # # Display the Show Results button if at least one of the models is processed
    # if 'cof_event_df' in st.session_state or 'prepayment_event_df' in st.session_state:
    #     if st.button("Show Results"):
    #         # Ensure the COF combined_df is available
    #         if 'cof_combined_df' not in st.session_state or st.session_state.cof_combined_df is None:
    #             # Attempt to retrieve the COF combined_df from the pickle file
    #             try:
    #                 with open("combined_df.pkl", "rb") as f:
    #                     st.session_state.cof_combined_df = pickle.load(f)
                        
    #             except FileNotFoundError:
    #                 st.error("The combined DataFrame for the COF model is not available. Please ensure the COF model is selected and processed.")
    #                 return

    #         # Use the COF combined_df for plotting
    #         combined_df_to_use = st.session_state.cof_combined_df

    #         # Check if combined_df_to_use is valid
    #         if combined_df_to_use is None:
    #             st.error("The combined DataFrame is not available. Please ensure the COF model is selected and processed.")
    #             return

    #         # Filter rows where OPB is not None
    #         filtered_combined_df = combined_df_to_use[combined_df_to_use['OPB'].notna()]

    #         # Group data by Month and calculate the mean for plotting
    #         grouped_df = filtered_combined_df.groupby('Month').agg({
    #             'Outstanding_Principal': 'mean',
    #             'Charge_Off_Bal': 'mean'
    #         }).reset_index()

    #         # Plot Outstanding_Principal vs. Month
    #         st.subheader("Outstanding Principal vs. Month")
    #         fig1 = go.Figure()
    #         fig1.add_trace(go.Scatter(
    #             x=grouped_df['Month'],
    #             y=grouped_df['Outstanding_Principal'],
    #             mode='lines+markers',
    #             name='Outstanding Principal'
    #         ))
    #         fig1.update_layout(
    #             title="Outstanding Principal vs. Month",
    #             xaxis_title="Month",
    #             yaxis_title="Outstanding Principal",
    #             template="plotly_white"
    #         )
    #         st.plotly_chart(fig1, use_container_width=True)

    #         # Plot Charge Off Amount vs. Month
    #         st.subheader("Charge Off Amount vs. Month")
    #         fig2 = go.Figure()
    #         fig2.add_trace(go.Scatter(
    #             x=grouped_df['Month'],
    #             y=grouped_df['Charge_Off_Bal'],
    #             mode='lines+markers',
    #             name='Charge Off Amount'
    #         ))
    #         fig2.update_layout(
    #             title="Charge Off Amount vs. Month",
    #             xaxis_title="Month",
    #             yaxis_title="Charge Off Amount",
    #             template="plotly_white"
    #         )
    #         st.plotly_chart(fig2, use_container_width=True)

    #         # Pass the COF combined_df to the next page
    #         st.session_state.final_combined_df = filtered_combined_df
    #         store_processed_data(filtered_combined_df, "Combined_COFPREPAYMENT")
    # else:
    #     st.warning("Please select all three models to generate the combined DataFrame.")

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

if st.button("Proceed to Results Page"):
        st.switch_page("pages/Results.py")