import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import os
# import shutil # Not used in the provided snippet after clearing function was simplified
# import uuid # No longer needed for historical_insights keys

# Path to the default data directory
DEFAULT_DATA_DIR = "default_data" # Not actively used in this script, but defined
DATA_REGISTRY_DIR = "data_registry"

# Clear the data_registry directory at the start of a new session
def clear_data_registry():
    if os.path.exists(DATA_REGISTRY_DIR):
        for root, dirs, files in os.walk(DATA_REGISTRY_DIR, topdown=False):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    os.unlink(file_path)
                except Exception as e:
                    st.error(f"Failed to delete file {file_path}: {e}")
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    os.rmdir(dir_path)
                except Exception as e:
                    st.error(f"Failed to delete directory {dir_path}: {e}")
        try:
            os.rmdir(DATA_REGISTRY_DIR)
        except Exception as e:
            st.error(f"Failed to delete directory {DATA_REGISTRY_DIR}: {e}")
    
    # Ensure the directory exists after attempting to clear it
    os.makedirs(DATA_REGISTRY_DIR, exist_ok=True)


# Clear the `data_registry` directory only once per session
if "data_registry_cleared" not in st.session_state:
    clear_data_registry()
    st.session_state["data_registry_cleared"] = True

# --- Helper Functions ---
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

    data.columns = data.columns.str.strip()
    # Optional: Normalize specific important column names if variations are common
    # e.g., if 'loan_status' in data.columns and 'Loan_Status' not in data.columns:
    #     data = data.rename(columns={'loan_status': 'Loan_Status'})

    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
        if data['Timestamp'].isnull().all(): # Check if all values are NaT after coercion
            st.error("The 'Timestamp' column could not be converted to datetime for any rows. Please check your data.")
            # Optionally, you could allow proceeding without time-based features or return None
            # For now, we'll mark it as problematic but proceed, historical features will be limited.
            # To be stricter, you could return None here.
    else:
        st.error("The dataset does not contain a 'Timestamp' column. This column is essential for historical insights.")
        # Return None if timestamp is critical and missing.
        # Depending on strictness, you might allow proceeding if other non-time-based insights are valuable.
        # For this dashboard, it seems critical.
        return None


    if 'Customer_ID' in data.columns:
        data['Customer_ID'] = data['Customer_ID'].astype(str)
    else:
        st.warning("The dataset does not contain a 'Customer_ID' column. Some insights may be limited.")

    return data

def filter_data_by_date(df, start_date, end_date):
    if 'Timestamp' not in df.columns or df['Timestamp'].isnull().all(): # Added check for all null Timestamps
        st.error("Cannot filter by date: 'Timestamp' column is missing or contains no valid date values.")
        return pd.DataFrame() 

    # Ensure start_date and end_date are datetime objects
    start_dt = pd.to_datetime(start_date)
    # Adjust end_date to include the entire day
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


    mask = (df['Timestamp'] >= start_dt) & (df['Timestamp'] <= end_dt)
    return df.loc[mask]

def calculate_change(current, previous):
    if previous == 0:
        if current == 0: return 0.0 # No change if both are zero
        return 0.0 # Or a large number like 100.0 or np.inf if you prefer to show growth from zero
    return round(((current - previous) / previous) * 100, 2)

def today_rates(df):
    if 'Loan_Status' not in df.columns:
        st.warning("Information regarding loan status is not available: 'Loan_Status' column is missing.")
        return 0.0, 0.0
    if 'Customer_ID' not in df.columns:
        st.warning("Skipping Today's Loan Rates: 'Customer_ID' column not found in the data.")
        return 0.0, 0.0
    if df.empty:
        return 0.0, 0.0

    df_cleaned = df.dropna(subset=['Timestamp'])
    df_cleaned = df_cleaned[df_cleaned['Customer_ID'] != '0'] # Assuming '0' is an invalid/placeholder Customer_ID
    
    if df_cleaned['Loan_Status'].isnull().all():
        st.warning("Information regarding loan status is not available: 'Loan_Status' contains only missing values.")
        return 0.0, 0.0

    latest_status_per_customer = df_cleaned.dropna(subset=['Loan_Status']).sort_values(by='Timestamp', ascending=True).groupby('Customer_ID')['Loan_Status'].last().reset_index()

    total_unique_customers_with_status = len(latest_status_per_customer)
    if total_unique_customers_with_status == 0:
        return 0.0, 0.0

    latest_status_per_customer['Loan_Status'] = pd.to_numeric(latest_status_per_customer['Loan_Status'], errors='coerce')
    latest_status_per_customer.dropna(subset=['Loan_Status'], inplace=True) # Remove rows where Loan_Status became NaN after coercion

    # Recalculate total after coercing and dropping NaNs from Loan_Status
    # This ensures the denominator is based on customers with valid, numeric loan statuses
    valid_status_customers = latest_status_per_customer['Customer_ID'].nunique()
    if valid_status_customers == 0:
        return 0.0, 0.0

    approved_customers = latest_status_per_customer[latest_status_per_customer['Loan_Status'] == 1].shape[0]
    
    approved_pct = (approved_customers / valid_status_customers) * 100
    # Assuming Loan_Status is 1 for approved, 0 for rejected. Other values are ignored by the '==1' condition.
    # Rejected percentage should be based on valid statuses that are not 1.
    rejected_customers = latest_status_per_customer[latest_status_per_customer['Loan_Status'] == 0].shape[0]
    # If other statuses exist (e.g., pending), this calculation might need adjustment
    # For now, if only 0 and 1 are valid, then rejected_pct = 100 - approved_pct is fine relative to (approved+rejected)
    # However, to be precise against `valid_status_customers`:
    if valid_status_customers > 0:
        rejected_pct = (rejected_customers / valid_status_customers) * 100
    else:
        rejected_pct = 0.0

    # If you want rejected_pct to be simply 100 - approved_pct from the valid statuses that are either 0 or 1:
    # total_binary_outcomes = approved_customers + rejected_customers
    # if total_binary_outcomes > 0:
    #     approved_pct = (approved_customers / total_binary_outcomes) * 100
    #     rejected_pct = (rejected_customers / total_binary_outcomes) * 100
    # else:
    #     approved_pct = 0.0
    #     rejected_pct = 0.0
    
    return approved_pct, rejected_pct


def average_approval_rate(df):
    if 'Loan_Status' not in df.columns:
        st.warning("Information regarding loan status is not available: 'Loan_Status' column is missing.")
        return 0.0, 0.0
    if df.empty or 'Customer_ID' not in df.columns:
        st.warning("Skipping Average Approval Rate: DataFrame is empty or 'Customer_ID' column not found.")
        return 0.0, 0.0

    df_cleaned = df.dropna(subset=['Timestamp'])
    df_cleaned = df_cleaned[df_cleaned['Customer_ID'] != '0']

    if df_cleaned['Loan_Status'].isnull().all():
        st.warning("Information regarding loan status is not available: 'Loan_Status' contains only missing values for average rate calculation.")
        return 0.0, 0.0
    
    df_cleaned = df_cleaned.dropna(subset=['Loan_Status'])
    if df_cleaned.empty:
        return 0.0, 0.0

    # Ensure Loan_Status is numeric before grouping
    df_cleaned['Loan_Status_Numeric'] = pd.to_numeric(df_cleaned['Loan_Status'], errors='coerce')
    df_cleaned.dropna(subset=['Loan_Status_Numeric'], inplace=True)

    if df_cleaned.empty: # Check again after numeric conversion and dropna
        return 0.0, 0.0

    grouped_by_date_customer = df_cleaned.sort_values(by='Timestamp', ascending=True).groupby([df_cleaned['Timestamp'].dt.date, 'Customer_ID'])['Loan_Status_Numeric'].last()
    
    # Now, aggregate daily rates
    daily_summary = grouped_by_date_customer.groupby(level=0).agg( # level=0 is the date
        total_applications=lambda x: x.count(),
        approved_applications=lambda x: (x == 1).sum()
    ).reset_index()

    daily_summary = daily_summary[daily_summary['total_applications'] > 0]
    if daily_summary.empty:
        return 0.0, 0.0

    daily_summary['approved_pct'] = daily_summary['approved_applications'] / daily_summary['total_applications'] * 100
    # Assuming only 0 and 1 are valid numeric statuses for rejection/approval
    daily_summary['rejected_pct'] = 100 - daily_summary['approved_pct'] 

    return daily_summary['approved_pct'].mean(), daily_summary['rejected_pct'].mean()

# --- Display Functions ---
def display_insights(data, dataset_name):
    st.header(f"{dataset_name} Data Feature Summary")
    with st.expander(f"View {dataset_name} Data Feature Overview"):
        # Exclude Timestamp for general feature overview if it exists
        cols_to_drop_overview = [col for col in data.columns if 'Timestamp' in col]
        data_no_timestamp = data.drop(columns=cols_to_drop_overview, errors='ignore')


        def get_most_repeating_value(series):
            try:
                val_counts = series.value_counts(dropna=True)
                if not val_counts.empty:
                    most_common = val_counts.idxmax()
                    return f"{most_common} ({val_counts.max()})"
            except Exception: # Catch a broad exception if idxmax fails or other issues
                return ''
            return ''

        summary_list = []
        for col in data_no_timestamp.columns:
            col_summary = {
                "Feature": col,
                "Data Type": data_no_timestamp[col].dtype,
                "% Missing": data_no_timestamp[col].isnull().mean() * 100,
                "# Distinct Values": data_no_timestamp[col].nunique(),
                "Most Repeating Value": get_most_repeating_value(data_no_timestamp[col])
            }
            if pd.api.types.is_numeric_dtype(data_no_timestamp[col].dtype):
                col_summary["Min"] = data_no_timestamp[col].min()
                col_summary["Max"] = data_no_timestamp[col].max()
                col_summary["Avg"] = data_no_timestamp[col].mean()
            else:
                col_summary["Min"] = ''
                col_summary["Max"] = ''
                col_summary["Avg"] = ''
            summary_list.append(col_summary)
        
        data_summary = pd.DataFrame(summary_list)
        st.dataframe(data_summary, height=400)

    st.subheader(f"Yesterday’s Loan Insights & Historical Comparison ({dataset_name} Data)")
    today = datetime.now().date()
    yesterday_date_obj = today - timedelta(days=1)

    # Filter data for yesterday based on Timestamp
    # Ensure data['Timestamp'] is datetime before .dt accessor
    if 'Timestamp' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Timestamp']):
        yesterday_data = data[data['Timestamp'].dt.date == yesterday_date_obj]
    else:
        yesterday_data = pd.DataFrame() # Empty if no valid timestamp
        st.warning(f"Cannot calculate yesterday's insights for {dataset_name} due to missing or invalid Timestamp column.")


    with st.container():
        col1, col2, col3 = st.columns(3)

        total_applications_yesterday = 0
        if 'Customer_ID' in yesterday_data.columns and not yesterday_data.empty:
            total_applications_yesterday = yesterday_data['Customer_ID'].nunique()
        elif not yesterday_data.empty: # Fallback if Customer_ID is missing but data exists
            total_applications_yesterday = len(yesterday_data)
        
        col1.metric("Total Unique Loan Applications Yesterday", total_applications_yesterday)
        
        # Key for the selectbox should be unique and stable
        comparison_period_key = f"comparison_period_select_{dataset_name}"
        comparison_period = st.selectbox(f"Compare Approved/Rejected % Against ({dataset_name} Data)", 
                                         ['Yesterday', 'Last Week', 'Last Month', 'Last Year'],
                                         key=comparison_period_key)

        compare_data_start = None
        compare_data_end = today - timedelta(days=2) # Common end for comparison periods

        if comparison_period == 'Yesterday': # Comparing yesterday with day before yesterday
            compare_data_start = today - timedelta(days=2)
        elif comparison_period == 'Last Week': # Last week, excluding yesterday
            compare_data_start = today - timedelta(days=8)
        elif comparison_period == 'Last Month': # Last month, excluding yesterday
            compare_data_start = today - timedelta(days=31) 
        else: # Last Year
            compare_data_start = today - timedelta(days=366)
        
        compare_data = pd.DataFrame() # Initialize as empty
        if compare_data_start and compare_data_end >= compare_data_start:
             compare_data = filter_data_by_date(data, compare_data_start, compare_data_end)
        else:
            st.warning(f"Invalid comparison period generated for {dataset_name}.")


        current_approved_pct, current_rejected_pct = today_rates(yesterday_data)
        previous_approved_pct, previous_rejected_pct = average_approval_rate(compare_data)

        approved_change = calculate_change(current_approved_pct, previous_approved_pct)
        rejected_change = calculate_change(current_rejected_pct, previous_rejected_pct)

        col2.metric(
            label="Approved Loan % (Yesterday)",
            value=f"{current_approved_pct:.2f}%",
            delta=f"vs {previous_approved_pct:.2f}% ({comparison_period}, {approved_change:+.2f}%)",
            delta_color="normal" if approved_change >= 0 else "inverse"
        )

        col3.metric(
            label="Rejected Loan % (Yesterday)",
            value=f"{current_rejected_pct:.2f}%",
            delta=f"vs {previous_rejected_pct:.2f}% ({comparison_period}, {rejected_change:+.2f}%)",
            # For rejected, an increase is "bad" (inverse), a decrease is "good" (normal)
            delta_color="inverse" if rejected_change >= 0 else "normal" 
        )

    display_historical_insights(data, dataset_name)

def display_bureau_insights(data, dataset_name):
    st.header(f"{dataset_name} Data Feature Summary")
    with st.expander(f"View {dataset_name} Data Feature Overview"):
        cols_to_drop_for_overview = [col for col in ['Timestamp'] if col in data.columns]
        data_for_summary = data.drop(columns=cols_to_drop_for_overview, errors='ignore')

        data_summary_df = pd.DataFrame({
            "Feature": data_for_summary.columns,
            "Data Type": data_for_summary.dtypes.values,
            "% Missing": data_for_summary.isnull().mean().values * 100,
            "# Distinct Values": data_for_summary.nunique().values,
        })
        st.dataframe(data_summary_df, height=400)

    st.subheader(f"Loan Insights & Metrics ({dataset_name} Data)")
    col1, col2, col3_metrics = st.columns([1, 1, 2]) # Renamed col3 to avoid conflict
    
    # Stable keys for date inputs
    bureau_start_date_key = f"bureau_start_date_{dataset_name}"
    bureau_end_date_key = f"bureau_end_date_{dataset_name}"

    with col1:
        start_date = st.date_input(f"Start Date ({dataset_name} Data)", datetime.now().date() - timedelta(days=30), key=bureau_start_date_key)
    with col2:
        end_date = st.date_input(f"End Date ({dataset_name} Data)", datetime.now().date(), key=bureau_end_date_key)

    filtered_data = filter_data_by_date(data, start_date, end_date)

    if not filtered_data.empty:
        required_bureau_cols = ['DEROG_TRDLN_TOTAL', 'COLLECTIONS_CUR_BAL_TOTAL', 'CREDIT_SCORE_AVG_CALC', 'DELINQ_CNT_30_DAY_TOTAL', 'DEROG_CUR_BAL_TOTAL']
        
        missing_cols = [col for col in required_bureau_cols if col not in filtered_data.columns]
        if not missing_cols:
            # Convert relevant columns to numeric, coercing errors
            for col in required_bureau_cols:
                if col in filtered_data.columns: # Check again, as some might be optional in a broader sense
                    filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
            
            # Define conditions for high-risk customers, handling potential NaNs from coercion
            # NaNs in boolean conditions usually evaluate to False, so they won't be included
            high_risk_condition = (
                (filtered_data['DEROG_TRDLN_TOTAL'].fillna(0) > 1) &
                (filtered_data['COLLECTIONS_CUR_BAL_TOTAL'].fillna(0) > 0) &
                (filtered_data['CREDIT_SCORE_AVG_CALC'].fillna(1000) < 550) & # fillna with value outside typical risk range
                (filtered_data['DELINQ_CNT_30_DAY_TOTAL'].fillna(0) > 0)
            )
            high_risk_customers = filtered_data[high_risk_condition]
            
            total_customers_in_filter = len(filtered_data)
            high_risk_pct = (len(high_risk_customers) / total_customers_in_filter) * 100 if total_customers_in_filter > 0 else 0.0
            
            total_unpaid_balance = filtered_data['DEROG_CUR_BAL_TOTAL'].sum() # sum() handles NaNs by default (treats as 0)
        
            with col3_metrics:
                st.metric(
                    label="High-Risk Customers (%)",
                    value=f"{high_risk_pct:.2f}%"
                )
                st.metric(
                    label="Total Unpaid Balance on Derogatory Accounts",
                    value=f"£{total_unpaid_balance / 1_000_000:.1f}M"
                )
        else:
            with col3_metrics:
                st.warning(f"Missing one or more required columns for Bureau insights: {', '.join(missing_cols)}. Please check your data.")

        display_historical_insights(data, dataset_name) # Pass original data for full historical view
    else:
        st.info(f"No data available for the selected date range in Bureau insights for {dataset_name}.")


def display_historical_insights(data, dataset_name):
    st.subheader(f"Historical Insights ({dataset_name} Data)")

    base_key = f"hist_insights_{dataset_name}" # Base for key uniqueness

    hist_start_date_key = f"{base_key}_start_date"
    hist_end_date_key = f"{base_key}_end_date"
    chart_type_widget_key = f"{base_key}_chart_type_widget"
    selected_feature_widget_key = f"{base_key}_selected_feature_widget"
    previous_chart_type_session_key = f"{base_key}_previous_chart_type_val" # For tracking chart_type changes

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Select Date Range")
        # Default to last 30 days if data allows, or full range
        min_date = data['Timestamp'].min().date() if 'Timestamp' in data.columns and not data['Timestamp'].dropna().empty else datetime.now().date() - timedelta(days=30)
        max_date = data['Timestamp'].max().date() if 'Timestamp' in data.columns and not data['Timestamp'].dropna().empty else datetime.now().date()
        
        default_start_date = max_date - timedelta(days=29) # Ensure default start is not before min_date
        if default_start_date < min_date: default_start_date = min_date


        hist_start_date = st.date_input(
            f"Start Date",
            default_start_date,
            min_value=min_date,
            max_value=max_date,
            key=hist_start_date_key
        )
        hist_end_date = st.date_input(
            f"End Date",
            max_date,
            min_value=min_date,
            max_value=max_date,
            key=hist_end_date_key
        )

    historical_data = filter_data_by_date(data, hist_start_date, hist_end_date)

    if historical_data.empty:
        st.warning("No data available for the selected date range. Please adjust the dates.")
        return

    with col2:
        st.markdown("### Select Feature Type and Feature")
        
        chart_type = st.selectbox(
            f"Choose Feature Type",
            ["Categorical", "Continuous"],
            key=chart_type_widget_key
        )

        features = []
        base_features = [col for col in data.columns if col not in ['Timestamp', 'Customer_ID']]

        if chart_type == "Categorical":
            features = [col for col in base_features if historical_data[col].nunique(dropna=False) <= 10 and historical_data[col].nunique(dropna=False) > 0] # Adjusted threshold, ensure some unique values
        elif chart_type == "Continuous":
            features = [col for col in base_features if pd.api.types.is_numeric_dtype(historical_data[col]) and historical_data[col].nunique(dropna=False) > 10]


        if not features:
            st.warning(f"No **{chart_type.lower()}** features found in the filtered data for **{dataset_name}** based on the current criteria (excluding 'Timestamp' and 'Customer_ID'). Adjust date range or check data.")
            return

        # Manage selected_feature state based on chart_type changes
        current_widget_selected_feature = st.session_state.get(selected_feature_widget_key)
        
        index_for_selectbox = 0
        if st.session_state.get(previous_chart_type_session_key) != chart_type or \
           current_widget_selected_feature not in features:
            if features: # Should always be true due to the check above
                st.session_state[selected_feature_widget_key] = features[0] 
                index_for_selectbox = 0
        elif current_widget_selected_feature in features:
            index_for_selectbox = features.index(current_widget_selected_feature)
        # else: first run or some other edge case, default to 0. If features[0] exists.
        # if features, this ensures index is valid.

        st.session_state[previous_chart_type_session_key] = chart_type
            
        selected_feature = st.selectbox(
            f"Select a Feature",
            features,
            index=index_for_selectbox,
            key=selected_feature_widget_key
        )

    st.markdown("---")
    if not selected_feature: # Should not happen if features list is populated
        st.warning("No feature selected.")
        return

    st.markdown(f"## {selected_feature} Analysis")
    chart_col, stats_col = st.columns([3, 2])

    if chart_type == "Continuous":
        if selected_feature not in historical_data.columns: # Should be caught by feature list generation
            chart_col.warning(f"The selected continuous feature '{selected_feature}' is not present.")
            return
        
        feature_series = historical_data[selected_feature].dropna() # Drop NaNs for plotting and stats
        if feature_series.empty:
            chart_col.warning(f"The selected continuous feature '{selected_feature}' contains only missing values or no data in the selected date range after dropping NaNs.")
            return

        fig = px.line(historical_data, x='Timestamp', y=selected_feature, # Plot with NaNs will show gaps
                      title=f"Trend of **{selected_feature}** over Time ({dataset_name} Data)",
                      labels={'Timestamp': 'Date', selected_feature: selected_feature})
        chart_col.plotly_chart(fig, use_container_width=True)

        if pd.api.types.is_numeric_dtype(historical_data[selected_feature]): # Redundant check if logic for features list is correct
            stats = feature_series.describe(percentiles=[.25, .5, .75])
            null_pct_original = historical_data[selected_feature].isnull().mean() * 100

            stats_data = {
                "Statistic": ["Minimum", "Maximum", "Mean", "Missing Values (%)", "25th Percentile", "50th Percentile (Median)", "75th Percentile"],
                "Value": [
                    f"{stats.get('min', 'N/A'):.2f}" if pd.notna(stats.get('min')) else "N/A",
                    f"{stats.get('max', 'N/A'):.2f}" if pd.notna(stats.get('max')) else "N/A",
                    f"{stats.get('mean', 'N/A'):.2f}" if pd.notna(stats.get('mean')) else "N/A",
                    f"{null_pct_original:.2f}%",
                    f"{stats.get('25%', 'N/A'):.2f}" if pd.notna(stats.get('25%')) else "N/A",
                    f"{stats.get('50%', 'N/A'):.2f}" if pd.notna(stats.get('50%')) else "N/A",
                    f"{stats.get('75%', 'N/A'):.2f}" if pd.notna(stats.get('75%')) else "N/A"
                ]
            }
            stats_table = pd.DataFrame(stats_data)

            with stats_col:
                st.markdown("### Key Statistics")
                st.table(stats_table)
        else: # Should not be reached if feature selection logic is correct
            with stats_col:
                st.info(f"Statistics are not applicable as '{selected_feature}' is not numeric.")

    else: # Categorical
        if selected_feature not in historical_data.columns:
            chart_col.warning(f"The selected categorical feature '{selected_feature}' is not present.")
            return

        # For categorical, show counts including NaNs as a category if desired
        # dropna=False in value_counts includes NaNs.
        counts = historical_data[selected_feature].value_counts(normalize=True, dropna=False).nlargest(10) * 100
        if counts.empty:
            chart_col.warning(f"No data to display for '{selected_feature}' in the selected date range.")
            return

        fig = px.bar(
            x=counts.index.astype(str), 
            y=counts.values,
            labels={'x': selected_feature, 'y': 'Percentage'},
            title=f"Top Frequencies for **{selected_feature}** ({dataset_name} Data)"
        )
        chart_col.plotly_chart(fig, use_container_width=True)

        counts_table_df = pd.DataFrame({
            "Category": counts.index.astype(str),
            "Percentage": [f"{value:.2f}%" for value in counts.values]
        })

        with stats_col:
            st.markdown("### Top Frequencies")
            st.table(counts_table_df)

# --- Main App ---
st.sidebar.header("Upload Files")
on_us_file = st.sidebar.file_uploader("Upload On-Us File", type=["csv", "xlsx"], key="on_us_uploader")
bureau_file = st.sidebar.file_uploader("Upload Bureau File", type=["csv", "xlsx"], key="bureau_uploader")
installments_file = st.sidebar.file_uploader("Upload Installments File", type=["csv", "xlsx"], key="installments_uploader")

if on_us_file is not None:
    on_us_data = load_data(on_us_file)
    if on_us_data is not None:
        if 'Timestamp' in on_us_data.columns and not on_us_data['Timestamp'].isnull().all():
            display_insights(on_us_data, "On-Us")
            save_dir = DATA_REGISTRY_DIR # Use defined constant
            # os.makedirs(save_dir, exist_ok=True) # Ensured by clear_data_registry
            data_path = os.path.join(save_dir, "on_us_data.parquet")
            try:
                on_us_data.to_parquet(data_path, index=False)
                st.session_state["on_us_data_path"] = data_path
                st.success("On-Us data processed and saved to data_registry.")
            except Exception as e:
                st.error(f"Failed to save On-Us data as parquet: {e}")
        else:
            st.error("On-Us data loaded but 'Timestamp' column is missing, invalid, or empty. Cannot proceed with insights.")


if bureau_file is not None:
    bureau_data = load_data(bureau_file)
    if bureau_data is not None:
        if 'Timestamp' in bureau_data.columns and not bureau_data['Timestamp'].isnull().all():
            display_bureau_insights(bureau_data, "Bureau")
            save_dir = DATA_REGISTRY_DIR
            data_path = os.path.join(save_dir, "bureau_data.parquet")
            try:
                bureau_data.to_parquet(data_path, index=False)
                st.session_state["bureau_data_path"] = data_path
                st.success("Bureau data processed and saved to data_registry.")
            except Exception as e:
                st.error(f"Failed to save Bureau data as parquet: {e}")
        else:
            st.error("Bureau data loaded but 'Timestamp' column is missing, invalid, or empty. Cannot proceed with insights.")


if installments_file is not None:
    installments_data = load_data(installments_file)
    if installments_data is not None:
        if 'Timestamp' in installments_data.columns and not installments_data['Timestamp'].isnull().all():
            # Assuming installments might also use general insights or need its own display function
            display_insights(installments_data, "Installments") 
            save_dir = DATA_REGISTRY_DIR
            data_path = os.path.join(save_dir, "installments_data.parquet")
            try:
                installments_data.to_parquet(data_path, index=False)
                st.session_state["installments_data_path"] = data_path
                st.success("Installments data processed and saved to data_registry.")
            except Exception as e:
                st.error(f"Failed to save Installments data as parquet: {e}")
        else:
            st.error("Installments data loaded but 'Timestamp' column is missing, invalid, or empty. Cannot proceed with insights.")