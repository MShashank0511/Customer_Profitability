import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import re
import uuid
import random
from datetime import datetime, timedelta
import json
import os
import model_states.feat_engg_backend as feat_engg_backend
import re # For parsing Gemini output
import google.generativeai as genai
from dotenv import load_dotenv
import shutil
import sys
import os
import datetime


try:
    from genai_recommend_features import (
        summarize_dataset_columns,
        get_recommended_features_gemini,
        parse_gemini_recommendations,
        apply_recommended_features,
        validate_code_snippet,
        sanitize_code_snippet,
        _dedent_code,
        extract_code_block,
        
        GEMINI_API_KEY_CONFIGURED
    )
except ImportError:
    st.error("Failed to import 'genai_recommend_features.py'. Make sure it's in the same directory or Python path.")
    # Stop execution if utils can't be imported
    GEMINI_API_KEY_CONFIGURED = False # Assume not configured
    st.stop()

if st.session_state.get("should_rerun", False):
    st.session_state["should_rerun"] = False  # Reset the flag
    st.rerun()

def feature_metadata_df(metadata_file="loan_feature_metadata.csv"):
    """Load feature metadata and return as a DataFrame."""
    try:
        metadata = pd.read_csv(metadata_file)
        if {'feature_name', 'simple_name', 'description'}.issubset(metadata.columns):
            return metadata
        else:
            st.error("The metadata file must contain 'feature_name', 'simple_name', and 'description' columns.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load feature metadata: {e}")
        return pd.DataFrame()

# Load metadata once
feature_metadata_df = feature_metadata_df("loan_feature_metadata.csv")
feature_desc_map = dict(zip(feature_metadata_df['feature_name'], feature_metadata_df['description']))

def make_json_serializable(obj):
    """Recursively convert pd.Timestamp, datetime, and date objects to ISO strings for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)
    elif isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        return obj.isoformat()
    else:
        return obj

def clear_model_states_folder():
    model_states_dir = "model_states"
    exclude_file = "feat_engg_backend.py"  # File to exclude from deletion

    if os.path.exists(model_states_dir):
        try:
            # Iterate through the contents of the folder
            for item in os.listdir(model_states_dir):
                item_path = os.path.join(model_states_dir, item)
                # Skip the excluded file
                if item == exclude_file:
                    continue
                # Remove directories
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                # Remove files
                elif os.path.isfile(item_path):
                    os.remove(item_path)

            print("Cleared model_states folder for a new session, excluding 'feat_engg_backend.py'.")
        except Exception as e:
            st.error(f"Error clearing model_states folder: {str(e)}")

# Call the function to clear the folder
clear_model_states_folder()

loan_level_data_path = st.session_state.get("loan_level_data_path")
 
bureau_data_path = st.session_state.get("bureau_data_path")

installments_data_path = st.session_state.get("installments_data_path")

print("ENTERED FEATURE ENGINEERING PAGE")

print(loan_level_data_path,bureau_data_path,installments_data_path)
print(f"Attempting to load Bureau Data from: {bureau_data_path}")
try:
    bureau_df = pd.read_parquet(bureau_data_path)
    print(f"Successfully loaded Bureau Data. Shape: {bureau_df.shape}")
except Exception as e:
    print(f"Error loading Bureau Data from '{bureau_data_path}': {e}")

print(f"Attempting to load Loan-Level Data from: {loan_level_data_path}")
try:
    loan_level_df = pd.read_parquet(loan_level_data_path)
    print(f"Successfully loaded Loan-Level Data. Shape: {loan_level_df.shape}")
except Exception as e:
    print(f"Error loading Loan-Level Data from '{loan_level_data_path}': {e}")

print(f"Attempting to load Installments Data from: {installments_data_path}")
try:
    installments_df = pd.read_parquet(installments_data_path)
    print(f"Successfully loaded Installments Data. Shape: {installments_df.shape}")
except Exception as e:
    print(f"Error loading Installments Data from '{installments_data_path}': {e}")


# def load_data_from_data_engineering(model_name):
#     """
#     Loads the 'on_us_data' parquet file saved by data_engineering.py.
#     If the parquet file is not found, loads 'loan_data.csv' as a fallback.

#     Returns:
#         pd.DataFrame: The loaded DataFrame, or None if an error occurs.
#     """
#     data_path = st.session_state.get("loan_level_data_path")  # Access from session_state

#     if data_path and os.path.exists(data_path):
#         print(st.session_state)
#         try:
#             df = pd.read_parquet(data_path)
#             st.success(f"Data successfully loaded from: {data_path}")
#             return df
#         except Exception as e:
#             st.error(f"An error occurred while loading parquet data: {e}. Loading loan_data.csv as fallback.")
#             try:
#                 df = pd.read_csv("loan_data.csv")
#                 st.success("Fallback: loan_data.csv loaded successfully.")
#                 return df
#             except Exception as e2:
#                 st.error(f"An error occurred while loading fallback CSV data: {e2}")
#                 return None
#     else:
#         st.warning("No parquet data path found or file does not exist. Loading loan_data.csv as fallback.")
#         try:
#             df = pd.read_csv("loan_data.csv")
#             st.success("Fallback: loan_data.csv loaded successfully.")
#             return df
#         except Exception as e:
#             st.error(f"An error occurred while loading fallback CSV data: {e}")
#             return None

# --- Model Definitions ---
MODEL_NAMES = ["Forecast Model", "Charge-Off Model", "Prepayment Model"]

def initialize_new_model_state(model_name):
    """Initialize a fresh state for a new model."""
    # Ensure bureau_df, loan_level_df, and installments_df are defined or passed.
    # For demonstration, let's assume they are globally available or passed as arguments.
    # In a real Streamlit app, these would typically come from user uploads or a data loading utility.
    try:
        # These dataframes should be available in the Streamlit session state
        # if they have been loaded previously by the user.
        # If not, provide empty dataframes to avoid errors.
        initial_loan_data = pd.read_csv("loan_data.csv") 
        try:
            initial_bureau_data = bureau_df.copy()
        except NameError:
            st.warning("Required data 'bureau_df' is not defined. Please upload the required data before proceeding.")
            initial_bureau_data = pd.DataFrame()
        try:
            initial_loan_level_data = loan_level_df.copy()
        except NameError:
            st.warning("Required data 'loan_level_df' is not defined. Please upload the required data before proceeding.")
            initial_loan_level_data = pd.DataFrame()

        try:
            initial_installments_data = installments_df.copy()
        except NameError:
            st.warning("Required data 'installments_df' is not defined. Please upload the required data before proceeding.")
            initial_installments_data = pd.DataFrame()
    except AttributeError:
        # Handle cases where session_state might not have these attributes yet
        initial_loan_data = pd.DataFrame()
        initial_bureau_data = pd.DataFrame()
        initial_loan_level_data = pd.DataFrame()
        initial_installments_data = pd.DataFrame()


    # Initialize raw_datasets and filtered_datasets dictionaries
    raw_datasets = {
        "Loan Data": initial_loan_data.copy(),
        "Bureau Data": initial_bureau_data.copy(),
        "Loan-Level Data": initial_loan_level_data.copy(),
        "Payment Data": initial_installments_data.copy(),
    }
    filtered_datasets = {}

    # Initialize main state for the model
    st.session_state[f"{model_name}_state"] = {
        "raw_datasets": raw_datasets,
        "filtered_datasets": filtered_datasets,
        "loan_data": raw_datasets["Loan Data"], # Point to the raw_datasets version
        "bureau_data": raw_datasets["Bureau Data"], # Point to the raw_datasets version
        "loan_level_data": raw_datasets["Loan-Level Data"], # Point to the raw_datasets version
        "installments_data": raw_datasets["Payment Data"], # Point to the raw_datasets version
        "show_popup1": False,
        "transform_blocks": [],
        "multi_transform_blocks": [],
        "final_transformed_features": pd.DataFrame(),
        "recommended_features": pd.DataFrame(),
        "final_dataset": pd.DataFrame(),
        "selected_features": [],
        "feature_checkboxes": {},
        "show_filter": False,
        "filtered_features": [],
        "filter_text": "",
        "merge_blocks": [{
            "left_table": "Bureau Data",
            "right_table": "Loan-Level Data",
            "how": "inner",
            "on": [],
            "left_on": [],
            "right_on": [],
            "merged_name": "",
        }],
        "merged_tables": {},
        "combined_dataset": None,
        "filter_blocks": [{
            "dataset": "Bureau Data",
            "feature": "",
            "operation": "Greater Than",
            "value": None,
            "output_name": ""
        }],
        "target_column": None,
        "target_feature": None,
        "final_dataset_json": None,
    }

    # Initialize other direct model-specific state variables
    st.session_state[f"{model_name}_operations_complete"] = {
        "merge": False,
        "recommend": False,
        "accept": False
    }
    st.session_state[f"{model_name}_show_filter_data"] = False
    st.session_state[f"{model_name}_show_merge"] = False
    st.session_state[f"{model_name}_single_transform_success"] = None
    st.session_state[f"{model_name}_multi_transform_success"] = None
    st.session_state[f"{model_name}_recommended_features"] = pd.DataFrame()
    st.session_state[f"{model_name}_final_dataset"] = pd.DataFrame()
    st.session_state[f"{model_name}_selected_features"] = []
    st.session_state[f"{model_name}_feature_checkboxes"] = {}
    st.session_state[f"{model_name}_filter_blocks"] = [{
        "dataset": "Bureau Data",
        "feature": "",
        "operation": "Greater Than",
        "value": None,
        "output_name": ""
    }]
    st.session_state[f"{model_name}_target_column"] = None
    st.session_state[f"{model_name}_target_feature"] = None
    st.session_state[f"{model_name}_final_dataset_json"] = None

    print(f"New state initialized for model: {model_name}")

def save_model_state(model_name):
    """
    Saves the complete state for a given model to a JSON file.
    Handles serialization of DataFrames and nested dictionaries containing DataFrames.
    """
    model_states_dir = "model_states"
    os.makedirs(model_states_dir, exist_ok=True)
    file_path = os.path.join(model_states_dir, f"{model_name}.json")

    state_to_save = {}

    # List all session state keys that belong to this model
    all_model_session_keys = [key for key in st.session_state.keys() if key.startswith(f"{model_name}_")]

    for session_key in all_model_session_keys:
        value = st.session_state[session_key]

        if session_key == f"{model_name}_state":
            serializable_main_state = {}
            for sub_key, sub_value in value.items():
                if sub_key == "raw_datasets":
                    # For raw_datasets, we want to preserve the actual user-uploaded data.
                    # DO NOT serialize them as they are supposed to be loaded initially by the user.
                    # We will simply store their structure, not their content.
                    # Or, more practically, we can just skip saving their content here
                    # as they are assumed to be loaded by the user at the start.
                    # However, to maintain the structure, we can store empty DataFrames for keys.
                    serializable_main_state[sub_key] = {
                        k: (v.to_json(orient='split', date_format='iso') if isinstance(v, pd.DataFrame) else v)
                        for k, v in sub_value.items()
                    }
                elif isinstance(sub_value, pd.DataFrame):
                    serializable_main_state[sub_key] = sub_value.to_json(orient='split', date_format='iso')
                elif isinstance(sub_value, dict):
                    serializable_nested_dict = {}
                    for nested_key, nested_df_value in sub_value.items():
                        if isinstance(nested_df_value, pd.DataFrame):
                            serializable_nested_dict[nested_key] = nested_df_value.to_json(orient='split', date_format='iso')
                        else:
                            serializable_nested_dict[nested_key] = nested_df_value
                    serializable_main_state[sub_key] = serializable_nested_dict
                else:
                    serializable_main_state[sub_key] = sub_value
            state_to_save[session_key] = serializable_main_state
        elif isinstance(value, pd.DataFrame):
            state_to_save[session_key] = value.to_json(orient='split', date_format='iso')
        else:
            state_to_save[session_key] = value

    # try:
    #     with open(file_path, "w") as f:
    #         json.dump(state_to_save, f, indent=4)
    #     print(f"State for model '{model_name}' saved to {file_path}")
    # except Exception as e:
    #     st.error(f"Error saving state for model '{model_name}': {e}")

    try:
        # Convert all non-serializable objects before saving
        serializable_state = make_json_serializable(state_to_save)
        with open(file_path, "w") as f:
            json.dump(serializable_state, f, indent=4)
        print(f"State for model '{model_name}' saved to {file_path}")
    except Exception as e:
        st.error(f"Error saving state for model '{model_name}': {e}")

def load_model_state(model_name):
    """
    Loads the complete state for a given model from a JSON file.
    Handles deserialization of DataFrames and nested dictionaries containing DataFrames.
    If no saved state is found, it initializes a new one.
    Crucially, for 'raw_datasets', it uses the already uploaded user data
    instead of re-loading from the saved JSON, which would be the default empty DataFrames.
    """
    model_states_dir = "model_states"
    file_path = os.path.join(model_states_dir, f"{model_name}.json")

    # Store the currently uploaded raw datasets before clearing session state
    # These are assumed to be present in st.session_state from initial user uploads
    current_raw_datasets = {
        "Loan Data": st.session_state.get("uploaded_loan_data", pd.DataFrame()),
        "Bureau Data": st.session_state.get("uploaded_bureau_data", pd.DataFrame()),
        "Loan-Level Data": st.session_state.get("uploaded_loan_level_data", pd.DataFrame()),
        "Installments Data": st.session_state.get("uploaded_installments_data", pd.DataFrame()),
    }

    # Clear all existing model-specific keys from session state
    for key in list(st.session_state.keys()):
        if key.startswith(f"{model_name}_"):
            del st.session_state[key]

    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                loaded_data = json.load(f)

            for session_key, value in loaded_data.items():
                if session_key == f"{model_name}_state":
                    deserialized_main_state = {}
                    for sub_key, sub_value in value.items():
                        if sub_key == "raw_datasets":
                            # Use the user-uploaded data for raw_datasets
                            deserialized_main_state[sub_key] = current_raw_datasets
                        elif isinstance(sub_value, str) and (
                            "df" in sub_key.lower() or "dataset" in sub_key.lower() or
                            sub_key in ["loan_data", "bureau_data", "loan_level_data", "installments_data",
                                        "filtered_data_loan", "filtered_data_bureau", "filtered_data_loan_level",
                                        "filtered_data_installments", "merged_dataframe", "selected_features_df",
                                        "final_transformed_features", "recommended_features", "final_dataset",
                                        "transformation_output_df"]
                        ):
                            try:
                                deserialized_main_state[sub_key] = pd.read_json(sub_value, orient='split') if sub_value else pd.DataFrame()
                            except Exception as inner_df_err:
                                print(f"Warning: Error deserializing DataFrame '{sub_key}' within main state: {inner_df_err}. Setting to empty DataFrame.")
                                deserialized_main_state[sub_key] = pd.DataFrame()
                        elif isinstance(sub_value, dict) and (sub_key in ["filtered_datasets", "merged_tables"]):
                            deserialized_nested_dict = {}
                            for nested_key, nested_json_df_str in sub_value.items():
                                try:
                                    deserialized_nested_dict[nested_key] = pd.read_json(nested_json_df_str, orient='split') if nested_json_df_str else pd.DataFrame()
                                except Exception as nested_df_err:
                                    print(f"Warning: Error deserializing nested DataFrame '{nested_key}' in '{sub_key}': {nested_df_err}. Setting to empty DataFrame.")
                                    deserialized_nested_dict[nested_key] = pd.DataFrame()
                            deserialized_main_state[sub_key] = deserialized_nested_dict
                        else:
                            deserialized_main_state[sub_key] = sub_value
                    st.session_state[session_key] = deserialized_main_state

                    # After loading, ensure individual DataFrame pointers refer to the raw_datasets
                    # This is crucial for consistency
                    if "raw_datasets" in st.session_state[session_key]:
                        st.session_state[session_key]["loan_data"] = st.session_state[session_key]["raw_datasets"].get("Loan Data", pd.DataFrame())
                        st.session_state[session_key]["bureau_data"] = st.session_state[session_key]["raw_datasets"].get("Bureau Data", pd.DataFrame())
                        st.session_state[session_key]["loan_level_data"] = st.session_state[session_key]["raw_datasets"].get("Loan-Level Data", pd.DataFrame())
                        st.session_state[session_key]["installments_data"] = st.session_state[session_key]["raw_datasets"].get("Installments Data", pd.DataFrame())

                elif isinstance(value, str) and (
                    "df" in session_key.lower() or "dataset" in session_key.lower() or
                    session_key in [f"{model_name}_recommended_features", f"{model_name}_final_dataset"]
                ):
                    try:
                        st.session_state[session_key] = pd.read_json(value, orient='split') if value else pd.DataFrame()
                    except Exception as direct_df_err:
                        print(f"Warning: Error deserializing direct DataFrame '{session_key}': {direct_df_err}. Setting to empty DataFrame.")
                        st.session_state[session_key] = pd.DataFrame()
                else:
                    st.session_state[session_key] = value

            print(f"State for model '{model_name}' loaded from {file_path}")
            return True
        except Exception as e:
            st.error(f"Error loading state for model '{model_name}': {e}. Initializing new state.")
            initialize_new_model_state(model_name)
            return False
    else:
        print(f"No saved state found for model '{model_name}'. Initializing new state.")
        initialize_new_model_state(model_name)
        return False


def add_new_model():
    """Add a new model (page) with fresh state."""
    try:
        # Save the current model's state before creating a new one
        current_model = st.session_state.active_model
        save_model_state(current_model)

        # Create new model name using the requested format
        new_model_name = f"Feature Transformation Page {len(st.session_state.models) + 1}"
        st.session_state.models.append(new_model_name)

        # Initialize fresh state for the new model
        initialize_new_model_state(new_model_name)

        # Switch to the new model
        st.session_state.active_model = new_model_name

        # Save the new model's initial state
        save_model_state(new_model_name)

        st.session_state["should_rerun"] = True
    except Exception as e:
        st.error(f"Error creating new model: {str(e)}")

def switch_model(model_name):
    """Switch to a different model (page), saving current state and loading the new model's state."""
    try:
        # Save the current model's state before switching
        current_model = st.session_state.active_model
        save_model_state(current_model)

        # Switch to the new model
        st.session_state.active_model = model_name

        # Try to load existing state from backend
        load_model_state(model_name)

        # If no saved state exists, initialize with default values (which will use uploaded data)
        if f"{model_name}_state" not in st.session_state:
            initialize_new_model_state(model_name)
            save_model_state(model_name)

        st.session_state["should_rerun"] = True
    except Exception as e:
        st.error(f"Error switching models: {str(e)}")
# Add caching decorators for expensive operations






# Add this callback function near the top with other callback functions
def show_merge_callback():
    """Callback to toggle the visibility of the merge section."""
    active_model = st.session_state.active_model
    model_state = st.session_state.get(f"{active_model}_state", {})
    # Toggle the show_merge state within the model_state
    model_state["show_merge"] = not model_state.get("show_merge", False)
    st.session_state[f"{active_model}_state"] = model_state # Update session state
    


def recommend_features_callback():
    try:
        # Get the active model and its state
        active_model = st.session_state.active_model
        model_state = st.session_state[f"{active_model}_state"]

        # Check if we have a combined dataset
        if "combined_dataset" in model_state and model_state["combined_dataset"] is not None:
            # Get a subset of features (15-20)
            all_features = model_state["combined_dataset"].columns.tolist()
            selected_features = all_features[:min(20, len(all_features))]

            # Create a DataFrame with feature descriptions
            feature_descriptions = {
                "OPB": "Outstanding Principal Balance of the customer's loan",
                "interest_rate": "Current interest rate applicable to the customer's loan",
                "tenure": "Duration of the loan in months",
                "credit_score_band": "Customer's credit score category (Excellent, Good, Fair, Poor)",
                "LTV": "Loan-to-Value ratio indicating the risk level of the loan",
                "age": "Customer's age in years",
                "income": "Customer's annual income",
                "employment_length": "Length of employment in years",
                "debt_to_income": "Ratio of total debt to income",
                "payment_history": "Customer's payment history score",
                "loan_amount": "Original loan amount",
                "loan_type": "Type of loan (Personal, Mortgage, etc.)",
                "property_value": "Value of the property (for mortgage loans)",
                "down_payment": "Amount of down payment made",
                "loan_purpose": "Purpose of the loan",
                "marital_status": "Customer's marital status",
                "education": "Customer's education level",
                "residence_type": "Type of residence (Own, Rent, etc.)",
                "number_of_dependents": "Number of dependents",
                "previous_loans": "Number of previous loans"
            }

            # Create feature info DataFrame
            feature_info = pd.DataFrame({
                'Feature': selected_features,
                'Description': [feature_descriptions.get(feat, f"Description for {feat}") for feat in selected_features]
            })

            # Store the selected features in model state
            model_state["recommended_features"] = model_state["combined_dataset"][selected_features].copy()

            # Display the features
            st.subheader("Recommended Features")
            # Create a dataframe for the features with the same styling as good-to-have section
            features_df = pd.DataFrame({
                "Feature": feature_info["Feature"],
                "Description": feature_info["Description"],
                "Min": feature_info["Min"],
                "Max": feature_info["Max"],
                "Mean": feature_info["Mean"],
                "Data Type": feature_info["Data Type"]
            })

            # Display the features in a dataframe with custom styling
            st.data_editor(
                features_df,
                column_config={
                    "Feature": st.column_config.TextColumn(
                        "Feature 🔍",
                        width="medium",
                        disabled=True
                    ),
                    "Description": st.column_config.TextColumn(
                        "Description",
                        width="large",
                        disabled=True
                    ),
                    "Min": st.column_config.TextColumn(
                        "Min",
                        width="small",
                        disabled=True
                    ),
                    "Max": st.column_config.TextColumn(
                        "Max",
                        width="small",
                        disabled=True
                    ),
                    "Mean": st.column_config.TextColumn(
                        "Mean",
                        width="small",
                        disabled=True
                    ),
                    "Data Type": st.column_config.TextColumn(
                        "Data Type",
                        width="small",
                        disabled=True
                    )
                },
                hide_index=True,
                use_container_width=True,
                key="recommended_features_editor"
            )
            st.success("Features have been recommended!")
            st.session_state["should_rerun"] = True
        else:
            st.warning("Please complete the merge operations first to get recommended features.")
    except Exception as e:
        st.error(f"Error recommending features: {str(e)}")

# Add this CSS at the beginning of your file, after the imports
st.markdown("""
    <style>
    .main-action-button {
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "single_transform_success" not in st.session_state:
    st.session_state.single_transform_success = None  # Initialize single_transform_success
if "multi_transform_success" not in st.session_state:
    st.session_state.multi_transform_success = None  # Initialize multi_transform_success

# --- Model Selection (Replace your existing Model Selection section) ---
st.sidebar.title("Model Selection")

# Initialize 'active_model' in session state if not already present
if "active_model" not in st.session_state:
    st.session_state.active_model = MODEL_NAMES[0] # Set 'Default Model' as the initial active model

# Get the current active model to set the initial value of the selectbox
current_active_model_index = MODEL_NAMES.index(st.session_state.active_model) if st.session_state.active_model in MODEL_NAMES else 0

def on_model_select_callback():
    # This callback is triggered when the selectbox value changes.

    # 1. Save the state of the *currently active* model BEFORE switching
    if st.session_state.get("active_model"): # Ensure active_model exists to prevent error on initial load
        print(f"Saving state for current active model: {st.session_state.active_model}")
        save_model_state(st.session_state.active_model)

    # 2. Update 'active_model' in session state to the newly selected model
    st.session_state.active_model = st.session_state.selected_model_dropdown_value

    # 3. Load the state for the *newly active* model
    # The load_model_state function will handle initializing if no saved state exists
    print(f"Loading state for new active model: {st.session_state.active_model}")
    load_model_state(st.session_state.active_model)

    # 4. Rerun the Streamlit app to reflect the loaded/initialized state
    st.session_state["should_rerun"] = True

# Streamlit Selectbox for model selection
selected_model = st.sidebar.selectbox(
    "Select Model",
    MODEL_NAMES,
    index=current_active_model_index,
    key="selected_model_dropdown_value", # Unique key for the selectbox value
    on_change=on_model_select_callback,
)

# Display Feature Engineering Title
st.markdown(F"# FEATURE ENGINEERING")

# Display the steps to follow on this page
st.markdown("""
<div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 10px; background-color: #f9f9f9; color: black;">
    <h3 style="color: #4CAF50;">🔍 Steps to Follow on This Page</h3>
    <ol>
        <li><b>Preview Data</b><br>
            Begin by reviewing the data selected in the previous page (Data Exploration).</li>
        <li><b>Select Input Data</b><br>
            Apply desired filters to prepare the input data for each model based on your requirements.</li>
        <li><b>Integrate Datasets</b><br>
            Use the Data Integration section to combine the filtered datasets into a unified dataset.</li>
        <li><b>AI-Powered Feature Recommendation</b><br>
            Leverage AI to generate new features from the existing ones. Review and add these features as recommended in the AI-Powered Feature Recommendation section.</li>
        <li><b>Feature Transformation</b><br>
            Perform feature transformation (individually or in combination) using the available transformation sub-sections.</li>
        <li><b>Preview Output Variable</b><br>
            Preview the column that represents the output variable for the selected model.</li>
        <li><b>Verify Mandatory Features</b><br>
            The system will display mandatory features for the selected model. If any are missing, revisit feature selection to ensure all required features are included.</li>
        <li><b>Select Additional Features</b><br>
            Optionally, choose additional features that may help improve the model’s predictive performance in the Good to have Features section.</li>
    </ol>
    <p style="color: red; font-weight: bold;">Note: Complete the same steps for the remaining two models to prepare their input data and features.</p>        
</div>
""", unsafe_allow_html=True)





# Display the current active model at the top of the main content area
st.markdown(f"## {st.session_state.active_model}")

# --- Initial/First-Load Setup for the Active Model ---
# This block ensures that the state for the currently active model is loaded or initialized
# when the app first starts or after a rerun from the callback.
# It checks if the model's main state dictionary is present in st.session_state.
if f"{st.session_state.active_model}_state" not in st.session_state:
    print(f"Initial setup: State for '{st.session_state.active_model}' not found in session. Attempting to load or initialize.")
    load_model_state(st.session_state.active_model) # This will initialize if no saved state exists

# Get the active model and its state
active_model = st.session_state.active_model
model_state = st.session_state[f"{active_model}_state"]

# Use model-specific operations complete state
operations_complete = st.session_state.get(f"{active_model}_operations_complete", {
    "merge": False,
    "recommend": False,
    "accept": False
})

# Use model-specific filter and merge states
show_filter_data = st.session_state.get(f"{active_model}_show_filter_data", False)
show_merge = st.session_state.get(f"{active_model}_show_merge", False)

# Use model-specific transform success states
single_transform_success = st.session_state.get(f"{active_model}_single_transform_success", None)
multi_transform_success = st.session_state.get(f"{active_model}_multi_transform_success", None)

# Use model-specific feature states
recommended_features = st.session_state.get(f"{active_model}_recommended_features", pd.DataFrame())
final_dataset = st.session_state.get(f"{active_model}_final_dataset", pd.DataFrame())
selected_features = st.session_state.get(f"{active_model}_selected_features", [])
feature_checkboxes = st.session_state.get(f"{active_model}_feature_checkboxes", {})

# Use model-specific filter blocks
filter_blocks = st.session_state.get(f"{active_model}_filter_blocks", [{
    "dataset": "Bureau Data",
    "feature": "",
    "operation": "Greater Than",
    "value": 0,
    "output_name": ""
}])

# Use model-specific target variable states
target_column = st.session_state.get(f"{active_model}_target_column", None)
target_feature = st.session_state.get(f"{active_model}_target_feature", None)
final_dataset_json = st.session_state.get(f"{active_model}_final_dataset_json", None)

# --- Dataset Selection Section ---
col1, col2, col3 = st.columns(3)

# Ensure the datasets are stored in session state
bureau_name = "Bureau Data"
loan_level_name = "Loan-Level Data"
installments_name = "Payment Data"

# Store the names and dataframes in a dictionary for easy access
dataset_mapping = {
    bureau_name: model_state["bureau_data"],
    loan_level_name: model_state["loan_level_data"],
    installments_name: model_state["installments_data"],
}

# Initialize visibility state for each dataset
if "show_bureau_data" not in st.session_state:
    st.session_state.show_bureau_data = False
if "show_loan_level_data" not in st.session_state:
    st.session_state.show_loan_level_data = False
if "show_installments_data" not in st.session_state:
    st.session_state.show_installments_data = False

# Create buttons for each dataset
with col1:
    if st.button(bureau_name, key="bureau_data_button"):
        st.session_state.show_bureau_data = not st.session_state.show_bureau_data

with col2:
    if st.button(loan_level_name, key="loan_level_data_button"):
        st.session_state.show_loan_level_data = not st.session_state.show_loan_level_data

with col3:
    if st.button(installments_name, key="installments_data_button"):
        st.session_state.show_installments_data = not st.session_state.show_installments_data

# Display the selected dataset
if st.session_state.show_bureau_data:
    st.subheader(f"{bureau_name} Preview")
    bureau_data = dataset_mapping[bureau_name]
    if bureau_data is not None and not bureau_data.empty:
        st.dataframe(bureau_data.head(), use_container_width=True)
    else:
        st.warning(f"{bureau_name} is empty or not loaded.")

if st.session_state.show_loan_level_data:
    st.subheader(f"{loan_level_name} Preview")
    loan_level_data = dataset_mapping[loan_level_name]
    if loan_level_data is not None and not loan_level_data.empty:
        st.dataframe(loan_level_data.head(), use_container_width=True)
    else:
        st.warning(f"{loan_level_name} is empty or not loaded.")

if st.session_state.show_installments_data:
    st.subheader(f"{installments_name} Preview")
    installments_data = dataset_mapping[installments_name]
    if installments_data is not None and not installments_data.empty:
        st.dataframe(installments_data.head(), use_container_width=True)
    else:
        st.warning(f"{installments_name} is empty or not loaded.")

# Store the names and dataframes in a dictionary for easy access
dataset_mapping = {
    bureau_name: model_state["bureau_data"],
    loan_level_name: model_state["loan_level_data"],
    installments_name: model_state["installments_data"],
}

# --- Further operations ---
# You can now easily replace the datasets in session state
# Example:
# new_bureau_data = pd.read_csv("new_bureau_data.csv")  # Load your actual data
# st.session_state.bureau_data = new_bureau_data  # Replace the data in session state

# The rest of your feature engineering code can then use the data from the session state,
# ensuring it's using the updated datasets.
st.markdown("---")

# --- Filter Data Section ---
def filter_data_section():
    """
    Displays the UI and handles the logic for filtering data. This function assumes
    that the dataframes and session state variables like 'bureau_data', 'loan_level_data',
    'installments_data' and 'filter_blocks' are already initialized. It also assumes
    the dataset names are stored as: bureau_name, loan_level_name, installments_name.
    Restructured to arrange inputs in two rows per filter block.
    """
    active_model = st.session_state.active_model
    model_state = st.session_state.get(f"{active_model}_state", {})
    raw_datasets = model_state.get("raw_datasets", {})
    combined_dataset = model_state.get("combined_dataset", pd.DataFrame()) # Get the combined dataset

    # Ensure combined_dataset is always a DataFrame, even if stored as None
    combined_dataset = model_state.get("combined_dataset")
    if combined_dataset is None: # Check if it was explicitly stored as None
        combined_dataset = pd.DataFrame() # Default to empty DataFrame if None

    # --- Initialize filter blocks in session state ---
    if f"{active_model}_filter_blocks" not in st.session_state:
         dataset_names = feat_engg_backend.get_table_names(raw_datasets) # Use backend to get dataset names
         st.session_state[f"{active_model}_filter_blocks"] = [{
             "dataset": dataset_names[0] if dataset_names else "", # Default to the first dataset name
             "feature": "",
             "operation": feat_engg_backend.get_filter_operations()[0], # Default operation from backend
             "value": None, # Initialize value as None
             "output_name": ""
         }]
    filter_blocks = st.session_state[f"{active_model}_filter_blocks"] # Get the blocks for the active model

    # --- Initialize intermediate_filtered_datasets in model_state ---
    # This dictionary will store all datasets generated by filter operations (e.g., A_filtered)
    if "intermediate_filtered_datasets" not in model_state:
        model_state["intermediate_filtered_datasets"] = {}

    # --- Initialize current_filtered_dataset in session state ---
    # This DataFrame will be the one actively filtered and displayed
    if f"{active_model}_current_filtered_dataset" not in st.session_state:
        st.session_state[f"{active_model}_current_filtered_dataset"] = combined_dataset.copy()
    
    # Update current_filtered_dataset if combined_dataset changes
    last_combined_for_filter = model_state.get("last_combined_dataset_for_filter")
    # Ensure last_combined_for_filter is a DataFrame for .equals() comparison
    if not isinstance(last_combined_for_filter, pd.DataFrame):
        last_combined_for_filter = pd.DataFrame() # Treat non-DataFrame or None as empty DataFrame for comparison

    if not combined_dataset.empty and not combined_dataset.equals(last_combined_for_filter):
        st.session_state[f"{active_model}_current_filtered_dataset"] = combined_dataset.copy()
        model_state["last_combined_dataset_for_filter"] = combined_dataset.copy()
        st.session_state[f"{active_model}_state"] = model_state


    # --- Define available operations ---
    OPERATIONS = feat_engg_backend.get_filter_operations() # Get operations from backend

    # Display Filter section description
    st.markdown(f"##### Filter any of the datasets based on any selected feature to tailor the input data for model.")

    # Use a container for the filter controls
    filter_container = st.container()

    with filter_container:
        # Track all available datasets as you go through each filter block
        available_datasets = raw_datasets.copy()
        available_datasets.update(model_state.get("intermediate_filtered_datasets", {}))

        # This dict will be updated as we go through each filter block
        intermediate_filtered_datasets = model_state.get("intermediate_filtered_datasets", {})

        for i, filter_block in enumerate(filter_blocks):
            # For each filter block after the first, add outputs from previous blocks
            if i > 0:
                for prev_block in filter_blocks[:i]:
                    prev_output_name = prev_block.get("output_name")
                    if prev_output_name and prev_output_name in model_state["intermediate_filtered_datasets"]:
                        available_datasets[prev_output_name] = model_state["intermediate_filtered_datasets"][prev_output_name]

            all_available_datasets = list(available_datasets.keys())

            

            st.subheader(f"Filter {i + 1}")

            # --- Row 1: Remove button, Select Table, Select Column ---
            # Using adjusted ratios for column widths
            row1_cols = st.columns([0.5, 3, 3])

            with row1_cols[0]:
                # Remove button
                if st.button("❌", key=f"remove_filter_{i}"):
                    active_model = st.session_state.active_model
                    filter_blocks_state = st.session_state.get(f"{active_model}_filter_blocks", []) # Get from state
                    if i < len(filter_blocks_state): # Add a check to prevent index errors
                        filter_blocks_state.pop(i)
                        st.session_state[f"{active_model}_filter_blocks"] = filter_blocks_state # Explicitly update session state
                        st.rerun()

            with row1_cols[1]:
                current_dataset_selection = filter_block.get("dataset", all_available_datasets[0] if all_available_datasets else "")
                if current_dataset_selection not in all_available_datasets:
                    current_dataset_selection = all_available_datasets[0] if all_available_datasets else ""

                selected_dataset_name = st.selectbox(
                    "Select Table",
                    all_available_datasets,
                    index=all_available_datasets.index(current_dataset_selection) if current_dataset_selection in all_available_datasets else 0,
                    key=f"dataset_{i}"
                )
                filter_blocks[i]["dataset"] = selected_dataset_name

            with row1_cols[2]:
                selected_dataset_name = filter_blocks[i]["dataset"]
                df_for_features = available_datasets.get(selected_dataset_name, pd.DataFrame())
                available_features = feat_engg_backend.get_features_for_table_df(df_for_features)
                current_feature = filter_block.get("feature")
                if current_feature not in available_features:
                    filter_blocks[i]["feature"] = available_features[0] if available_features else ""
                    current_feature = filter_blocks[i]["feature"]

                selected_feature = st.selectbox("Select Column", available_features,
                                                index=available_features.index(current_feature) if current_feature in available_features else 0,
                                                key=f"feature_{i}")
                filter_blocks[i]["feature"] = selected_feature

            # --- Row 2: Select Filter Type, Enter Value, Output Table ---
            # Using adjusted ratios for column widths
            row2_cols = st.columns([3, 3, 3])


            # # Check if the selected feature is a datetime column
            # is_datetime = False
            # if selected_feature and not df_for_features.empty:
            #     try:
            #         is_datetime = pd.api.types.is_datetime64_any_dtype(df_for_features[selected_feature])
            #     except Exception:
            #         is_datetime = False

            # if is_datetime and filter_blocks[i].get("operation") != "Between":
            #     filter_blocks[i]["operation"] = "Between"


            with row2_cols[0]:
                # Select Filter Type
                # Use feat_engg_backend to get available operations
                OPERATIONS = feat_engg_backend.get_filter_operations()
                operation = st.selectbox("Select Filter Type", OPERATIONS,
                                         index=OPERATIONS.index(filter_block.get("operation", OPERATIONS[0])) if filter_block.get("operation") in OPERATIONS else 0,
                                         key=f"operation_{i}")
                # Update the filter block's operation immediately
                filter_blocks[i]["operation"] = operation

            with row2_cols[1]:
                # Enter Value to Filter By (Handle dynamic input based on the selected operation)
                current_value = filter_block.get("value")
                operation = filter_blocks[i]["operation"] # Get the updated operation

                # Detect if the selected feature is a datetime column
                is_datetime = False
                if selected_feature and not df_for_features.empty:
                    try:
                        is_datetime = pd.api.types.is_datetime64_any_dtype(df_for_features[selected_feature])
                    except Exception:
                        is_datetime = False

                # --- Datetime filtering ---
                if is_datetime:
                    # Default to min/max of the column if available
                    col_min = df_for_features[selected_feature].min()
                    col_max = df_for_features[selected_feature].max()
                    
                    # Operations that require a single date
                    single_date_ops = [
                        "Greater Than", "Less Than", "Equal To", "Not Equal To",
                        "Greater Than or Equal To", "Less Than or Equal To"
                    ]
                    # Operations that should freeze the value input for datetime
                    freeze_ops = ["Is In List", "Is Null", "Is Not Null", "Contains String"]

                    if operation == "Between":
                        # Use two separate date pickers for start and end
                        default_start = current_value[0] if isinstance(current_value, (list, tuple)) and len(current_value) == 2 else col_min
                        default_end = current_value[1] if isinstance(current_value, (list, tuple)) and len(current_value) == 2 else col_max

                        col_start, col_end = st.columns(2)
                        with col_start:
                            start_date = st.date_input(
                                "Start Date",
                                value=default_start,
                                min_value=col_min,
                                max_value=col_max,
                                key=f"start_date_{i}"
                            )
                        with col_end:
                            end_date = st.date_input(
                                "End Date",
                                value=default_end,
                                min_value=col_min,
                                max_value=col_max,
                                key=f"end_date_{i}"
                            )

                        # Ensure start_date <= end_date
                        if pd.to_datetime(start_date) > pd.to_datetime(end_date):
                            st.warning("Start date is after end date. Please select a valid range.")
                        value = (pd.to_datetime(start_date), pd.to_datetime(end_date))
                        filter_blocks[i]["value"] = value

                    elif operation in single_date_ops:
                        default_date = current_value if isinstance(current_value, (pd.Timestamp, datetime.date, datetime.datetime)) else col_min
                        selected_date = st.date_input(
                            "Select Date",
                            value=default_date,
                            min_value=col_min,
                            max_value=col_max,
                            key=f"single_date_{i}"
                        )
                        value = pd.to_datetime(selected_date)
                        filter_blocks[i]["value"] = value

                    elif operation in freeze_ops:
                        st.text_input("Value", value="N/A", key=f"value_{i}", disabled=True)
                        filter_blocks[i]["value"] = None

                    else:
                        st.warning("Selected operation is not supported for datetime columns.")
                        filter_blocks[i]["value"] = None
                else:
                    # ...existing logic for other types...
                    if operation in ["Greater Than", "Less Than", "Equal To", "Not Equal To", "Greater Than or Equal To", "Less Than or Equal To"]:
                        default_value = current_value if isinstance(current_value, (int, float)) else 0.0
                        value = st.number_input("Enter Value to Filter By", value=default_value, key=f"value_{i}")
                    elif operation == "Is In List":
                        if isinstance(current_value, str):
                            default_value = current_value
                        elif isinstance(current_value, list):
                            default_value = ','.join(map(str, current_value))
                        else:
                            default_value = ''
                        value = st.text_input("Enter values (comma-separated)", value=default_value, key=f"value_{i}")
                    elif operation == "Between":
                        default_value1 = 0.0
                        default_value2 = 0.0
                        if isinstance(current_value, (tuple, list)) and len(current_value) == 2:
                            if isinstance(current_value[0], (int, float)):
                                default_value1 = current_value[0]
                            if isinstance(current_value[1], (int, float)):
                                default_value2 = current_value[1]
                        col_val1, col_val2 = st.columns(2)
                        with col_val1:
                            value1 = st.number_input("Start Value", value=default_value1, key=f"value_{i}_start")
                        with col_val2:
                            value2 = st.number_input("End Value", value=default_value2, key=f"value_{i}_end")
                        value = (value1, value2)
                    elif operation in ["Is Null", "Is Not Null"]:
                        st.text_input("Value", value="N/A", key=f"value_{i}", disabled=True)
                        value = None
                    elif operation == "Contains String":
                        default_value = current_value if isinstance(current_value, str) else ''
                        value = st.text_input("Enter substring", value=default_value, key=f"value_{i}")
                    else:
                        default_value = str(current_value) if current_value is not None else ''
                        value = st.text_input("Select Value", value=default_value, key=f"value_{i}")

                    filter_blocks[i]["value"] = value

            with row2_cols[2]:
                # Output Table Name
                selected_feature = filter_blocks[i]["feature"]
                operation = filter_blocks[i]["operation"]
                value = filter_blocks[i]["value"]
                suggested_name = ""

                if selected_feature and operation:
                    suggested_name = f"{selected_feature}_{operation.replace(' ', '_').lower()}"
                    if value is not None and operation not in ["Is Null", "Is Not Null"]:
                         if isinstance(value, (tuple, list)):
                             value_str = '_'.join(map(lambda x: str(x).replace('.', '').replace('-', 'neg'), value))
                             suggested_name += f"_{value_str}"
                         elif isinstance(value, str):
                             value_str = value.replace(' ', '_').replace('.', '').replace('-', 'neg')
                             suggested_name += f"_{value_str}"
                         else:
                             suggested_name += f"_{str(value).replace('.', '').replace('-', 'neg')}"
                    suggested_name = suggested_name[:50] # Arbitrary length limit for name


                filter_description_name = st.text_input(
                    "Output Table Name", # Changed label for clarity
                    value=filter_blocks[i].get("output_name", suggested_name),
                    key=f"filter_description_{i}",
                    help="Provide a unique name for the new table generated by this filter."
                )
                filter_blocks[i]["output_name"] = filter_description_name

            # --- Immediately apply the filter and store the result ---
            input_dataset_name = filter_blocks[i].get("dataset")
            output_table_name = filter_blocks[i].get("output_name")
            if input_dataset_name and output_table_name and input_dataset_name in available_datasets:
                df_to_filter = available_datasets[input_dataset_name].copy()
                processed_block = filter_blocks[i].copy()
                if processed_block.get("operation") == "Is In List" and isinstance(processed_block.get("value"), str):
                    processed_block["value"] = [v.strip() for v in processed_block["value"].split(',') if v.strip()]
                filtered_df = feat_engg_backend.apply_filter_block(df_to_filter, processed_block)
                if not filtered_df.empty and len(filtered_df.columns) > 0:
                    intermediate_filtered_datasets[output_table_name] = filtered_df
                    model_state["intermediate_filtered_datasets"][output_table_name] = filtered_df
                else:
                    st.warning(f"Filtered table '{output_table_name}' is empty or has no columns. Please check your filter logic.")

            # Update available_datasets for the next filter block
            available_datasets = raw_datasets.copy()
            available_datasets.update(intermediate_filtered_datasets)

        # Save changes back to session state
        st.session_state[f"{active_model}_filter_blocks"] = filter_blocks
        model_state["intermediate_filtered_datasets"] = intermediate_filtered_datasets
        st.session_state[f"{active_model}_state"] = model_state


        if st.button("➕ Add Filter"):
            print("Add filter button clicked.")
            active_model = st.session_state.active_model
            filter_blocks_state = st.session_state.get(f"{active_model}_filter_blocks", [])
            # Update dataset_names to include intermediate_filtered_datasets
            all_available_datasets = list(raw_datasets.keys()) + list(model_state["intermediate_filtered_datasets"].keys())
            
            filter_blocks_state.append({
                "dataset": all_available_datasets[0] if all_available_datasets else "", # Default to first available dataset
                "feature": "",
                "operation": feat_engg_backend.get_filter_operations()[0],
                "value": None,
                "output_name": ""
            })
            st.session_state[f"{active_model}_filter_blocks"] = filter_blocks_state
            st.rerun()
            print("Exiting add filter button click.")


        # --- Button to apply all filters ---
        if st.button("Apply All Filters", key="apply_all_filters_button"):
            # Prepare a working set of all datasets (raw + intermediate)
            # This allows filters to pick from any available table
            working_datasets = raw_datasets.copy()
            working_datasets.update(model_state["intermediate_filtered_datasets"]) # Add intermediate ones

            latest_filtered_datasets_per_original = {} # To track the latest version of each original table

            if not filter_blocks:
                 st.warning("No filter blocks defined to apply.")
            else:
                try:
                    for i, block in enumerate(filter_blocks):
                        input_dataset_name = block.get("dataset")
                        output_table_name = block.get("output_name")
                        
                        if not input_dataset_name or not output_table_name:
                            st.warning(f"Filter {i+1} is missing input dataset or output name. Skipping.")
                            continue

                        if input_dataset_name not in working_datasets:
                            st.error(f"Input dataset '{input_dataset_name}' for Filter {i+1} not found. Skipping filter.")
                            continue
                        
                        df_to_filter = working_datasets[input_dataset_name].copy() # Get a copy to filter

                        # Process 'Is In List' value if it's a string
                        processed_block = block.copy()
                        if processed_block.get("operation") == "Is In List" and isinstance(processed_block.get("value"), str):
                             processed_block["value"] = [v.strip() for v in processed_block["value"].split(',') if v.strip()]


                        # Apply the filter using the backend function
                        filtered_df = feat_engg_backend.apply_filter_block(df_to_filter, processed_block)

                        # Store the result in intermediate_filtered_datasets
                        model_state["intermediate_filtered_datasets"][output_table_name] = filtered_df
                        working_datasets[output_table_name] = filtered_df # Add to working set for subsequent filters
                        st.success(f"Filter {i+1} applied: '{input_dataset_name}' -> '{output_table_name}'.")

                        # Track the latest filtered version of each original dataset
                        # This logic assumes the output_table_name implies a lineage back to an original table.
                        # You might need more sophisticated lineage tracking if names don't imply original.
                        # For simplicity, if the output name contains an original dataset name, we update it.
                        for original_name in raw_datasets.keys():
                            if output_table_name.startswith(original_name) or output_table_name == original_name:
                                latest_filtered_datasets_per_original[original_name] = output_table_name
                                break
                        
                        # If a filter results in an empty DataFrame, you might want to stop further processing for that lineage
                        if filtered_df.empty:
                            st.warning(f"Output table '{output_table_name}' is empty. Subsequent filters relying on this might yield empty results.")
                            # Consider if you want to explicitly remove it from working_datasets
                            # del working_datasets[output_table_name] # Optional: Remove empty table from consideration


                    # After all filters are processed, update the current_filtered_dataset for preview
                    # You might want to default to the last generated table for preview, or the combined_dataset
                    if model_state["intermediate_filtered_datasets"]:
                        # Get the name of the last generated filtered table
                        last_filtered_table_name = list(model_state["intermediate_filtered_datasets"].keys())[-1]
                        st.session_state[f"{active_model}_current_filtered_dataset"] = model_state["intermediate_filtered_datasets"][last_filtered_table_name]
                    else:
                        st.session_state[f"{active_model}_current_filtered_dataset"] = combined_dataset.copy() # Fallback

                    st.success("✅ All filters processed successfully!")    

                    # Preview the last filtered dataset if available
                    if model_state["intermediate_filtered_datasets"]:
                        last_filtered_table_name = list(model_state["intermediate_filtered_datasets"].keys())[-1]
                        last_filtered_df = model_state["intermediate_filtered_datasets"][last_filtered_table_name]
                        if isinstance(last_filtered_df, pd.DataFrame) and not last_filtered_df.empty:
                            with st.expander(f"📂 Preview of Filtered Table: {last_filtered_table_name}", expanded=False):
                                st.markdown(f"#### Preview of Filtered Table: {last_filtered_table_name}")
                                st.write(f"Shape: {last_filtered_df.shape}")
                                st.dataframe(last_filtered_df, use_container_width=True)
                    else:
                        st.info("The last filtered dataset is empty. Please check your filter configuration.")

                except Exception as e:
                    st.error(f"Error applying filters: {str(e)}")
            
            # Save the latest filtered tables of original tables
            # This will update model_state["intermediate_filtered_datasets"] which acts as the saved state
            model_state["filtered_datasets"] = model_state["intermediate_filtered_datasets"].copy()
            st.info("Latest filtered tables are saved and will be available for merging.")


    st.session_state[f"{active_model}_state"] = model_state # Always save changes back to session state

# --- Main Section ---
# Define a callback function to toggle the visibility of the filter section
def show_filter_data_callback():
    st.session_state.show_filter_data = not st.session_state.show_filter_data  # Toggle visibility

# Add the button with a callback
if "show_filter_data" not in st.session_state:
    st.session_state["show_filter_data"] = False

# Filter Data Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.button("🔄 Input Data Selection for Model", key="show_filter_data_button", on_click=show_filter_data_callback, use_container_width=True)


# Display the filter section only if the button has been clicked
if st.session_state.show_filter_data:
    filter_data_section()

st.markdown("---")
# Debugging breakpoint to inspect session state and variables
# --- Initialize Session State ---
# Initialize session state variables if they don't exist
active_model = st.session_state.active_model # Get active model name here
model_state = st.session_state.get(f"{active_model}_state", {}) # Get model state

if f"{active_model}_show_merge" not in st.session_state:
    st.session_state[f"{active_model}_show_merge"] = False
if f"{active_model}_merge_blocks" not in st.session_state:
    # Initialize with one default merge block
    st.session_state[f"{active_model}_merge_blocks"] = [{
        "left_table": "Bureau Data (Filtered)", # Default to the filtered data
        "right_table": "Loan-Level Data (Filtered)", # Default to the filtered data
        "how": "inner",
        "on": [],
        "left_on": [],
        "right_on": [],
        "merged_name": "",
    }]
if f"{active_model}_merged_tables" not in st.session_state:
    st.session_state[f"{active_model}_merged_tables"] = {}
if f"{active_model}_combined_dataset" not in st.session_state:
    st.session_state[f"{active_model}_combined_dataset"] = None # This will hold the final merged result


# --- Main Section ---
# Define a callback function to toggle the visibility of the filter section

# Debugging breakpoint to inspect session state and variables
# --- Initialize Session State ---
# Initialize session state variables if they don't exist
active_model = st.session_state.active_model # Get active model name here
model_state = st.session_state.get(f"{active_model}_state", {}) # Get model state

if f"{active_model}_show_merge" not in st.session_state:
    st.session_state[f"{active_model}_show_merge"] = False
if f"{active_model}_merge_blocks" not in st.session_state:
    # Initialize with one default merge block
    st.session_state[f"{active_model}_merge_blocks"] = [{
        "left_table": "Bureau Data (Filtered)", # Default to the filtered data
        "right_table": "Loan-Level Data (Filtered)", # Default to the filtered data
        "how": "inner",
        "on": [],
        "left_on": [],
        "right_on": [],
        "merged_name": "",
    }]
if f"{active_model}_merged_tables" not in st.session_state:
    st.session_state[f"{active_model}_merged_tables"] = {}
if f"{active_model}_combined_dataset" not in st.session_state:
    st.session_state[f"{active_model}_combined_dataset"] = None # This will hold the final merged result


############################################################################################################################################3

############################################################################################################################################3

# --- Merge Datasets Section ---
# Merge Datasets Button
col1, col2, col3 = st.columns([1, 2, 1]) # You might want to adjust these ratios for better button centering if needed
with col2:
    # Ensure the show_merge state is initialized for the active model
    active_model = st.session_state.active_model
    model_state = st.session_state.get(f"{active_model}_state", {})
    if "show_merge" not in model_state:
         model_state["show_merge"] = False
         st.session_state[f"{active_model}_state"] = model_state # Update session state if initializing

    # Use the updated width for the button column if you changed it previously
    # col1, col2, col3 = st.columns([1, 5, 1]) # Example if you want a wider button container
    st.button("🔄 Data Integration", key="merge_btn", on_click=show_merge_callback, use_container_width=True)


# Display the merge section if show_merge is True for the active model
active_model = st.session_state.active_model
model_state = st.session_state.get(f"{active_model}_state", {}) # Get model state

if model_state.get("show_merge", False): # Check the show_merge state within the model state

    # Prepare available tables for merging
    # Get raw, filtered, and previously merged tables from model state
    raw_datasets = model_state.get("raw_datasets", {})
    filtered_datasets = model_state.get("filtered_datasets", {})
    merged_tables = model_state.get("merged_tables", {})

    # Use the backend function to get the names of all tables available for merging
    table_names = feat_engg_backend.get_merge_available_table_names(raw_datasets, filtered_datasets, merged_tables)

    # Create a dictionary mapping these names to the actual DataFrames for UI logic
    # Combine raw, filtered, and merged datasets
    available_tables = raw_datasets.copy()
    available_tables.update(filtered_datasets)
    available_tables.update(merged_tables)

    # Initialize merge_blocks in model state if empty when the section is shown
    if "merge_blocks" not in model_state or not model_state["merge_blocks"]:
         # Add a default merge block if the list is empty and the section is shown
         if table_names: # Only add if there are tables available
              # Determine a sensible default for the left table (last merged or first available)
              last_merged_name = list(merged_tables.keys())[-1] if merged_tables else (table_names[0] if table_names else "") # Fallback, handle empty table_names
              # Ensure the last merged name is actually in the available tables
              if last_merged_name not in table_names:
                   last_merged_name = table_names[0] if table_names else ""


              # Determine a sensible default for the right table
              default_right_table = table_names[0] if table_names else ""
              if default_right_table == last_merged_name and len(table_names) > 1:
                   # Find a different table if the first available is the same as the default left
                   try:
                       current_index = table_names.index(default_right_table)
                       if current_index + 1 < len(table_names):
                           default_right_table = table_names[current_index + 1]
                       elif current_index > 0: # If next index is out of bounds, try previous
                            default_right_table = table_names[current_index - 1]
                       else:
                           default_right_table = table_names[0] # Fallback to first if only one table

                   except ValueError:
                       pass # Should not happen if default_right_table is in the list


              model_state["merge_blocks"] = [{
                  "left_table": last_merged_name, # Default to the name of the last merged table or first available
                  "right_table": default_right_table, # Default to another available table
                  "how": "inner",
                  "on": [],
                  "left_on": [],
                  "right_on": [],
                  "merged_name": "", # Start with an empty output name by default
              }]
         else:
              model_state["merge_blocks"] = [] # Ensure it's an empty list if no tables
         st.session_state[f"{active_model}_state"] = model_state # Update session state
         # st.rerun() # No need to rerun here, as the section is already being displayed


    # Default suffixes for handling duplicate column names after merge
    default_suffixes = ("_x", "_y")

    # --- Display and Configure Merge Operations ---
    # Iterate through each merge block defined in model state
    merge_blocks = model_state.get("merge_blocks", []) # Get the blocks from model_state

    # if not table_names and merge_blocks:
    #      st.warning("No datasets available for merging. Please check your data loading and filtering steps.")
         # Optionally clear merge blocks if they refer to non-existent tables
         # model_state["merge_blocks"] = []
         # st.session_state[f"{active_model}_state"] = model_state
         # st.rerun() # Rerun if clearing blocks
         # return # Exit the merge section if no tables
    # Start with all available tables (raw, filtered, merged)
    base_tables = raw_datasets.copy()
    base_tables.update(filtered_datasets)
    base_tables.update(merged_tables)

    # This dict will be updated as we go through each merge block
    intermediate_merged_tables = merged_tables.copy()

    # Display merge section description
    st.markdown(f"##### Combine all the tables into a unified dataset for streamlined analysis and modeling.")
    st.markdown(
    '<span style="color: red; font-weight: bold;">Note: Avoid selecting similar datasets (e.g., the original dataset and its filtered version) together, as this may lead to duplication or inconsistent merging.</span>',
    unsafe_allow_html=True
)


    for i, block in enumerate(merge_blocks):
        if i > 0:
            for prev_block in merge_blocks[:i]:
                prev_output_name = prev_block.get("merged_name")
                if prev_output_name and prev_output_name in intermediate_merged_tables:
                    base_tables[prev_output_name] = intermediate_merged_tables[prev_output_name]
        table_names = list(base_tables.keys())
        available_tables = base_tables.copy()  # <-- Make sure to update this for each block
        
        # --- Row 1: Remove button, Select Left Table, Select Right Table ---
        row1_cols = st.columns([0.5, 3, 3]) # Ratios adjusted for consistency

        with row1_cols[0]:
            # Cross Button to Remove Iteration
            if st.button("❌", key=f"remove_merge_{i}"):
                merge_blocks.pop(i)
                model_state["merge_blocks"] = merge_blocks # Update model_state
                st.session_state[f"{active_model}_state"] = model_state # Update session state
                st.rerun()

        # Ensure table_names is not empty before displaying subsequent elements in this block
        if not table_names:
             # Warning already displayed above the loop, just continue to next block
             continue # Skip rendering the rest of this block

        with row1_cols[1]:
            # Select the left table for the current merge operation
            left_table = st.selectbox(
                "Select Left Table",
                table_names,  # Options are all available table names
                key=f"merge_left_table_{i}",
                # Set default based on saved state or fallback to the first available table
                index=table_names.index(block.get("left_table", table_names[0])) if block.get("left_table") in table_names else 0
            )
            # Update block immediately
            merge_blocks[i]["left_table"] = left_table


        with row1_cols[2]:
            # Select the right table for the current merge operation
            right_table = st.selectbox(
                "Select Right Table",
                table_names,  # Options are all available table names
                key=f"merge_right_table_{i}",
                # Set default based on saved state or fallback to the first available table
                index=table_names.index(block.get("right_table", table_names[0])) if block.get("right_table") in table_names else 0
            )
            # Update block immediately
            merge_blocks[i]["right_table"] = right_table

        # Get column names for the selected left and right tables
        # Ensure the selected tables exist in available_tables before accessing columns
        left_df = available_tables.get(left_table, pd.DataFrame())
        right_df = available_tables.get(right_table, pd.DataFrame())

        left_cols = left_df.columns.tolist()
        right_cols = right_df.columns.tolist()
        # Find common columns between the two selected tables
        common_cols = list(set(left_cols) & set(right_cols))


        # --- Row 2: Select Column to Merge On, Select Columns from Left Table, Select Columns from Right Table ---
        # Use columns for the join key selections
        join_cols = st.columns([1, 1, 1]) # Adjust ratios as needed

        with join_cols[0]:
            # --- Select Merge On Column ---
            current_on_value = block.get("on", [])
            on_selected = bool(current_on_value and current_on_value[0])
            left_on_selected = bool(block.get("left_on"))
            right_on_selected = bool(block.get("right_on"))
            freeze_on = left_on_selected or right_on_selected

            # Only show column names, no blank, no default selection
            on = st.selectbox(
                "Select Merge On Column",
                common_cols,
                index=common_cols.index(current_on_value[0]) if on_selected and current_on_value[0] in common_cols else None,
                key=f"merge_on_{i}",
                help="Select a single column present in both tables to join on.",
                disabled=freeze_on,
                placeholder="Choose an option"  # Only available in Streamlit >=1.29
            )
            if on:
                merge_blocks[i]["on"] = [on]
            else:
                merge_blocks[i]["on"] = []
        
        freeze_left_right = on_selected

        with join_cols[1]:
            # --- Select Left On Columns ---
            left_on = st.multiselect(
                "Select Left On Columns",
                left_cols,
                default=block.get("left_on", []),
                key=f"merge_left_on_{i}",
                help="Select multiple columns from the left table to join on.",
                disabled=freeze_left_right  # Freeze if 'on' is selected
            )
            if not freeze_left_right:
                merge_blocks[i]["left_on"] = left_on

        with join_cols[2]:
            # --- Select Right On Columns ---
            right_on = st.multiselect(
                "Select Right On Columns",
                right_cols,
                default=block.get("right_on", []),
                key=f"merge_right_on_{i}",
                help="Select multiple columns from the right table to join on.",
                disabled=freeze_left_right  # Freeze if 'on' is selected
            )
            if not freeze_left_right:
                merge_blocks[i]["right_on"] = right_on

        # --- Row 3: Select Merge Type, Output Table ---
        # Use columns to place these two elements side-by-side
        merge_type_col, output_name_col = st.columns([1, 2]) # Adjust ratios as needed

        with merge_type_col:
            # Select the type of join (how)
            how = st.selectbox(
                "Select Merge Type",
                ["inner", "left", "right", "outer", "cross"],
                key=f"merge_how_{i}",
                # Set default based on saved state
                index=["inner", "left", "right", "outer", "cross"].index(block.get("how", "inner").lower())
            )
            # Update block immediately
            merge_blocks[i]["how"] = how

        with output_name_col:
            # --- Name the Resulting DataFrame ---
            # Calculate a suggested name (optional, but can be helpful)
            block_left_table = merge_blocks[i].get("left_table", "LeftTable")
            block_right_table = merge_blocks[i].get("right_table", "RightTable")
            block_how = merge_blocks[i].get("how", "inner")
            suggested_merged_name = f"{block_left_table}_merged_{block_right_table}_{block_how}_{i+1}" # Still calculate suggested name

            merged_name = st.text_input(
                "Output Table",
                value=block.get("merged_name", ""),  # Set default value to ""
                key=f"merge_merged_name_{i}",
                
            )
            # Update block immediately
            merge_blocks[i]["merged_name"] = merged_name

        # --- End of loop for a single merge block ---
        # The block dictionary is updated immediately after each widget's value is read,
        # and merge_blocks is a reference to the list in model_state.
        # An explicit consolidated update is not strictly necessary here but can be kept for clarity:
        # model_state["merge_blocks"][i] = {
        #     "left_table": merge_blocks[i]["left_table"],
        #     "right_table": merge_blocks[i]["right_table"],
        #     "how": merge_blocks[i]["how"],
        #     "on": merge_blocks[i]["on"],
        #     "left_on": merge_blocks[i]["left_on"],
        #     "right_on": merge_blocks[i]["right_on"],
        #     "merged_name": merge_blocks[i]["merged_name"],
        # }
        # st.session_state[f"{active_model}_state"] = model_state # Update session state


    model_state["merge_blocks"] = merge_blocks
    st.session_state[f"{active_model}_state"] = model_state
    
    # --- Buttons to Manage Merge Operations (outside the loop) ---
    if st.button("➕ Add Merge Operation", key="add_merge"):
        active_model = st.session_state.active_model
        model_state = st.session_state[f"{active_model}_state"]
        raw_datasets = model_state.get("raw_datasets", {})
        filtered_datasets = model_state.get("filtered_datasets", {})
        merged_tables = model_state.get("merged_tables", {})

        # Build available tables for merging (including all previous merged tables)
        available_tables = raw_datasets.copy()
        available_tables.update(filtered_datasets)
        available_tables.update(merged_tables)

        merge_blocks = model_state.get("merge_blocks", [])

        # Only try to merge if there is at least one block and it has a name
        if merge_blocks and merge_blocks[-1].get("merged_name"):
            try:
                # Only merge the last block (the new one)
                last_block = merge_blocks[-1]
                # Prepare a working copy of available tables for this merge
                working_tables = available_tables.copy()
                # Use backend to perform the merge for just this block
                merge_result = feat_engg_backend.apply_merge_blocks(working_tables, [last_block])
                # merge_result is a dict: {merged_name: merged_df}
                if last_block["merged_name"] in merge_result:
                    merged_tables[last_block["merged_name"]] = merge_result[last_block["merged_name"]]
                    model_state["merged_tables"] = merged_tables
                    st.success(f"Merged table '{last_block['merged_name']}' created successfully and is now available for further merges.")
                else:
                    st.error("Merge operation did not produce a table. Please check your configuration.")
            except Exception as e:
                st.error(f"Error creating merged table: {e}")

        # Add a new empty merge block for the next operation (default left_table to last merged)
        merged_keys = list(merged_tables.keys())
        last_merged_name = merged_keys[-1] if merged_keys else ""
        # Build available table names for the new block
        available_table_names = list(available_tables.keys()) + ([last_merged_name] if last_merged_name and last_merged_name not in available_tables else [])
        default_right_table = available_table_names[0] if available_table_names else ""
        if default_right_table == last_merged_name and len(available_table_names) > 1:
            default_right_table = available_table_names[1]
        model_state["merge_blocks"].append({
            "left_table": last_merged_name,
            "right_table": default_right_table,
            "how": "inner",
            "on": [],
            "left_on": [],
            "right_on": [],
            "merged_name": ""
        })
        st.session_state[f"{active_model}_state"] = model_state
        st.rerun()

    # --- Apply Merge Operations Button ---
    if st.button("✅ Apply Merge Operations", key="execute_merges", use_container_width=True):
        try:
            active_model = st.session_state.active_model
            model_state = st.session_state[f"{active_model}_state"]

            # Prepare the dictionary of all available datasets for the backend
            # Include raw, filtered, and previously merged tables
            all_available_datasets = model_state.get("raw_datasets", {}).copy()
            all_available_datasets.update(model_state.get("filtered_datasets", {}))
            # This is crucial: merged_tables should already contain results from previous 'Add Merge' clicks
            all_available_datasets.update(model_state.get("merged_tables", {}))


            if not all_available_datasets:
                 raise ValueError("No datasets available to perform merge operations.")

            merge_blocks = model_state.get("merge_blocks", []) # Get the current list of merge blocks

            if not merge_blocks:
                 raise ValueError("No merge operations defined to apply.")

            # Get only the last merge block
            last_merge_block = merge_blocks[-1]

            # Extract inputs for the last merge operation
            left_table_name = last_merge_block.get("left_table")
            right_table_name = last_merge_block.get("right_table")
            how = last_merge_block.get("how")
            final_merged_name = last_merge_block.get("merged_name") # This is the output name of the last merge

            if not left_table_name or not right_table_name:
                raise ValueError("Left or Right table name is missing for the last merge operation.")
            if left_table_name not in all_available_datasets:
                raise ValueError(f"Left table '{left_table_name}' for the last merge operation not found in available datasets.")
            if right_table_name not in all_available_datasets:
                raise ValueError(f"Right table '{right_table_name}' for the last merge operation not found in available datasets.")

            df_left = all_available_datasets[left_table_name]
            df_right = all_available_datasets[right_table_name]

            # Dynamically build merge parameters to avoid passing conflicting None values
            merge_params = {
                "df_left": df_left,
                "df_right": df_right,
                "how": how,
                "suffixes": ("_x", "_y") # Assuming default suffixes are still desired
            }

            # Only add 'on' if it's explicitly set (not None and not empty list)
            on_val = last_merge_block.get("on")
            if on_val and (isinstance(on_val, str) or (isinstance(on_val, list) and len(on_val) > 0)):
                merge_params["on"] = on_val
            else:
                # If 'on' is not used, check for 'left_on' and 'right_on'
                left_on_val = last_merge_block.get("left_on")
                right_on_val = last_merge_block.get("right_on")
                if (left_on_val and (isinstance(left_on_val, str) or (isinstance(left_on_val, list) and len(left_on_val) > 0))) and \
                   (right_on_val and (isinstance(right_on_val, str) or (isinstance(right_on_val, list) and len(right_on_val) > 0))):
                    merge_params["left_on"] = left_on_val
                    merge_params["right_on"] = right_on_val
                else:
                    # If no valid merge keys are specified at all, raise an error
                    raise ValueError("Merge requires 'on' OR 'left_on' and 'right_on' to be specified for the last operation.")


            # Perform only the last merge operation using the new backend function
            final_merged_df = feat_engg_backend.apply_single_merge(**merge_params)

            # Store the result of this single last merge in model state's merged_tables
            if final_merged_name:
                model_state["merged_tables"][final_merged_name] = final_merged_df
                model_state["combined_dataset"] = final_merged_df
                model_state["merge_status_message"] = f"✅ Last merge operation completed successfully! Result saved as: '{final_merged_name}'."
                model_state["merge_status_type"] = "success"
            else:
                # If the last merge block has no output name, generate a temporary one
                temp_merged_name = f"Last_Merge_Result_{uuid.uuid4().hex[:8]}" # Use UUID for uniqueness
                model_state["merged_tables"][temp_merged_name] = final_merged_df
                model_state["combined_dataset"] = final_merged_df
                model_state["merge_status_message"] = f"⚠️ Last merge block had no output name. Result saved as a temporary table: '{temp_merged_name}'."
                model_state["merge_status_type"] = "warning"

            # No st.rerun() here. The display will happen outside this button block.
            st.session_state[f"{active_model}_state"] = model_state # Save state immediately
        except ValueError as ve:
            model_state["merge_status_message"] = f"❌ Merge configuration error: {ve}"
            model_state["merge_status_type"] = "error"
            model_state["combined_dataset"] = None # Clear combined dataset on error
            st.session_state[f"{active_model}_operations_complete"]["merge"] = False
            st.session_state[f"{active_model}_state"] = model_state # Update session state
        except Exception as e:
            model_state["merge_status_message"] = f"❌ Error during merge operations: {str(e)}"
            model_state["merge_status_type"] = "error"
            model_state["combined_dataset"] = None # Clear combined dataset on error
            st.session_state[f"{active_model}_operations_complete"]["merge"] = False
            st.session_state[f"{active_model}_state"] = model_state # Update session state

    # --- Display Merge Status and Preview (Outside the button's if block) ---
    if model_state.get("merge_status_message"):
        if model_state["merge_status_type"] == "success":
            st.success(model_state["merge_status_message"])
        elif model_state["merge_status_type"] == "warning":
            st.warning(model_state["merge_status_message"])
        elif model_state["merge_status_type"] == "error":
            st.error(model_state["merge_status_message"])

        final_merged_df_to_display = model_state.get("combined_dataset")
        if isinstance(final_merged_df_to_display, pd.DataFrame) and not final_merged_df_to_display.empty:
            with st.expander("📂 Preview of Merged Table", expanded=False):
                st.markdown("#### Preview of Merged Table")
                st.write(f"Shape of the last merged dataset: {final_merged_df_to_display.shape}") # Debug print
                st.dataframe(final_merged_df_to_display.head(10), use_container_width=True, hide_index=True)
        elif model_state["merge_status_type"] != "error": # Only show info if not already an error message
            st.info("The last merged dataset is empty. Please check your merge configuration.")

        # Clear the message after displaying to avoid persistence across unrelated reruns
        # This is important for new operations to show fresh status
        model_state["merge_status_message"] = None
        model_state["merge_status_type"] = None
        st.session_state[f"{active_model}_state"] = model_state # Update session state after clearing

st.markdown("---")  # Separator for clarity after merge section

############################################################################################################################################3

# --- Recommend Features Button ---
if 'active_model' not in st.session_state:
    st.session_state.active_model = "Forecast_Model"
active_model = st.session_state.active_model

# Initialize session state dictionaries if not present
if f"{active_model}_operations_complete" not in st.session_state:
    st.session_state[f"{active_model}_operations_complete"] = {}
if f"{active_model}_state" not in st.session_state:
    st.session_state[f"{active_model}_state"] = {}

# Initialize per-model accept flags
if f"{active_model}_accept_ai_done" not in st.session_state:
    st.session_state[f"{active_model}_accept_ai_done"] = False
if f"{active_model}_accept_ai_success" not in st.session_state:
    st.session_state[f"{active_model}_accept_ai_success"] = False
##############################################################################  
# --- Recommend Features Button ---
st.markdown("## AI Generated Features that are recommended to be used in the model.")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("✨ Recommend Features (AI)", key="recommend_features_ai", use_container_width=True):
        if st.session_state.get(f"{active_model}_accept_ai_done", False):
            st.info("AI recommendations have already been accepted for this model. Skipping LLM call.")
        elif not GEMINI_API_KEY_CONFIGURED:
            st.error("Gemini API Key is not configured. Cannot recommend features. Please check `genai_utils.py` and your .env file.")
        else:
            try:
                with st.spinner("🧠 Calling Generative AI for feature recommendations..."):
                    current_dataset = model_state.get("combined_dataset", pd.DataFrame())

                    if current_dataset is None or not isinstance(current_dataset, pd.DataFrame) or current_dataset.empty:
                        st.error("The 'merged_dataset' is empty or not a DataFrame. Cannot generate recommendations.")
                    else:
                        dataset_description = summarize_dataset_columns(current_dataset.head())  # Use head() for brevity
                        
                        st.markdown("#### Dataset Summary Sent to AI:")
                        st.text_area("Input to AI", dataset_description, height=100, disabled=True)

                        recommendations_text = get_recommended_features_gemini(dataset_description)

                        if recommendations_text.startswith("Error:") or "An error occurred" in recommendations_text:
                            st.error(f"Failed to get recommendations: {recommendations_text}")
                        else:
                            st.markdown("#### Raw AI Response:")
                            st.text_area("Gemini Raw Output", recommendations_text, height=150, disabled=True)
                            
                            recommended_features_df = parse_gemini_recommendations(recommendations_text)

                            if not recommended_features_df.empty:
                                st.session_state.feature_info = recommended_features_df
                                st.session_state.recommended_features = recommended_features_df.copy()
                                
                                st.session_state[f"{active_model}_operations_complete"]["recommend"] = True
                                st.success("AI Recommended features generated!")
                                st.rerun()
                            else:
                                st.warning("AI recommendations received, but no features could be parsed or extracted.")
            except Exception as e:
                st.error(f"Error during AI feature recommendation process: {str(e)}")

# Display recommended features if available
if st.session_state.get(f"{active_model}_operations_complete", {}).get("recommend", False) and \
   hasattr(st.session_state, 'feature_info') and \
   isinstance(st.session_state.feature_info, pd.DataFrame) and \
   not st.session_state.feature_info.empty:

    with st.expander("💡 AI Recommended Engineered Features", expanded=False):

        display_df = st.session_state.feature_info.copy()
        expected_cols_display = ["Feature", "Description", "Primary Event Impact", "Data Type", "Derivation", "Justification"]
        for col in expected_cols_display:
            if col not in display_df.columns:
                display_df[col] = "N/A"

        column_config = {
            "Feature": st.column_config.TextColumn("Feature 💡", width="medium", disabled=True),
            "Description": st.column_config.TextColumn("Full Description 📝", width="large", disabled=True),
            "Primary Event Impact": st.column_config.TextColumn("Primary Impact 🎯", width="medium", disabled=True),
            "Data Type": st.column_config.TextColumn("Data Type", width="small", disabled=True),
        }

        st.data_editor(
            display_df[["Feature", "Description", "Primary Event Impact", "Data Type"]],
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            key="recommended_features_ai_editor"
        )

        st.markdown("---")
        st.markdown("#### 🧪 Full Derivations and Justifications")
        st.dataframe(
            display_df[["Feature", "Derivation", "Justification"]],
            use_container_width=True
        )
# "Min": st.column_config.TextColumn("Min", width="small", disabled=True),
# "Max": st.column_config.TextColumn("Max", width="small", disabled=True),
# "Mean": st.column_config.TextColumn("Mean", width="small", disabled=True),
# Apply recommended features to dataset if not already accepted


if st.session_state.get(f"{active_model}_operations_complete", {}).get("recommend", False) and \
   hasattr(st.session_state, 'recommended_features') and \
   isinstance(st.session_state.recommended_features, pd.DataFrame) and \
   not st.session_state.recommended_features.empty and \
   not st.session_state.get(f"{active_model}_accept_ai_done", False):

    current_dataset = model_state.get("combined_dataset", pd.DataFrame())
    recommended_features = st.session_state.recommended_features

    if current_dataset.empty:
        st.warning("The merged dataset is empty. Cannot execute recommended features.")
    else:
        try:
            recommended_features_records = recommended_features.to_dict(orient="records")
            updated_dataset,output_dict = apply_recommended_features(current_dataset, recommended_features_records)
            retry_no = 0
            max_retries = 4

            while retry_no < max_retries:
                res_error = output_dict['res_error']

                print("RETRY NO : ",f"{retry_no}")

                if not res_error:
                    break

                retry_no += 1
                updated_dataset,output_dict = apply_recommended_features(current_dataset, recommended_features_records)
            
        
            # Save updated dataset in session state
            st.session_state[f"{active_model}_updated_dataset"] = updated_dataset

            # Save in model_state
            model_state = st.session_state[f"{active_model}_state"]
            model_state["updated_dataset"] = updated_dataset.copy()

            # Mark acceptance done
            st.session_state[f"{active_model}_operations_complete"]["accept_ai"] = True
            st.session_state[f"{active_model}_accept_ai_success"] = True
            st.session_state[f"{active_model}_accept_ai_done"] = True

            st.rerun()
        except Exception as e:
            st.error(f"Error accepting AI recommended features: {str(e)}")

        

if st.session_state.get(f"{active_model}_operations_complete", {}).get("recommend", False) and \
   hasattr(st.session_state, 'recommended_features') and \
   isinstance(st.session_state.recommended_features, pd.DataFrame) and \
   not st.session_state.recommended_features.empty and \
   not st.session_state.get(f"{active_model}_accept_ai_done", False):

    col1_accept, col2_accept, col3_accept = st.columns([1, 2, 1])
    with col2_accept:
        if st.button("✅ Accept AI Recommended Features", key="accept_recommended_features_ai", use_container_width=True):
            try:
                updated_dataset = st.session_state.get(f"{active_model}_updated_dataset", pd.DataFrame())

                if updated_dataset.empty:
                    st.warning("Updated dataset is empty. Cannot save.")
                else:
                    model_state = st.session_state[f"{active_model}_state"]
                    model_state["updated_dataset"] = updated_dataset.copy()

                    st.session_state[f"{active_model}_operations_complete"]["accept_ai"] = True
                    st.session_state[f"{active_model}_accept_ai_success"] = True
                    st.session_state[f"{active_model}_accept_ai_done"] = True
                    st.rerun()
            except Exception as e:
                st.error(f"Error accepting AI recommended features: {str(e)}")

# Display success message
if st.session_state.get(f"{active_model}_accept_ai_success", False):
    st.success("✅ AI recommended features have been accepted and saved!")
    st.session_state[f"{active_model}_accept_ai_success"] = False



# Always display updated dataset if available
updated_dataset = st.session_state.get(f"{active_model}_updated_dataset", pd.DataFrame())
if isinstance(updated_dataset, pd.DataFrame) and not updated_dataset.empty:
    with st.expander("📂 View Updated Dataset with Recommended Features", expanded=False):
        st.markdown("### Updated Dataset")
        st.write(f"Shape: {updated_dataset.shape}")
        st.dataframe(updated_dataset.head(), use_container_width=True)

st.markdown("---")

############# --- Data Transformation Buttons ---###############
st.markdown("## 🔧 Feature Transformation")
st.markdown("##### Customize the features based on the business requirements.")

# Create a centered container for the buttons
col1, col2, col3 = st.columns([1, 2, 1])  # Unequal columns to center the buttons
with col2:  # Middle column
    if st.button(" Single Feature Transformation", key="transform_btn", use_container_width=True):
        model_state["show_popup1"] = True
        st.rerun()

# --- Popup 1: Single Feature Transformations ---
if model_state.get("show_popup1", False):
    st.markdown("### 🔧 Single Feature Transformation")

    # --- INPUT CHANGE: Explicitly mention the input dataset ---
    st.info("Applying single feature transformations to the **Recommended Features**.")

    input_features_df = model_state.get("updated_dataset",pd.DataFrame())

    if input_features_df.empty:
        st.warning("No features available for single feature transformation. Please acknowledge recommended features first.")
        input_features = []
    else:
        input_features = input_features_df.columns.tolist()


    # Initialize transform blocks if empty
    if not model_state["transform_blocks"]:
        model_state["transform_blocks"] = [{
            "feature": input_features[0] if input_features else "",
            "operation": "Addition", # Default operation
            "value": 1.0, # Default value
            "output_name": ""
        }]

    # Show transformation blocks
    # Fetch operations from backend
    all_single_operations = feat_engg_backend.get_single_feature_transform_operations()

    for i, block in enumerate(model_state["transform_blocks"]):
        st.markdown(f"**Transformation #{i+1}**")
        col1, col2, col3, col4, col5 = st.columns([0.5, 2, 2, 2, 2])

        with col1:
            if st.button("❌", key=f"remove_single_{i}"):
                model_state["transform_blocks"].pop(i)
                st.rerun()

        with col2:
            # Only show feature selection if we have features
            if input_features:
                feature = st.selectbox(
                    "Select Feature",
                    input_features,
                    key=f"single_feature_{i}",
                    index=input_features.index(block.get("feature", input_features[0])) if block.get("feature") in input_features else 0
                )
            else:
                # If no features, default to None or an empty string, warning is already above
                feature = None

        with col3:
            operation = st.selectbox(
                "Operation",
                # --- Changed: Use backend function to get operations ---
                all_single_operations,
                key=f"single_operation_{i}",
                index=all_single_operations.index(block.get("operation", "Addition")) if block.get("operation", "Addition") in all_single_operations else 0
            )

        with col4:
            # --- Changed: Added 'Rounding' to freeze_value_ops ---
            freeze_value_ops = ["Rename", "Log", "Square Root", "Absolute Value", "Rounding"]
            if operation in freeze_value_ops:
                default_val = 0 if operation == "Rename" else 1 # Rename could technically use a default of 0 or similar
                value = st.number_input(
                    "Value",
                    value=default_val,
                    key=f"single_value_{i}",
                    disabled=True # Value input disabled for these operations
                )
            elif operation in ["Addition", "Subtraction", "Multiplication", "Division", "Power"]:
                # Ensure value is float for arithmetic operations' default input
                value = st.number_input(
                    "Value",
                    value=float(block.get("value", 1.0)), # Cast to float for consistency
                    key=f"single_value_{i}"
                )
            else: # Fallback for any other operation type that might need a value
                value = st.number_input(
                    "Value",
                    value=float(block.get("value", 1.0)), # Cast to float
                    key=f"single_value_{i}"
                )


        with col5:
            if feature:
                # --- Changed: Added 'Rounding' to output name suggestion logic ---
                if operation == "Rename":
                    suggested_output = f"{feature}_renamed"
                elif operation == "Log":
                    suggested_output = f"{feature}_log"
                elif operation == "Square Root":
                    suggested_output = f"{feature}_sqrt"
                elif operation == "Absolute Value":
                    suggested_output = f"{feature}_abs"
                elif operation == "Rounding":
                    suggested_output = f"{feature}_rounded"
                elif value is not None:
                    # Generic suggestion for operations with a numerical value
                    op_symbol = {
                        "Addition": "_plus_", "Subtraction": "_minus_",
                        "Multiplication": "_mult_", "Division": "_div_",
                        "Power": "_pow_"
                    }.get(operation, "_")
                    suggested_output = f"{feature}{op_symbol}{str(value).replace('.', '')}"
                else: # Fallback if no specific suggestion can be made
                    suggested_output = f"{feature}_{operation.replace(' ', '').lower()}"

                prev_suggestion = block.get("prev_suggestion", "")
                prev_output_name = block.get("output_name", "")

                # Logic to retain user's custom output name unless it matches a previous suggestion
                if not prev_output_name or prev_output_name == prev_suggestion:
                    output_name = suggested_output
                else:
                    output_name = prev_output_name

                output_name = st.text_input(
                    "Output Feature",
                    value=output_name,
                    key=f"single_output_{i}"
                )

                # Update the model_state with current block's configuration
                model_state["transform_blocks"][i] = {
                    "feature": feature,
                    "operation": operation,
                    "value": value,
                    "output_name": output_name,
                    "prev_suggestion": suggested_output # Store the current suggestion to compare next time
                }

    if st.button("➕ Add Transformation", key="add_transform"):
        if input_features:
            model_state["transform_blocks"].append({
                "feature": input_features[0],
                "operation": "Addition",
                "value": 1.0,
                "output_name": ""
            })
            st.rerun()
        else:
            st.warning("Please ensure data is available before adding transformations.")

    # --- Apply Transformation Button ---
    if st.button("✅ Apply Transformation", key="apply_single_transform"):
        try:
            if input_features_df.empty:
                st.warning("No features available. Please ensure data is available.")
            else:
                transformed_features = {}
                successful_transformations = []
                # Get the current state of the input features for sequential application
                current_transformed_df = input_features_df.copy()

                for block in model_state["transform_blocks"]:
                    feature = block["feature"]
                    operation = block["operation"]
                    output_name = block["output_name"]

                    if not feature or not operation or not output_name:
                        st.error(f"Skipping incomplete transformation block: {block}")
                        continue # Skip to the next block if essential fields are missing

                    # Apply transformation using the backend function
                    try:
                         current_transformed_df = feat_engg_backend.apply_single_feature_transform(
                             current_transformed_df,
                             block
                         )
                         successful_transformations.append(f"- Applied transformation '{operation}' on '{feature}' to create '{output_name}'")
                    except (ValueError, TypeError, Exception) as e:
                         st.error(f"Error applying transformation block for '{feature}' with operation '{operation}': {e}")
                         # Do not continue processing if a block fails, to prevent cascading errors
                         break # Stop processing further blocks on first error

                # Update session state with the new DataFrame containing transformed features
                # Only update if all blocks processed without a fatal error (no break occurred)
                if not successful_transformations and not model_state["transform_blocks"]:
                    # If no blocks were processed and there were no blocks, maybe a message
                    pass
                elif len(successful_transformations) == len(model_state["transform_blocks"]):
                    model_state["single_transformed_features"] = current_transformed_df # Update the single_transformed_features
                    st.session_state.single_transform_success = "✅ Single feature transformations applied successfully!"
                    # Display list of successful transformations
                    st.info("Successful transformations:\n" + "\n".join(successful_transformations))
                elif successful_transformations: # Some blocks worked, but not all (due to 'break' above)
                    model_state["single_transformed_features"] = current_transformed_df # Update with partially transformed data
                    st.session_state.single_transform_success = "⚠️ Some single feature transformations applied. Check errors above for skipped transformations."
                    st.info("Successful transformations:\n" + "\n".join(successful_transformations))
                else: # No successful transformations
                    st.session_state.single_transform_success = "❌ No single feature transformations were applied due to errors."


                # Clear the transform blocks after successful application (or partial application if errors occurred)
                # This ensures the UI resets for the next set of transformations
                model_state["transform_blocks"] = []
                st.rerun()
        except Exception as e:
            # Catching any remaining unexpected errors from the entire 'Apply Transformation' process
            st.error(f"An unexpected error occurred during the application of transformations: {str(e)}")


    # Display success message if it exists (from previous rerun)
    if st.session_state.get("single_transform_success"):
        if "✅" in st.session_state.single_transform_success:
            st.success(st.session_state.single_transform_success)
        elif "⚠️" in st.session_state.single_transform_success:
            st.warning(st.session_state.single_transform_success)
        elif "❌" in st.session_state.single_transform_success:
            st.error(st.session_state.single_transform_success)

        st.session_state.single_transform_success = None # Clear after displaying



##########################################################################################################################################################

# --- Multi-Feature Transformation Section ---
st.markdown("###  Multiple Features Transformation")

# IMPORTANT: Ensure active_model and model_state are defined here or globally
if "active_model" not in st.session_state:
    st.info("Please select or initialize a model first to use Multi-Feature Transformation.")
    st.stop() # Stops execution of the rest of the script if no active model

active_model = st.session_state.active_model
model_state = st.session_state[f"{active_model}_state"]

# Get the current DataFrame for this section from the model's state.
# This now comes from single_transformed_features
current_df_for_multi_feature = model_state.get("single_transformed_features", pd.DataFrame())

# --- IMPORTANT: Early Exit if no data is available ---
if current_df_for_multi_feature.empty:
    st.info("No data available for Multi-Feature Transformation. Please ensure you have completed single feature transformation steps.")
    st.stop() # Stops execution of the rest of this section if no data


# Initialize multi transform blocks if empty
# This provides a default empty block if the list was cleared or never initialized for this model.
if not model_state["multi_transform_blocks"]:
    model_state["multi_transform_blocks"] = [{
        "features": [],
        "operation": "",
        "output_name": ""
    }]

# Show transformation blocks
for i, block in enumerate(model_state["multi_transform_blocks"]):
    st.markdown(f"**Transformation #{i+1}**")
    col1, col2, col3, col4 = st.columns([0.5, 2, 2, 2])

    with col1:
        # Corrected key for remove button
        if st.button("❌", key=f"remove_multi_block_{active_model}_{i}"):
            model_state["multi_transform_blocks"].pop(i)
            st.rerun()

    with col2:
        available_features = feat_engg_backend.get_features_for_table("data", {"data": current_df_for_multi_feature})
        if available_features:
            selected_features = st.multiselect(
                "Choose Features to Combine",
                available_features,
                default=block.get("features", []),
                key=f"multi_features_select_{active_model}_{i}" # Explicit key
            )
            model_state["multi_transform_blocks"][i]["features"] = selected_features
        else:
            st.warning("No features available in the current dataset. Please check data loading and previous steps.")

    with col3:
        # The AI backend expects a free-form text description of the operation.
        operation_text = st.text_input(
             "Describe the transformation (AI will interpret)",
             value=block.get("operation", ""),
             placeholder="e.g.,'ratio of col_A to col_B'",
             key=f"multi_operation_text_{active_model}_{i}" # Explicit key
        )
        model_state["multi_transform_blocks"][i]["operation"] = operation_text

    with col4:
        output_name = st.text_input(
            "Name for New Feature",
            value=block.get("output_name", ""),
            key=f"multi_output_name_{active_model}_{i}" # Explicit key
            )   
        model_state["multi_transform_blocks"][i]["output_name"] = output_name


if st.button("➕ Add New Feature Combination", key=f"add_multi_transform_button_{active_model}"): # Explicit key
    if not current_df_for_multi_feature.empty:
        model_state["multi_transform_blocks"].append({
            "features": [],
            "operation": "",
            "output_name": ""
        })
        st.rerun()
    else:
        st.warning("Please ensure data is loaded and available before adding transformations.")

# --- Apply All Multi-Feature Transformations Button ---

if st.button("✅ Apply all transformations", key=f"apply_multi_transforms_button_{active_model}"): # Explicit key
    try:
        if not model_state["multi_transform_blocks"]:
            st.warning("No multi-feature transformation blocks defined. Please add at least one.")
            # No st.stop() here, allow warning to display
        else:
            # Call the new backend function for AI-driven multi-feature transformations
            final_transformed_df = feat_engg_backend.apply_all_ai_driven_multi_feature_transforms(
                current_df_for_multi_feature.copy(), # Pass a copy of the input DataFrame
                model_state["multi_transform_blocks"] # Pass the entire list of transformation blocks
            )

            # Store the result in model_state["final_transformed_features"]
            model_state["final_transformed_features"] = final_transformed_df

            # Update success message for the model state (handled by the display block below)
            model_state["multi_transform_success"] = "✅ Multi-feature transformations applied successfully!"

            # Optional: Save the updated features to a CSV file (use final_transformed_features)
            # This is a good place if you want to persist the output of multi-feature transformations.
            # combined_dataset_file = os.path.join("data_registry", active_model, "transformed_features.parquet")
            # os.makedirs(os.path.dirname(combined_dataset_file), exist_ok=True)
            # model_state["final_transformed_features"].to_parquet(combined_dataset_file, index=False)
            # st.info(f"Transformed features saved to: {combined_dataset_file}")

            # Clear the transform blocks after successful application
            model_state["multi_transform_blocks"] = []
            st.rerun() # Rerun to clear blocks and update UI (this will also trigger the success message display)

    except Exception as e:
        st.error(f"An unexpected error occurred during multi-feature transformations: {str(e)}")
        # Update error message for the model state
        model_state["multi_transform_success"] = "❌ Multi-feature transformations failed."

# --- Display Success/Error Message for Multi-Feature Transformation ---
# This block should be placed directly after the "Apply all transformations" button section.
if model_state.get("multi_transform_success"): # Use .get() for safety
    if "✅" in model_state["multi_transform_success"]:
        st.success(model_state["multi_transform_success"])
    elif "❌" in model_state["multi_transform_success"]:
        st.error(model_state["multi_transform_success"])
    elif "⚠️" in model_state["multi_transform_success"]:
        st.warning(model_state["multi_transform_success"])

    # IMPORTANT: Clear the message after displaying it so it doesn't persist
    model_state["multi_transform_success"] = None




st.markdown("---")


###################################################################################################################################################

# --- Data Selection Section (Start of the requested section) ---

# --- Target Variable Selection ---
st.markdown("## 🎯 Target Variable Selection")
st.markdown("#### Preview the Feature that will store the value of output for selected model.")

# Hardcoded mapping for each model
MODEL_TARGET_MAP = {
    "Forecast Model": "Profitability_GBP",
    "Charge-Off Model": "COF_EVENT_LABEL",
    "Prepayment Model": "PREPAYMENT_EVENT_LABEL"
}

input_df_for_target = model_state.get("final_transformed_features", pd.DataFrame())
target_variable = MODEL_TARGET_MAP.get(active_model, "")

if not input_df_for_target.empty and target_variable:
    st.text_input(
    "Target Variable (auto-selected for this model)",
    value=target_variable,
    disabled=True,
    key=f"target_variable_placeholder_{active_model}"
)
    # Remove the target variable from the features for mandatory/good-to-have
    features_for_mandatory_df = input_df_for_target.drop(columns=[target_variable], errors='ignore').copy()
    model_state["features_for_mandatory"] = features_for_mandatory_df
    model_state["final_dataset"] = input_df_for_target.copy()
    model_state["target_column"] = target_variable
    model_state["target_feature"] = target_variable
else:
    st.info("Please complete multi-feature transformations first to select a target variable.")

st.markdown("---") # Separator after target selection


# --- Mandatory Features Section ---
st.subheader("📌 Mandatory Features")

combined_data_for_mandatory = model_state.get("features_for_mandatory", pd.DataFrame())

# Build a combined feature description map (metadata + AI)
combined_feature_desc_map = feature_desc_map.copy()
ai_feature_info_df = st.session_state.get('feature_info', pd.DataFrame())
if not ai_feature_info_df.empty and "Feature" in ai_feature_info_df.columns and "Description" in ai_feature_info_df.columns:
    for _, row in ai_feature_info_df.iterrows():
        feat = row["Feature"]
        desc = row["Description"]
        if pd.notna(feat) and pd.notna(desc):
            combined_feature_desc_map[feat] = desc

if combined_data_for_mandatory.empty:
    st.warning("No features available after target variable selection. Please ensure a target variable is selected.")
else:
    # Get mandatory features from backend
    mandatory_features = feat_engg_backend.select_mandatory_features(combined_data_for_mandatory)
    model_state["selected_mandatory_features"] = mandatory_features

    # Remove mandatory features from main dataset
    features_after_mandatory_df = combined_data_for_mandatory.drop(
        columns=[f for f in mandatory_features if f in combined_data_for_mandatory.columns],
        errors='ignore'
    ).copy()
    model_state["features_after_mandatory"] = features_after_mandatory_df

    # Define or expand feature descriptions
    feature_descriptions = {
        "OPB": "Outstanding Principal Balance of the customer's loan",
        "interest_rate": "Current interest rate applicable to the customer's loan",
        "tenure": "Duration of the loan in months",
        "credit_score_band": "Customer's credit score category (Excellent, Good, Fair, Poor)",
        "LTV": "Loan-to-Value ratio indicating the risk level of the loan",
        "loan_amount": "The total amount of the loan",
        "age": "Age of the customer",
        "employment_status": "Current employment status of the customer"
    }
    for feature in combined_data_for_mandatory.columns:
        feature_descriptions.setdefault(feature, "Transformed or engineered feature based on original data.")

    present_mandatory_features = [feat for feat in mandatory_features if feat in combined_data_for_mandatory.columns]

    if present_mandatory_features:
        mandatory_features_df = pd.DataFrame({
            "Mandatory Feature": present_mandatory_features,
            "Description": [combined_feature_desc_map.get(f, "No description available") for f in present_mandatory_features]
        })
        st.dataframe(mandatory_features_df, hide_index=True)
        # Check if all defined mandatory features are present (from the backend's selection)
        if len(present_mandatory_features) == len(mandatory_features):
            st.success("All mandatory attributes are available and selected by the model.")
        else:
            missing_mandatory = [feat for feat in mandatory_features if feat not in combined_data_for_mandatory.columns]
            st.warning(f"Some mandatory features selected by the model are missing: {', '.join(missing_mandatory)}")
    else:
        st.info("No mandatory features identified in the dataset.")

st.markdown("---")



# --- Good-to-Have Features Section ---
st.subheader("🧠 Optional AI-Recommended Features")

# Exclude mandatory features from optional list
available_optional_features = [
    feat for feat in combined_data_for_mandatory.columns
    if feat not in present_mandatory_features
]

checkbox_state_key = f"{active_model}_feature_checkboxes"
select_all_state_key = f"{active_model}_select_all_clicked"

# Initialize session state
if checkbox_state_key not in st.session_state:
    st.session_state[checkbox_state_key] = {}

if select_all_state_key not in st.session_state:
    st.session_state[select_all_state_key] = False

# Search functionality
search_query = st.text_input("🔍 Search Features (name or description)", value="", placeholder="e.g. age, purchase_count")

    # --- Select All Checkbox ---
select_all_key = f"{active_model}_select_all"
if select_all_key not in st.session_state:
    st.session_state[select_all_key] = False



# --- Build a combined feature description map ---
combined_feature_desc_map = feature_desc_map.copy()


# Add/override with AI-recommended feature descriptions if available
ai_feature_info_df = st.session_state.get('feature_info', pd.DataFrame())
if not ai_feature_info_df.empty and "Feature" in ai_feature_info_df.columns and "Description" in ai_feature_info_df.columns:
    for _, row in ai_feature_info_df.iterrows():
        feat = row["Feature"]
        desc = row["Description"]
        if pd.notna(feat) and pd.notna(desc):
            combined_feature_desc_map[feat] = desc

all_features_df = pd.DataFrame({
"Feature": available_optional_features,
"Description": [combined_feature_desc_map.get(feat, "No description available") for feat in available_optional_features],
})

if search_query:
    filtered_df = all_features_df[
        all_features_df["Feature"].str.contains(search_query, case=False, na=False) |
        all_features_df["Description"].str.contains(search_query, case=False, na=False)
    ]
else:
    filtered_df = all_features_df.copy()

# Select all checkbox
select_all_ui = st.checkbox("Select All Features", value=False, key="select_all_ui")

# Apply select all / deselect all
if select_all_ui and not st.session_state[select_all_state_key]:
    for feat in filtered_df["Feature"]:
        st.session_state[checkbox_state_key][feat] = True
    st.session_state[select_all_state_key] = True
    st.rerun()

elif not select_all_ui and st.session_state[select_all_state_key]:
    for feat in filtered_df["Feature"]:
        st.session_state[checkbox_state_key][feat] = False
    st.session_state[select_all_state_key] = False
    st.rerun()

# Reflect selection state
filtered_df["Select"] = [
    st.session_state[checkbox_state_key].get(feat, False) for feat in filtered_df["Feature"]
]
filtered_df["Select"] = filtered_df["Select"].astype(bool)  # <-- Ensure boolean type

# Display table
edited_df = st.data_editor(
    filtered_df,
    column_config={
        "Feature": st.column_config.TextColumn("Feature 🔍", width="medium", disabled=True),
        "Description": st.column_config.TextColumn("Description", width="large", disabled=True),
        "Select": st.column_config.CheckboxColumn("Select", width="small", help="Select this feature"),
    },
    hide_index=True,
    use_container_width=True,
    key=f"{active_model}_feature_editor"
)

# Update selections
for feature, selected in zip(edited_df["Feature"], edited_df["Select"]):
    st.session_state[checkbox_state_key][feature] = selected

# Save selected features globally
selected_features = [
    feat for feat, selected in st.session_state[checkbox_state_key].items()
    if selected and feat not in present_mandatory_features
]
st.session_state[f"{active_model}_selected_features"] = selected_features

st.markdown(f"✅ **{len(selected_features)} features selected.**")


if st.button("📊 Show Selected Attributes"):
    all_features_summary = []
    feature_types = []

    selected_mandatory_features = model_state.get("selected_mandatory_features", [])
    for feature in selected_mandatory_features:
        all_features_summary.append(feature)
        feature_types.append("Mandatory")

    for feature in st.session_state.get(f"{active_model}_selected_features", []):
        all_features_summary.append(feature)
        feature_types.append("Selected")

    original_df_for_final_dataset = model_state.get("final_transformed_features", pd.DataFrame()).copy()
    target_col_name = model_state.get("target_feature")

    if target_col_name and target_col_name in original_df_for_final_dataset.columns:
        all_features_summary.append(target_col_name)
        feature_types.append("Target")
    else:
        if not target_col_name:
            st.warning("No target variable selected. The final dataset will not include a target column.")
        else:
            st.warning(f"Selected target variable '{target_col_name}' not found in the transformed dataset.")

    features_to_include_in_final = [f for f in all_features_summary if f in original_df_for_final_dataset.columns]
    features_to_include_in_final = list(dict.fromkeys(features_to_include_in_final))  # Remove duplicates

    if all_features_summary:
        summary_df = pd.DataFrame({
            "Feature": all_features_summary,
            "Type": feature_types
        })

        st.subheader("Selected Features Summary")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        if features_to_include_in_final and not original_df_for_final_dataset.empty:
            model_state[f"final_dataset"] = original_df_for_final_dataset[features_to_include_in_final].copy()
        else:
            st.warning("No features selected to create the final dataset.")
            model_state[f"final_dataset"] = pd.DataFrame()


        # --- PARQUET FILE SAVING LOGIC (NOW HERE at the very end) ---
        final_dataset_with_target = model_state.get("final_dataset", pd.DataFrame())
        if not final_dataset_with_target.empty:
                # Create a subfolder in `data_registry` for the active model
                data_registry_subfolder = os.path.join("data_registry", active_model)
                os.makedirs(data_registry_subfolder, exist_ok=True)

                # Save the final dataset as a Parquet file
                final_dataset_path = os.path.join(data_registry_subfolder, f"{model_state[f'target_feature']}_final_dataset.parquet")
                final_dataset_with_target.to_parquet(final_dataset_path, index=False)

                # Save the file path in session state for use in the next page
                st.session_state[f"{active_model}_final_dataset_path"] = final_dataset_path

                st.success(f"Feature selection for '{active_model}' completed successfully!")
        else:
            st.warning("Final dataset is empty, cannot save Parquet file.")

        




# ####################################################################################################