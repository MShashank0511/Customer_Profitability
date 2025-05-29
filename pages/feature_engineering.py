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

try:
    from genai_recommend_features import (
        summarize_dataset_columns,
        get_recommended_features_gemini,
        parse_gemini_recommendations,
        apply_recommended_features,
        validate_code_snippet,
        sanitize_code_snippet,
        _dedent_code,
        GEMINI_API_KEY_CONFIGURED
    )
except ImportError:
    st.error("Failed to import 'genai_utils.py'. Make sure it's in the same directory or Python path.")
    # Stop execution if utils can't be imported
    GEMINI_API_KEY_CONFIGURED = False # Assume not configured
    st.stop()

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

on_us_data_path = st.session_state.get("on_us_data_path")
 
bureau_data_path = st.session_state.get("bureau_data_path")

installments_data_path = st.session_state.get("installments_data_path")

print("ENTERED FEATURE ENGINEERING PAGE")

print(on_us_data_path,bureau_data_path,installments_data_path)
print(f"Attempting to load Bureau Data from: {bureau_data_path}")
try:
    bureau_df = pd.read_parquet(bureau_data_path)
    print(f"Successfully loaded Bureau Data. Shape: {bureau_df.shape}")
except Exception as e:
    print(f"Error loading Bureau Data from '{bureau_data_path}': {e}")

print(f"Attempting to load On-Us Data from: {on_us_data_path}")
try:
    on_us_df = pd.read_parquet(on_us_data_path)
    print(f"Successfully loaded On-Us Data. Shape: {on_us_df.shape}")
except Exception as e:
    print(f"Error loading On-Us Data from '{on_us_data_path}': {e}")

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
#     data_path = st.session_state.get("on_us_data_path")  # Access from session_state

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
MODEL_NAMES = ["Default Model", "Charge-Off Model", "Prepayment Model", "Churn Model", "Extrapolation Model"]

def initialize_new_model_state(model_name):
    """Initialize a fresh state for a new model."""
    # Ensure bureau_df, on_us_df, and installments_df are defined or passed.
    # For demonstration, let's assume they are globally available or passed as arguments.
    # In a real Streamlit app, these would typically come from user uploads or a data loading utility.
    try:
        # These dataframes should be available in the Streamlit session state
        # if they have been loaded previously by the user.
        # If not, provide empty dataframes to avoid errors.
        initial_loan_data = pd.read_csv("loan_data.csv") 
        initial_bureau_data = bureau_df.copy()
        initial_on_us_data = on_us_df.copy()
        initial_installments_data = installments_df.copy()
    except AttributeError:
        # Handle cases where session_state might not have these attributes yet
        initial_loan_data = pd.DataFrame()
        initial_bureau_data = pd.DataFrame()
        initial_on_us_data = pd.DataFrame()
        initial_installments_data = pd.DataFrame()


    # Initialize raw_datasets and filtered_datasets dictionaries
    raw_datasets = {
        "Loan Data": initial_loan_data.copy(),
        "Bureau Data": initial_bureau_data.copy(),
        "On-Us Data": initial_on_us_data.copy(),
        "Applications Data": initial_installments_data.copy(),
    }
    filtered_datasets = {}

    # Initialize main state for the model
    st.session_state[f"{model_name}_state"] = {
        "raw_datasets": raw_datasets,
        "filtered_datasets": filtered_datasets,
        "loan_data": raw_datasets["Loan Data"], # Point to the raw_datasets version
        "bureau_data": raw_datasets["Bureau Data"], # Point to the raw_datasets version
        "onus_data": raw_datasets["On-Us Data"], # Point to the raw_datasets version
        "installments_data": raw_datasets["Applications Data"], # Point to the raw_datasets version
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
            "right_table": "On-Us Data",
            "how": "inner",
            "on": [],
            "left_on": [],
            "right_on": [],
            "merged_name": "Merged_1",
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

    try:
        with open(file_path, "w") as f:
            json.dump(state_to_save, f, indent=4)
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
        "On-Us Data": st.session_state.get("uploaded_on_us_data", pd.DataFrame()),
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
                            sub_key in ["loan_data", "bureau_data", "onus_data", "installments_data",
                                        "filtered_data_loan", "filtered_data_bureau", "filtered_data_onus",
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
                        st.session_state[session_key]["onus_data"] = st.session_state[session_key]["raw_datasets"].get("On-Us Data", pd.DataFrame())
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

        st.rerun()
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

        st.rerun()
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
    st.rerun()


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
            st.rerun()
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
    st.rerun()

# Streamlit Selectbox for model selection
selected_model = st.sidebar.selectbox(
    "Select Model",
    MODEL_NAMES,
    index=current_active_model_index,
    key="selected_model_dropdown_value", # Unique key for the selectbox value
    on_change=on_model_select_callback,
)

# Display the current active model at the top of the main content area
st.markdown(f"### Current Model: {st.session_state.active_model}")

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
onus_name = "On-Us Data"
installments_name = "Applications Data"

# Store the names and dataframes in a dictionary for easy access
dataset_mapping = {
    bureau_name: model_state["bureau_data"],
    onus_name: model_state["onus_data"],
    installments_name: model_state["installments_data"],
}

# Initialize visibility state for each dataset
if "show_bureau_data" not in st.session_state:
    st.session_state.show_bureau_data = False
if "show_onus_data" not in st.session_state:
    st.session_state.show_onus_data = False
if "show_installments_data" not in st.session_state:
    st.session_state.show_installments_data = False

# Create buttons for each dataset
with col1:
    if st.button(bureau_name, key="bureau_data_button"):
        st.session_state.show_bureau_data = not st.session_state.show_bureau_data

with col2:
    if st.button(onus_name, key="onus_data_button"):
        st.session_state.show_onus_data = not st.session_state.show_onus_data

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

if st.session_state.show_onus_data:
    st.subheader(f"{onus_name} Preview")
    onus_data = dataset_mapping[onus_name]
    if onus_data is not None and not onus_data.empty:
        st.dataframe(onus_data.head(), use_container_width=True)
    else:
        st.warning(f"{onus_name} is empty or not loaded.")

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
    onus_name: model_state["onus_data"],
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
    that the dataframes and session state variables like 'bureau_data', 'onus_data',
    'installments_data' and 'filter_blocks' are already initialized. It also assumes
    the dataset names are stored as: bureau_name, onus_name, installments_name.
    Restructured to arrange inputs in two rows per filter block.
    """
    # --- Add these lines here ---
    active_model = st.session_state.active_model
    model_state = st.session_state.get(f"{active_model}_state", {})
    raw_datasets = model_state.get("raw_datasets", {})
     # Get the raw datasets dictionary

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

    # --- Define available operations ---
    OPERATIONS = feat_engg_backend.get_filter_operations() # Get operations from backend



    # Use a container for the filter controls
    filter_container = st.container()

    # Create a mapping of dataset names to the actual DataFrames. Crucial for dynamic access.
    # This mapping is now raw_datasets
    dataset_mapping = raw_datasets # This is not directly used within this loop, but passed to backend functions

    with filter_container:
        for i, filter_block in enumerate(filter_blocks):
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
                # Select Table
                dataset_names = feat_engg_backend.get_table_names(raw_datasets)
                selected_dataset_name = st.selectbox("Select Table", dataset_names,
                                                     index=dataset_names.index(filter_block.get("dataset", dataset_names[0])) if filter_block.get("dataset") in dataset_names else 0,
                                                     key=f"dataset_{i}")
                # Update the filter block's dataset name immediately
                filter_blocks[i]["dataset"] = selected_dataset_name

            with row1_cols[2]:
                # Select Column
                selected_dataset_name = filter_blocks[i]["dataset"] # Use the potentially updated dataset name
                # Get the features from the selected DataFrame using feat_engg_backend
                available_features = feat_engg_backend.get_features_for_table(selected_dataset_name, raw_datasets)

                # Update the selected feature if the dataset changes and the previous feature is not in the new dataset
                current_feature = filter_block.get("feature")
                if current_feature not in available_features:
                    filter_blocks[i]["feature"] = available_features[0] if available_features else ""
                    current_feature = filter_blocks[i]["feature"] # Update current_feature

                selected_feature = st.selectbox("Select Column", available_features,
                                                 index=available_features.index(current_feature) if current_feature in available_features else 0,
                                                 key=f"feature_{i}")
                # Update the filter block's feature immediately
                filter_blocks[i]["feature"] = selected_feature

            # --- Row 2: Select Filter Type, Enter Value, Output Table ---
            # Using adjusted ratios for column widths
            row2_cols = st.columns([3, 3, 3])

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

                # Determine the default value based on the operation and current_value type
                default_value = None # Initialize default_value

                if operation in ["Greater Than", "Less Than", "Equal To", "Not Equal To", "Greater Than or Equal To", "Less Than or Equal To"]:
                    # Expected: single number
                    if isinstance(current_value, (int, float)):
                        default_value = current_value
                    else:
                        default_value = 0.0 # Default for number input

                    value = st.number_input("Enter Value to Filter By", value=default_value, key=f"value_{i}")

                elif operation == "Is In List":
                    # Expected: string (comma-separated) or list
                    if isinstance(current_value, str):
                        default_value = current_value
                    elif isinstance(current_value, list):
                         default_value = ','.join(map(str, current_value))
                    else:
                        default_value = '' # Default for text input

                    value = st.text_input("Enter values (comma-separated)", value=default_value, key=f"value_{i}")
                    # Store as string for now, parse when applying filters
                    # Parsing will happen in the "Apply All Filters" button logic

                elif operation == "Between":
                    # Expected: tuple or list of two numbers
                    default_value1 = 0.0
                    default_value2 = 0.0
                    if isinstance(current_value, (tuple, list)) and len(current_value) == 2:
                        if isinstance(current_value[0], (int, float)):
                            default_value1 = current_value[0]
                        if isinstance(current_value[1], (int, float)):
                            default_value2 = current_value[1]

                    col_val1, col_val2 = st.columns(2) # Nested columns for the two values in "Between"
                    with col_val1:
                        value1 = st.number_input("Start Value", value=default_value1, key=f"value_{i}_start")
                    with col_val2:
                        value2 = st.number_input("End Value", value=default_value2, key=f"value_{i}_end")
                    value = (value1, value2) # Store as a tuple

                elif operation in ["Is Null", "Is Not Null"]:
                    # No value needed
                    st.text_input("Value", value="N/A", key=f"value_{i}", disabled=True)
                    value = None # Store value as None

                elif operation == "Contains String":
                    # Expected: string
                    if isinstance(current_value, str):
                        default_value = current_value
                    else:
                        default_value = '' # Default for text input

                    value = st.text_input("Enter substring", value=default_value, key=f"value_{i}")

                else:
                    # Default input for any other operation (fallback)
                    if current_value is not None:
                        default_value = str(current_value) # Convert to string for text input
                    else:
                        default_value = ''

                    value = st.text_input("Select Value", value=default_value, key=f"value_{i}")

                # Update the filter block's value immediately
                filter_blocks[i]["value"] = value

            with row2_cols[2]:
                # Output Table
                selected_feature = filter_blocks[i]["feature"]
                operation = filter_blocks[i]["operation"]
                value = filter_blocks[i]["value"] # Get the updated value

                # Suggest a descriptive name for the filter condition (used for the boolean column name in backend)
                if selected_feature and operation:
                    suggested_name = f"{selected_feature}_{operation.replace(' ', '_').lower()}"
                    if value is not None and operation not in ["Is Null", "Is Not Null"]:
                         if isinstance(value, (tuple, list)):
                             # Sanitize tuple/list values for filename
                             value_str = '_'.join(map(lambda x: str(x).replace('.', '').replace('-', 'neg'), value))
                             suggested_name += f"_{value_str}"
                         elif isinstance(value, str):
                             # Sanitize string value for filename
                             value_str = value.replace(' ', '_').replace('.', '').replace('-', 'neg')
                             suggested_name += f"_{value_str}"
                         else:
                             # Sanitize single value
                             suggested_name += f"_{str(value).replace('.', '').replace('-', 'neg')}"

                    # Ensure the suggested name is not excessively long (optional but good practice)
                    suggested_name = suggested_name[:50] # Arbitrary length limit


                else:
                    suggested_name = ""

                # Use "Filter Description Name" for clarity in the UI
                filter_description_name = st.text_input(
                    "Output Table",
                    value=filter_blocks[i].get("output_name", suggested_name), # Use output_name from state or suggestion
                    key=f"filter_description_{i}",
                    help="A descriptive name for this filter condition."
                )
                # Update the filter block's output_name (used as the temporary boolean column name in backend)
                filter_blocks[i]["output_name"] = filter_description_name

            # No need for a consolidated update at the end of the block loop
            # because the block dictionary is updated immediately after each widget interaction,
            # and filter_blocks is already a reference to the list in session state.
            # The session state update happens implicitly when the block dict is modified.


        if st.button("➕ Add Filter"):
            active_model = st.session_state.active_model
            filter_blocks_state = st.session_state.get(f"{active_model}_filter_blocks", []) # Get from state
            raw_datasets = model_state.get("raw_datasets", {}) # Get raw datasets
            dataset_names = feat_engg_backend.get_table_names(raw_datasets) # Use feat_engg_backend to get dataset names
            filter_blocks_state.append({
                "dataset": dataset_names[0] if dataset_names else "", # Default dataset name
                "feature": "",
                "operation": feat_engg_backend.get_filter_operations()[0], # Default operation from feat_engg_backend
                "value": None, # Default value
                "output_name": ""
            })
            st.session_state[f"{active_model}_filter_blocks"] = filter_blocks_state # Explicitly update session state
            st.rerun()


        if st.button("Apply All Filters"):
            try:
                active_model = st.session_state.active_model
                model_state = st.session_state[f"{active_model}_state"]
                raw_datasets = model_state.get("raw_datasets", {})
                filter_blocks = st.session_state.get(f"{active_model}_filter_blocks", [])

                # Group filter blocks by the dataset they apply to
                filters_by_dataset = {}
                for block in filter_blocks:
                    dataset_name = block.get("dataset")
                    # Important: Parse the 'value' for 'Is In List' here before passing to backend
                    if block.get("operation") == "Is In List" and isinstance(block.get("value"), str):
                         block["value"] = [v.strip() for v in block["value"].split(',') if v.strip()] # Parse comma-separated string into list

                    if dataset_name:
                        if dataset_name not in filters_by_dataset:
                            filters_by_dataset[dataset_name] = []
                        filters_by_dataset[dataset_name].append(block)

                # Apply filters for each dataset and store the result in filtered_datasets
                model_state["filtered_datasets"] = {} # Clear previous filtered results

                if not filters_by_dataset:
                     st.warning("No filter blocks defined to apply.")
                else:
                    for dataset_name, blocks in filters_by_dataset.items():
                        if dataset_name in raw_datasets:
                            original_df = raw_datasets[dataset_name]
                            # Call the backend function to apply all filters for this table
                            filtered_df = feat_engg_backend.apply_all_filters_for_table(original_df, blocks) # Use feat_engg_backend

                            # Define the name for the resulting filtered table
                            filtered_table_name = f"{dataset_name}_Filtered"
                            # Store the filtered DataFrame in session state
                            model_state["filtered_datasets"][filtered_table_name] = filtered_df
                            st.success(f"Filters applied to '{dataset_name}'. Result stored as '{filtered_table_name}'.")
                        else:
                            st.error(f"Dataset '{dataset_name}' not found in raw datasets. Cannot apply filters.")

                # Optionally, display the resulting filtered datasets or a success message
                # You might want to add a section to display these filtered_datasets later.

                # --- Place the overall success message here ---
                if filters_by_dataset: # Only show overall success if filters were actually applied
                     st.success("✅ All filters applied successfully!")

                st.rerun() # Rerun to update the UI with the new state
            except Exception as e:
                st.error(f"Error applying filters: {str(e)}")

# --- Main Section ---
# Define a callback function to toggle the visibility of the filter section
def show_filter_data_callback():
    st.session_state.show_filter_data = not st.session_state.show_filter_data  # Toggle visibility

# Add the button with a callback
if "show_filter_data" not in st.session_state:
    st.session_state.show_filter_data = False

# Filter Data Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.button("🔄 Filter Data", key="show_filter_data_button", on_click=show_filter_data_callback, use_container_width=True)

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
        "right_table": "On-Us Data (Filtered)", # Default to the filtered data
        "how": "inner",
        "on": [],
        "left_on": [],
        "right_on": [],
        "merged_name": "Merged_1",
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
    st.button("🔄 Merge Datasets", key="merge_btn", on_click=show_merge_callback, use_container_width=True)


# Display the merge section if show_merge is True for the active model
active_model = st.session_state.active_model
model_state = st.session_state.get(f"{active_model}_state", {}) # Get model state

if model_state.get("show_merge", False): # Check the show_merge state within the model state

    st.header("Merge Datasets")

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

    if not table_names and merge_blocks:
         st.warning("No datasets available for merging. Please check your data loading and filtering steps.")
         # Optionally clear merge blocks if they refer to non-existent tables
         # model_state["merge_blocks"] = []
         # st.session_state[f"{active_model}_state"] = model_state
         # st.rerun() # Rerun if clearing blocks
         # return # Exit the merge section if no tables

    for i, block in enumerate(merge_blocks):
        st.markdown(f"---")  # Separator for clarity between merge blocks

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
            # Select columns present in both tables to join on ('on' parameter)
            # Ensure common_cols is not empty before displaying selectbox
            if common_cols:
                # Determine default index based on saved state
                current_on_value = block.get("on", []) # Get the saved value (expected to be a list)
                default_on_index = 0
                # Check if current_on_value is a non-empty list and its first element is in common_cols
                if current_on_value and isinstance(current_on_value, list) and current_on_value[0] in common_cols:
                     default_on_index = common_cols.index(current_on_value[0])

                on = st.selectbox(
                    "Select Column to Merge On",
                    common_cols,  # Options are common columns
                    index=default_on_index,
                    key=f"merge_on_{i}",
                    help="Select a single column present in both tables to join on."
                )
                # Update block immediately
                merge_blocks[i]["on"] = [on] # Store as a list
            else:
                # Warning and disabled input if no common columns
                st.warning(f"No common columns between '{left_table}' and '{right_table}' for 'on' join.")
                st.text_input("Column to Join On", value="No common columns", key=f"merge_on_{i}", disabled=True)
                merge_blocks[i]["on"] = [] # Store as empty list if no common columns


        with join_cols[1]:
            # Select columns from the left table to join on ('left_on' parameter)
            left_on = st.multiselect(
                "Select Columns from Left Table",
                left_cols,  # Options are columns from the left table
                default=block.get("left_on", []),  # Default to saved values
                key=f"merge_left_on_{i}",
                help="Select multiple columns from the left table to join on."
            )
            # Update block immediately
            merge_blocks[i]["left_on"] = left_on


        with join_cols[2]:
            # Select columns from the right table to join on ('right_on' parameter)
            right_on = st.multiselect(
                "Select Columns from Right Table",
                right_cols,  # Options are columns from the right table
                default=block.get("right_on", []),  # Default to saved values
                key=f"merge_right_on_{i}",
                help="Select multiple columns from the right table to join on."
            )
            # Update block immediately
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
                index=["inner", "left", "right", "outer", "cross"].index(block.get("how", "inner"))
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


    # --- Buttons to Manage Merge Operations (outside the loop) ---
    if st.button("➕ Add Merge Operation", key="add_merge"):
         active_model = st.session_state.active_model
         model_state = st.session_state[f"{active_model}_state"]
         raw_datasets = model_state.get("raw_datasets", {})
         filtered_datasets = model_state.get("filtered_datasets", {})
         merged_tables = model_state.get("merged_tables", {})

         # Get the names of all available tables for merging using the backend function
         available_table_names = feat_engg_backend.get_merge_available_table_names(raw_datasets, filtered_datasets, merged_tables)

         if not available_table_names:
              st.warning("No datasets available to start a new merge operation.")
              # No return needed here, just don't append the block
         else: # Add an 'else' block to contain the code that should only run if tables are available
              # Determine a sensible default for the left table (last merged or first available)
              # Use the last merged table name if available, otherwise the first available table name
              last_merged_name = list(merged_tables.keys())[-1] if merged_tables else (available_table_names[0] if available_table_names else "") # Fallback

              # Determine a sensible default for the right table
              default_right_table = available_table_names[0] if available_table_names else ""
              if default_right_table == last_merged_name and len(available_table_names) > 1:
                   # Find a different table if the first available is the same as the default left
                   try:
                       current_index = available_table_names.index(default_right_table)
                       if current_index + 1 < len(available_table_names):
                           default_right_table = available_table_names[current_index + 1]
                       elif current_index > 0: # If next index is out of bounds, try previous
                            default_right_table = available_table_names[current_index - 1]
                       else:
                           default_right_table = available_table_names[0] # Fallback to first if only one table
                   except ValueError:
                       pass # Should not happen if default_right_table is in the list


              # Generate a default name for the NEW block based on the chosen default tables and the *next* index
              next_block_index = len(merge_blocks) + 1
              # Using empty string as default value, so no generated name here
              # default_new_merged_name = f"{last_merged_name}_merged_{default_right_table}_{next_block_index}"


              merge_blocks.append({
                   "left_table": last_merged_name, # Default to the name of the last merged table or first available
                   "right_table": default_right_table, # Default to another available table
                   "how": "inner",
                   "on": [],
                   "left_on": [],
                   "right_on": [],
                   "merged_name": "", # Start with an empty output name by default
              })
              model_state["merge_blocks"] = merge_blocks # Update model_state
              st.session_state[f"{active_model}_state"] = model_state # Update session state
              st.rerun() # Rerun only if a new merge block was added


    # --- Apply Merge Operations Button ---
    if st.button("✅ Apply Merge Operations", key="execute_merges", use_container_width=True):
        try:
            active_model = st.session_state.active_model
            model_state = st.session_state[f"{active_model}_state"]

            # Prepare the dictionary of all available datasets for the backend
            # Include raw, filtered, and previously merged tables
            all_available_datasets = model_state.get("raw_datasets", {}).copy()
            all_available_datasets.update(model_state.get("filtered_datasets", {}))
            all_available_datasets.update(model_state.get("merged_tables", {}))


            if not all_available_datasets:
                 raise ValueError("No datasets available to perform merge operations.")

            merge_blocks = model_state.get("merge_blocks", []) # Get the current list of merge blocks

            if not merge_blocks:
                 raise ValueError("No merge operations defined to apply.")

            # Call the backend function to apply all merge blocks sequentially
            # Pass ALL available datasets as potential inputs
            all_merged_results = feat_engg_backend.apply_merge_blocks(
                all_available_datasets, # Pass all available datasets
                merge_blocks
            )

            # Store all resulting merged tables in model state
            model_state["merged_tables"] = all_merged_results

            # Identify the final merged dataset (the result of the last merge block)
            if merge_blocks:
                final_merged_name = merge_blocks[-1].get("merged_name")
                if final_merged_name and final_merged_name in all_merged_results:
                    model_state["combined_dataset"] = all_merged_results[final_merged_name].copy()
                    # merged_dataset = all_merged_results[final_merged_name].copy()
                    model_state["merged dataset"] = all_merged_results[final_merged_name].copy() # Store the final result
                    st.success(f"✅ All merge operations completed successfully! Final dataset: '{final_merged_name}'.")
                elif not final_merged_name and all_merged_results:
                     # If the last block had no output name, use the name from the last key in results
                     last_result_name = list(all_merged_results.keys())[-1]
                     model_state["combined_dataset"] = all_merged_results[last_result_name].copy()
                     st.warning(f"Last merge block had no output name. Final dataset is the result of the last operation: '{last_result_name}'.")
                else:
                    st.warning("Merge operations completed, but could not identify the final merged dataset.")
                    model_state["combined_dataset"] = None # Set combined_dataset to None on warning
            else:
                 st.warning("No merge operations were defined.")
                 model_state["combined_dataset"] = None # Set combined_dataset to None

            # Set merge operation as complete in model-specific state
            # model_state["operations_complete"]["merge"] = True
            st.session_state[f"{active_model}_state"] = model_state # Update session state

            # Clear the merge blocks after successful application (optional, depends on desired workflow)
            # model_state["merge_blocks"] = []
            # st.session_state[f"{active_model}_state"] = model_state # Update session state

            st.rerun() # Rerun to update the UI with the new state

        except ValueError as ve:
            st.error(f"Merge configuration error: {ve}")
            # Reset merge complete state on error in model-specific state
            st.session_state[f"{active_model}_operations_complete"]["merge"] = False
            st.session_state[f"{active_model}_state"] = model_state # Update session state

        except Exception as e:
            st.error(f"Error during merge operations: {str(e)}")
            # Reset merge complete state on error in model-specific state
            st.session_state[f"{active_model}_operations_complete"]["merge"] = False
            st.session_state[f"{active_model}_state"] = model_state # Update session state


st.markdown("---")

############################################################################################################################################3

# --- Recommend Features Button ---
if 'active_model' not in st.session_state:
    st.session_state.active_model = "default_model" 
active_model = st.session_state.active_model

# Initialize session state dictionaries if not present
if f"{active_model}_operations_complete" not in st.session_state:
    st.session_state[f"{active_model}_operations_complete"] = {}
if f"{active_model}_state" not in st.session_state: # This ensures model_state (the dictionary) itself exists
     st.session_state[f"{active_model}_state"] = {}

# --- Recommend Features Button ---
st.markdown("## AI-Powered Feature Recommendation")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("✨ Recommend Features (AI)", key="recommend_features_ai", use_container_width=True):
        if not GEMINI_API_KEY_CONFIGURED:
             st.error("Gemini API Key is not configured. Cannot recommend features. Please check `genai_utils.py` and your .env file.")
        else:
            try:
                with st.spinner("🧠 Calling Generative AI for feature recommendations..."):
                    current_dataset = model_state.get("merged dataset", pd.DataFrame())

                    if current_dataset is None or not isinstance(current_dataset, pd.DataFrame) or current_dataset.empty:
                        st.error("The 'merged_dataset' is empty or not a DataFrame. Cannot generate recommendations.")
                    else:
                        dataset_description = summarize_dataset_columns(current_dataset.head()) # Use head() for brevity
                        
                        st.markdown("#### Dataset Summary Sent to AI:")
                        st.text_area("Input to AI",dataset_description, height=100, disabled=True)

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
                                st.warning("AI recommendations received, but no features could be parsed or extracted. Please check the raw AI response above. The response might be empty or not in the expected format.")
            except Exception as e:
                st.error(f"Error during AI feature recommendation process: {str(e)}")
                # You might want to log the full traceback here for debugging
                # import traceback
                # st.text_area("Full Error Traceback", traceback.format_exc(), height=200)


# Display recommended features if they exist
# Display recommended features if they exist
if st.session_state.get(f"{active_model}_operations_complete", {}).get("recommend", False) and \
   hasattr(st.session_state, 'feature_info') and \
   isinstance(st.session_state.feature_info, pd.DataFrame) and \
   not st.session_state.feature_info.empty:
    st.markdown("### AI Recommended Engineered Features")
    
    display_df = st.session_state.feature_info.copy()
    # Ensure all expected columns exist for data_editor for consistent display
    expected_cols_display = ["Feature", "Description", "Primary Event Impact", "Min", "Max", "Mean", "Data Type", "Derivation", "Justification","Raw Features", "Code Snippet"]
    for col in expected_cols_display:
        if col not in display_df.columns:
            display_df[col] = "N/A" 

    # Convert any Series in column_config to strings or lists
    column_config = {
        "Feature": st.column_config.TextColumn(
            "Feature 💡", 
            width="medium", 
            disabled=True, 
            help=display_df["Feature"].to_list()  # Convert Series to list
        ),
        "Description": st.column_config.TextColumn(
            "Full Description 📝", 
            width="large", 
            disabled=True, 
            help=display_df["Description"].to_list()  # Convert Series to list
        ),
        "Primary Event Impact": st.column_config.TextColumn(
            "Primary Impact 🎯", 
            width="medium", 
            disabled=True, 
            help=display_df["Primary Event Impact"].to_list()  # Convert Series to list
        ),
        "Raw Features": st.column_config.TextColumn(
            "Raw Features", 
            width="medium",
            disabled=True,
            help=display_df["Raw Features"].to_list()  # Convert Series to list
        ),
        "Code Snippet": st.column_config.TextColumn(
            "Code Snippet",
            width="medium",
            disabled=True,
            help=display_df["Code Snippet"].to_list()  # Convert Series to list
        ),
        "Min": st.column_config.TextColumn("Min", width="small", disabled=True),
        "Max": st.column_config.TextColumn("Max", width="small", disabled=True),
        "Mean": st.column_config.TextColumn("Mean", width="small", disabled=True),
        "Data Type": st.column_config.TextColumn("Data Type", width="small", disabled=True),
    }

    st.data_editor(
        display_df[["Feature", "Description", "Primary Event Impact", "Min", "Max", "Mean", "Data Type","Raw Features", "Code Snippet"]],
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        key="recommended_features_ai_editor"
    )
    with st.expander("See Full Derivations and Justifications"):
        st.dataframe(display_df[["Feature", "Derivation", "Justification", "Primary Event Impact","Raw Features", "Code Snippet"]], use_container_width=True)

# Check if recommended features exist in session state
if st.session_state.get(f"{active_model}_operations_complete", {}).get("recommend", False) and \
   hasattr(st.session_state, 'recommended_features') and \
   isinstance(st.session_state.recommended_features, pd.DataFrame) and \
   not st.session_state.recommended_features.empty:
    
    # Get the current dataset and recommended features
    current_dataset = model_state.get("merged dataset", pd.DataFrame())
    recommended_features = st.session_state.recommended_features

    # Ensure the dataset is not empty
    if current_dataset.empty:
        st.warning("The merged dataset is empty. Cannot execute recommended features.")
    else:
        try:
            # Ensure recommended_features contains the required column
            if "Code Snippet" not in recommended_features.columns:
                raise ValueError("Recommended features DataFrame must contain a 'Code Snippet' column.")

            # Debugging: Display the generated code snippets
            for feature_info in recommended_features.to_dict(orient="records"):
                feature_name = feature_info.get("Feature")
                generated_code = feature_info.get("Code Snippet")
                st.text_area(f"Generated Code for '{feature_name}'", generated_code, height=150, disabled=True)

                # Extract the actual code between delimiters ''' if present
                if generated_code.startswith("'''") and generated_code.endswith("'''"):
                    extracted_code = generated_code.strip("'''")
                else:
                    extracted_code = generated_code  # Use the original code if no delimiters

                # Dedent the extracted code to fix indentation issues
                dedented_code = _dedent_code(extracted_code)

                # Execute the validated and dedented code snippet
                try:
                    execution_context = {'df': current_dataset, 'np': np, 'pd': pd}  # Execution context
                    exec(dedented_code, {}, execution_context)

                    # After exec, update current_dataset from execution_context if modified
                    if 'df' in execution_context:
                        current_dataset = execution_context['df']

                    st.success(f"Successfully created feature '{feature_name}'.")
                except Exception as e:
                    st.error(f"Failed to create feature '{feature_name}'. Error: {e}")

            # Execute the function to apply recommended features (if this is your custom function)
            updated_dataset = apply_recommended_features(current_dataset, recommended_features.to_dict(orient="records"))

            # Store the updated dataset in session state for persistence
            st.session_state[f"{active_model}_updated_dataset"] = updated_dataset

            # Display the updated dataset as a collapsible section
            with st.expander("📂 View Updated Dataset with Recommended Features", expanded=False):
                st.markdown("### Updated Dataset")
                st.write(f"Shape: {updated_dataset.shape}")
                st.dataframe(updated_dataset.head(), use_container_width=True)

        except Exception as e:
            st.error(f"Error applying recommended features: {str(e)}")
else:
    st.info("No recommended features available. Please complete the recommendation process first.")
    
#Accept Recommended Features Button ---
if st.session_state.get(f"{active_model}_operations_complete", {}).get("recommend", False) and \
   hasattr(st.session_state, 'recommended_features') and \
   isinstance(st.session_state.recommended_features, pd.DataFrame) and \
   not st.session_state.recommended_features.empty:
    col1_accept, col2_accept, col3_accept = st.columns([1, 2, 1])
    with col2_accept:
        if st.button("✅ Accept AI Recommended Features", key="accept_recommended_features_ai", use_container_width=True):
            try:
                recommended_features_descriptions = st.session_state.recommended_features

                # Store in a more descriptive session state key for clarity
                st.session_state.final_ai_engineered_features_descriptions = recommended_features_descriptions.copy()

                csv_filename = "ai_recommended_engineered_features.csv"
                recommended_features_descriptions.to_csv(csv_filename, index=False)
                st.info(f"AI recommended feature descriptions saved to '{csv_filename}'")

                model_state = st.session_state[f"{active_model}_state"]
                model_state["ai_recommended_features_descriptions"] = recommended_features_descriptions.copy()
                
                st.session_state[f"{active_model}_operations_complete"]["accept_ai"] = True
                st.session_state.accept_ai_success = True
                st.rerun()
            except Exception as e:
                st.error(f"Error accepting AI recommended features: {str(e)}")

# Display success message
if st.session_state.get("accept_ai_success", False):
    st.success("✅ AI recommended features (descriptions) have been accepted and their definitions saved!")
    st.session_state.accept_ai_success = False

st.markdown("---")


# --- Data Transformation Buttons ---
st.subheader("Data Actions")
# Create a centered container for the buttons
col1, col2, col3 = st.columns([1, 2, 1])  # Unequal columns to center the buttons
with col2:  # Middle column
    if st.button("🔧 Single Feature Transformation", key="transform_btn", use_container_width=True):
        model_state["show_popup1"] = True
        st.rerun()

# --- Popup 1: Single Feature Transformations ---
if model_state["show_popup1"]:
    st.markdown("### 🔧 Single Feature Transformation")

    # --- INPUT CHANGE: Explicitly mention the input dataset ---
    st.info("Applying single feature transformations to the **Recommended Features** (from recommendation section).")

    # Get the dataset for single feature transformations
    # This now comes from features_for_single_transform
    input_features_df = model_state.get("merged dataset", pd.DataFrame())

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
st.markdown("### 🔧 Multiple Features Transformation")

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
             "Describe the transformation",
             value=block.get("operation", ""),
             placeholder="e.g.,'ratio of col_A to col_B'",
             key=f"multi_operation_text_{active_model}_{i}" # Explicit key
        )
        model_state["multi_transform_blocks"][i]["operation"] = operation_text

    with col4:
        output_name = st.text_input(
            "Name for New Feature",
            value=block.get("output_name", ""), # Always use the stored user-provided name
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

            st.session_state[f"{active_model}_state"] = model_state # Update session state with the new model state

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






###################################################################################################################################################

# --- Data Selection Section (Start of the requested section) ---
st.markdown("### 🔎 Feature Selection")

# --- Target Variable Selection ---
st.subheader("🎯 Target Variable Selection")

# Get the dataset after multi-feature transformation
input_df_for_target = st.session_state[f"{active_model}_state"].get("final_transformed_features", pd.DataFrame())

# --- DEBUGGING STEP 3 ---
st.markdown("---")
st.subheader("DEBUG: Feature Selection Section Data")
st.write("Columns in input_df_for_target:")
st.write(input_df_for_target.columns.tolist())
# --- END DEBUGGING STEP 3 ---

if not input_df_for_target.empty:
    # Get all column names from the transformed dataset
    available_target_columns = input_df_for_target.columns.tolist()

    # Allow the user to select a target variable
    target_column_selected = st.selectbox(
        "Select Target Variable",
        available_target_columns,
        # Set default value to the previously selected target_column if it exists and is in available columns
        index=available_target_columns.index(model_state.get("target_column")) if model_state.get("target_column") in available_target_columns else 0,
        key=f"target_column_select_{active_model}"
    )

    if st.button("Select Target Variable", key=f"add_target_btn_{active_model}"):
        try:
            target_feature_name = target_column_selected
            model_state[f"target_column"] = target_column_selected
            model_state[f"target_feature"] = target_feature_name

            # Create a new DataFrame with all features EXCEPT the target column
            # This will be the input for mandatory and good-to-have features
            features_for_mandatory_df = input_df_for_target.drop(columns=[target_feature_name], errors='ignore').copy()
            model_state["features_for_mandatory"] = features_for_mandatory_df

            # The final_dataset will be constructed later after good-to-have features
            # For now, it holds the full dataset from which features are being selected
            model_state["final_dataset"] = input_df_for_target.copy()

            st.success(f"Target variable '{target_column_selected}' selected successfully! Remaining features are ready for selection.")
            st.rerun()
        except Exception as e:
            st.error(f"Error selecting target variable: {str(e)}")
else:
    st.info("Please complete multi-feature transformations first to select a target variable.")

st.markdown("---") # Separator after target selection


# --- Mandatory Features Section ---
st.subheader("📌 Mandatory Features")

# Load the dataset for mandatory features (this DataFrame will NOT contain the target column)
combined_data_for_mandatory = model_state.get("features_for_mandatory", pd.DataFrame())

if combined_data_for_mandatory.empty:
     st.warning(f"No features available after target variable selection. Please ensure a target variable is selected.")
else:
    # Call backend to get mandatory features
    mandatory_features = feat_engg_backend.select_mandatory_features(combined_data_for_mandatory)

    # Store the list of mandatory features in model_state for persistence
    model_state["selected_mandatory_features"] = mandatory_features

    # Create the dataset *after* mandatory features have been conceptually "removed"
    # This will be the input for the "Good-to-Have Features" section
    features_after_mandatory_df = combined_data_for_mandatory.drop(
        columns=[f for f in mandatory_features if f in combined_data_for_mandatory.columns],
        errors='ignore'
    ).copy()
    model_state["features_after_mandatory"] = features_after_mandatory_df


    # Define feature descriptions (can be expanded)
    feature_descriptions = {
        "OPB": "Outstanding Principal Balance of the customer's loan",
        "interest_rate": "Current interest rate applicable to the customer's loan",
        "tenure": "Duration of the loan in months",
        "credit_score_band": "Customer's credit score category (Excellent, Good, Fair, Poor)",
        "LTV": "Loan-to-Value ratio indicating the risk level of the loan",
        # Add descriptions for other potential mandatory features
        "loan_amount": "The total amount of the loan",
        "age": "Age of the customer",
        "employment_status": "Current employment status of the customer"
    }

    # Add descriptions for combined features from the loaded data
    for feature in combined_data_for_mandatory.columns:
        if feature not in feature_descriptions:
            feature_descriptions[feature] = f"Transformed or engineered feature based on original data."


    # Filter mandatory features to only show those present in the combined data
    present_mandatory_features = [feat for feat in mandatory_features if feat in combined_data_for_mandatory.columns]
    if present_mandatory_features:
        st.dataframe(pd.DataFrame({"Mandatory Features": present_mandatory_features}), hide_index=True)
        # Check if all defined mandatory features are present (from the backend's selection)
        if len(present_mandatory_features) == len(mandatory_features):
             st.success("All mandatory attributes are available and selected by the model.")
        else:
             missing_mandatory = [feat for feat in mandatory_features if feat not in combined_data_for_mandatory.columns]
             st.warning(f"Some mandatory features selected by the model are missing in the current dataset: {', '.join(missing_mandatory)}")
    else:
        st.info("No mandatory features identified by the model in the current dataset.")


st.markdown("---")

# --- Good-to-Have Features Section ---
# Get all available features from the dataset after target selection (excluding mandatory ones already displayed)
all_features_after_mandatory = combined_data_for_mandatory.columns.tolist()
available_optional_features = [feat for feat in all_features_after_mandatory if feat not in present_mandatory_features]


# Initialize feature checkboxes in session state for the active model if not exists
if f"{active_model}_feature_checkboxes" not in st.session_state:
    st.session_state[f"{active_model}_feature_checkboxes"] = {feat: False for feat in available_optional_features}

# Display good-to-have feature selection
st.subheader("✨ Good-to-Have Features")

if available_optional_features:
    features_df = pd.DataFrame({
        "Feature": available_optional_features,
        "Description": [feature_descriptions.get(feat, "No description available") for feat in available_optional_features],
        "Select": [bool(st.session_state[f"{active_model}_feature_checkboxes"].get(feat, True)) for feat in available_optional_features]
    })

    edited_df = st.data_editor(
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
            "Select": st.column_config.CheckboxColumn(
                "Select",
                width="small",
                help="Select this feature",
                default=False,
            ),
        },
        hide_index=True,
        use_container_width=True,
        key=f"{active_model}_feature_editor"
    )

    st.session_state[f"{active_model}_selected_features"] = [
        feature for feature, is_selected in zip(available_optional_features, edited_df["Select"])
        if is_selected
    ]
else:
    st.info("No optional features available in the current dataset (after target selection).")

st.markdown("---")

if st.button("📊 Show Selected Attributes"):
    all_features_summary = []
    feature_types = []

    # Add mandatory features (from the backend's selection)
    selected_mandatory_features = model_state.get("selected_mandatory_features", [])
    for feature in selected_mandatory_features:
        all_features_summary.append(feature)
        feature_types.append("Mandatory")

    # Add good-to-have features selected by the user
    for feature in st.session_state.get(f"{active_model}_selected_features", []):
        all_features_summary.append(feature)
        feature_types.append("Selected")

    # --- IMPORTANT: Recommended features are NOT added to all_features_summary here ---
    # They are independent as per your request.

    original_df_for_final_dataset = model_state.get("final_transformed_features", pd.DataFrame()).copy()
    target_col_name = model_state.get("target_feature")

    if target_col_name and target_col_name in original_df_for_final_dataset.columns:
        all_features_summary.append(target_col_name)
        feature_types.append("Target")
        features_to_include_in_final = [f for f in all_features_summary if f in original_df_for_final_dataset.columns]
        features_to_include_in_final = list(dict.fromkeys(features_to_include_in_final)) # Remove duplicates
    else:
        features_to_include_in_final = [f for f in all_features_summary if f in original_df_for_final_dataset.columns]
        features_to_include_in_final = list(dict.fromkeys(features_to_include_in_final)) # Remove duplicates
        if not target_col_name:
            st.warning("No target variable selected. The final dataset will not include a target column.")
        else:
            st.warning(f"Selected target variable '{target_col_name}' not found in the transformed dataset. The final dataset will not include it.")


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
                target_feature = model_state["target_feature"]
                final_dataset_path = os.path.join(data_registry_subfolder, f"{target_feature}_final_dataset.parquet")
                final_dataset_with_target.to_parquet(final_dataset_path, index=False)

                # Save the file path in session state for use in the next page
                st.session_state[f"{active_model}_final_dataset_path"] = final_dataset_path

                st.success(f"Target variable '{target_column}' added to the final dataset successfully! Dataset saved as '{final_dataset_path}'.")
        else:
            st.warning("Final dataset is empty, cannot save Parquet file.")

        
else:
    st.info("Please select and show your features first to enable target variable selection.")

# # Collapsible section to display the merged dataset
# st.subheader("Merged dataset preview")
# # Collapsible section to display the merged dataset


# # Retrieve the merged dataset from model_state
# merged_dataset = model_state.get("merged dataset", pd.DataFrame())

# if merged_dataset is not None and not merged_dataset.empty:
#     with st.expander("📂 View Merged Dataset", expanded=False):
#         st.markdown("### Merged Dataset")
#         st.write(f"Shape: {merged_dataset.shape}")
#         st.dataframe(merged_dataset)
# else:
#     st.warning("Merged dataset is empty or not available. Please complete the merge operations first.")
# #Target Variable Selection
# st.subheader("🎯 Target Variable Selection")

# # Define the target variable options and their corresponding feature names
# target_variable_mapping = {
#     "Profitability": "Profitability_GBP",
#     "Charge-Off": "COF_EVENT_LABEL",
#     "Prepayment": "PREPAYMENT_EVENT_LABEL"
# }

# # Get the final dataset from session state for the active model
# # Retrieve the merged dataset from model_state
# # Retrieve the merged dataset from model_state
# final_dataset = model_state.get("merged dataset", pd.DataFrame())
# final_dataset = final_dataset.dropna()
# if not final_dataset.empty:
#     # Allow the user to select a target variable
#     target_column = st.selectbox("Select Target Variable", list(target_variable_mapping.keys()), key=f"target_column_select_{active_model}")

#     if st.button("Add Target Variable to Dataset", key=f"add_target_btn_{active_model}"):
#         try:
#             # Get the target feature name from the mapping
#             target_feature = target_variable_mapping[target_column]

#             # Ensure the target column exists in the merged dataset
#             if target_feature not in final_dataset.columns:
#                 st.error(f"Target variable '{target_feature}' is not present in the merged dataset. Please ensure it is included in the data.")
#                 st.stop()

#             # Add the target column to the dataset if not already present
#             if target_feature not in final_dataset.columns:
#                 st.error(f"Target variable '{target_feature}' is missing from the dataset.")
#                 st.stop()

#             # Update the final dataset in session state to include the entire dataset along with the target column
#             st.session_state[f"{active_model}_final_dataset"] = final_dataset.copy()

#             # Store target variable in model-specific session state
#             st.session_state[f"{active_model}_target_column"] = target_column
#             st.session_state[f"{active_model}_target_feature"] = target_feature

#             # Save the final dataset (with target) as a Parquet file in a subfolder within `data_registry`
#             final_dataset_with_target = st.session_state.get(f"{active_model}_final_dataset", pd.DataFrame())
#             if not final_dataset_with_target.empty:
#                 # Create a subfolder in `data_registry` for the active model
#                 data_registry_subfolder = os.path.join("data_registry", active_model)
#                 os.makedirs(data_registry_subfolder, exist_ok=True)

#                 # Save the final dataset as a Parquet file
#                 final_dataset_path = os.path.join(data_registry_subfolder, f"{target_column}_final_dataset.parquet")
#                 final_dataset_with_target.to_parquet(final_dataset_path, index=False)

#                 # Save the file path in session state for use in the next page
#                 st.session_state[f"{active_model}_final_dataset_path"] = final_dataset_path

#                 st.success(f"Target variable '{target_column}' added to the final dataset successfully! Dataset saved as '{final_dataset_path}'.")
#             else:
#                 st.warning("Final dataset is empty, cannot save Parquet file.")

#         except Exception as e:
#             st.error(f"Error adding target variable: {str(e)}")
# else:
#     st.info("Please select and show your features first to enable target variable selection.")










# ####################################################################################################
