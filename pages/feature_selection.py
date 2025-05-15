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
import feat_engg_backend

def save_model_state(model_name):
    """Save the current model's state to the backend."""
    try:
        # Create a directory for model states if it doesn't exist
        if not os.path.exists("model_states"):
            os.makedirs("model_states")

        # Save all model-specific state variables
        model_state = {
            # Main state
            "loan_data": st.session_state[f"{model_name}_state"]["loan_data"].to_json(orient="records"),
            "bureau_data": st.session_state[f"{model_name}_state"]["bureau_data"].to_json(orient="records"),
            "onus_data": st.session_state[f"{model_name}_state"]["onus_data"].to_json(orient="records"),
            "installments_data": st.session_state[f"{model_name}_state"]["installments_data"].to_json(orient="records"),
            "show_popup1": st.session_state[f"{model_name}_state"]["show_popup1"],
            "transform_blocks": st.session_state[f"{model_name}_state"]["transform_blocks"],
            "multi_transform_blocks": st.session_state[f"{model_name}_state"]["multi_transform_blocks"],
            "final_transformed_features": st.session_state[f"{model_name}_state"]["final_transformed_features"].to_json(orient="records"),
            "recommended_features": st.session_state[f"{model_name}_state"]["recommended_features"].to_json(orient="records"),
            "final_dataset": st.session_state[f"{model_name}_state"]["final_dataset"].to_json(orient="records"),
            "selected_features": st.session_state[f"{model_name}_state"]["selected_features"],
            "feature_checkboxes": st.session_state[f"{model_name}_state"]["feature_checkboxes"],
            "show_filter": st.session_state[f"{model_name}_state"]["show_filter"],
            "filtered_features": st.session_state[f"{model_name}_state"]["filtered_features"],
            "filter_text": st.session_state[f"{model_name}_state"]["filter_text"],
            "merge_blocks": st.session_state[f"{model_name}_state"]["merge_blocks"],
            "merged_tables": st.session_state[f"{model_name}_state"]["merged_tables"],
            "combined_dataset": st.session_state[f"{model_name}_state"]["combined_dataset"].to_json(orient="records") if st.session_state[f"{model_name}_state"]["combined_dataset"] is not None else None,
            # Other state variables
            "operations_complete": st.session_state.get(f"{model_name}_operations_complete", {}),
            "show_filter_data": st.session_state.get(f"{model_name}_show_filter_data", False),
            "show_merge": st.session_state.get(f"{model_name}_show_merge", False),
            "single_transform_success": st.session_state.get(f"{model_name}_single_transform_success", None),
            "multi_transform_success": st.session_state.get(f"{model_name}_multi_transform_success", None),
            "recommended_features_state": st.session_state.get(f"{model_name}_recommended_features", pd.DataFrame()).to_json(orient="records"),
            "final_dataset_state": st.session_state.get(f"{model_name}_final_dataset", pd.DataFrame()).to_json(orient="records"),
            "selected_features_state": st.session_state.get(f"{model_name}_selected_features", []),
            "feature_checkboxes_state": st.session_state.get(f"{model_name}_feature_checkboxes", {}),
            "filter_blocks_state": st.session_state.get(f"{model_name}_filter_blocks", []),
            "target_column": st.session_state.get(f"{model_name}_target_column", None),
            "target_feature": st.session_state.get(f"{model_name}_target_feature", None),
            "final_dataset_json": st.session_state.get(f"{model_name}_final_dataset_json", None)
        }

        # Save to a single file
        state_file = f"model_states/{model_name}_state.json"
        with open(state_file, 'w') as f:
            json.dump(model_state, f)

    except Exception as e:
        st.error(f"Error saving model state: {str(e)}")

def load_model_state(model_name):
    """Load a model's state from the backend."""
    try:
        state_file = f"model_states/{model_name}_state.json"
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                model_state = json.load(f)

            # Initialize main state
            st.session_state[f"{model_name}_state"] = {
                "loan_data": pd.read_json(model_state["loan_data"], orient="records"),
                "bureau_data": pd.read_json(model_state["bureau_data"], orient="records"),
                "onus_data": pd.read_json(model_state["onus_data"], orient="records"),
                "installments_data": pd.read_json(model_state["installments_data"], orient="records"),
                "show_popup1": model_state["show_popup1"],
                "transform_blocks": model_state["transform_blocks"],
                "multi_transform_blocks": model_state["multi_transform_blocks"],
                "final_transformed_features": pd.read_json(model_state["final_transformed_features"], orient="records"),
                "recommended_features": pd.read_json(model_state["recommended_features"], orient="records"),
                "final_dataset": pd.read_json(model_state["final_dataset"], orient="records"),
                "selected_features": model_state["selected_features"],
                "feature_checkboxes": model_state["feature_checkboxes"],
                "show_filter": model_state["show_filter"],
                "filtered_features": model_state["filtered_features"],
                "filter_text": model_state["filter_text"],
                "merge_blocks": model_state["merge_blocks"],
                "merged_tables": model_state["merged_tables"],
                "combined_dataset": pd.read_json(model_state["combined_dataset"], orient="records") if model_state["combined_dataset"] is not None else None
            }

            # Load other state variables with default values if not present
            st.session_state[f"{model_name}_operations_complete"] = model_state.get("operations_complete", {
                "merge": False,
                "recommend": False,
                "accept": False
            })
            st.session_state[f"{model_name}_show_filter_data"] = model_state.get("show_filter_data", False)
            st.session_state[f"{model_name}_show_merge"] = model_state.get("show_merge", False)
            st.session_state[f"{model_name}_single_transform_success"] = model_state.get("single_transform_success", None)
            st.session_state[f"{model_name}_multi_transform_success"] = model_state.get("multi_transform_success", None)
            st.session_state[f"{model_name}_recommended_features"] = pd.read_json(model_state.get("recommended_features_state", pd.DataFrame().to_json(orient="records")), orient="records")
            st.session_state[f"{model_name}_final_dataset"] = pd.read_json(model_state.get("final_dataset_state", pd.DataFrame().to_json(orient="records")), orient="records")
            st.session_state[f"{model_name}_selected_features"] = model_state.get("selected_features_state", [])
            st.session_state[f"{model_name}_feature_checkboxes"] = model_state.get("feature_checkboxes_state", {})
            st.session_state[f"{model_name}_filter_blocks"] = model_state.get("filter_blocks_state", [{
                "dataset": "Bureau Data",
                "feature": "",
                "operation": "Greater Than",
                "value": 0,
                "output_name": ""
            }])
            st.session_state[f"{model_name}_target_column"] = model_state.get("target_column", None)
            st.session_state[f"{model_name}_target_feature"] = model_state.get("target_feature", None)
            st.session_state[f"{model_name}_final_dataset_json"] = model_state.get("final_dataset_json", None)

    except Exception as e:
        st.error(f"Error loading model state: {str(e)}")
        # Initialize with default values if loading fails
        initialize_new_model_state(model_name)

def initialize_new_model_state(model_name):
    """Initialize a fresh state for a new model."""
    # Initialize main state
    st.session_state[f"{model_name}_state"] = {
        # --- Add raw_datasets and filtered_datasets dictionaries here ---
        "raw_datasets": {
            "Loan Data": pd.read_csv("loan_data.csv"),
            "Bureau Data": pd.read_csv("loan_data.csv").copy(), # Assuming loan_data.csv contains all initial data
            "On-Us Data": pd.read_csv("loan_data.csv").copy(),
            "Installments Data": pd.read_csv("loan_data.csv").copy(),
        },
        "filtered_datasets": {}, # Initialize as an empty dictionary to store filtered results

        # Your existing state variables follow (consider removing individual dataframes later for cleaner state):
        "loan_data": pd.read_csv("loan_data.csv"),
        "bureau_data": pd.read_csv("loan_data.csv").copy(),
        "onus_data": pd.read_csv("loan_data.csv").copy(),
        "installments_data": pd.read_csv("loan_data.csv").copy(),

        "show_popup1": False,
        "transform_blocks": [{
            "feature": "",
            "operation": "Addition",
            "value": 1.0,
            "output_name": ""
        }],
        "multi_transform_blocks": [{
            "features": [],
            "operation": "",
            "output_name": ""
        }],
        "final_transformed_features": pd.DataFrame(),
        "recommended_features": pd.DataFrame(),
        "final_dataset": pd.DataFrame(),
        "selected_features": [],
        "feature_checkboxes": {},
        "show_filter": False,
        "filtered_features": [],
        "filter_text": "",
        "merge_blocks": [{
            "left_table": "Bureau Data", # You might want to update these default names
            "right_table": "On-Us Data",  # to refer to the keys in raw_datasets
            "how": "inner",
            "on": [],
            "left_on": [],
            "right_on": [],
            "merged_name": "Merged_1",
        }],
        "merged_tables": {},
        "combined_dataset": None,
    }

    # Initialize other state variables (these seem to be outside the main model_state dict)
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
        "dataset": "Bureau Data", # You might want to update this default name
        "feature": "",
        "operation": "Greater Than",
        "value": 0,
        "output_name": ""
    }]
    st.session_state[f"{model_name}_target_column"] = None
    st.session_state[f"{model_name}_target_feature"] = None
    st.session_state[f"{model_name}_final_dataset_json"] = None

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

        # If no saved state exists, initialize with default values
        if f"{model_name}_state" not in st.session_state:
            initialize_new_model_state(model_name)
            save_model_state(model_name)

        st.rerun()
    except Exception as e:
        st.error(f"Error switching models: {str(e)}")

# Add caching decorators for expensive operations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(file_path: str) -> pd.DataFrame:
    """Load and optimize a DataFrame from CSV."""
    df = pd.read_csv(file_path)
    return optimize_dataframe(df)

@st.cache_data(ttl=3600)
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage and performance."""
    # Create a copy to avoid modifying the original
    df = df.copy()

    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if df[col].dtype == 'int64':
            # Downcast integers
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            # Downcast floats
            df[col] = pd.to_numeric(df[col], downcast='float')

    # Optimize categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')

    return df

@st.cache_data(ttl=3600)
def perform_merge(left_df: pd.DataFrame, right_df: pd.DataFrame, merge_kwargs: dict) -> pd.DataFrame:
    """Perform optimized merge operation."""
    return pd.merge(left_df, right_df, **merge_kwargs)


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
if "models" not in st.session_state:
    # Change initial model name
    st.session_state.models = ["Feature Transformation Page 1"]  # Start with Feature Transformation Page 1
if "active_model" not in st.session_state:
    # Default active model
    st.session_state.active_model = "Feature Transformation Page 1"
if "single_transform_success" not in st.session_state:
    st.session_state.single_transform_success = None  # Initialize single_transform_success
if "multi_transform_success" not in st.session_state:
    st.session_state.multi_transform_success = None  # Initialize multi_transform_success

# Model Selection Section
col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    # Create a row for dropdown and button
    dropdown_col, button_col = st.columns([2, 1.2])
    with dropdown_col:
        # Model selection dropdown
        selected_model = st.selectbox(
            "Select Page", # Change label
            st.session_state.models,
            index=st.session_state.models.index(st.session_state.active_model),
            key="model_selector"
        )
    with button_col:
        st.text("") # Add this line for vertical alignment
        # Create new model button - Change button text
        if st.button("➕ Create New Page", key="add_model_btn", use_container_width=True):
            add_new_model()

# Switch model if selection changed
if selected_model != st.session_state.active_model:
    switch_model(selected_model)

# Initialize state for the first page if it doesn't exist
first_model_name = "Feature Transformation Page 1"
if f"{first_model_name}_state" not in st.session_state:
    # Try to load existing state from backend
    load_model_state(first_model_name)

    # If no saved state exists, initialize with default values
    if f"{first_model_name}_state" not in st.session_state:
        st.session_state[f"{first_model_name}_state"] = {
            # --- Add raw_datasets and filtered_datasets dictionaries here ---
            "raw_datasets": {
                "Loan Data": pd.read_csv("loan_data.csv"),
                "Bureau Data": pd.read_csv("loan_data.csv").copy(), # Assuming loan_data.csv contains all initial data
                "On-Us Data": pd.read_csv("loan_data.csv").copy(),
                "Installments Data": pd.read_csv("loan_data.csv").copy(),
            },
            "filtered_datasets": {}, # Initialize as an empty dictionary to store filtered results

            # Your existing state variables follow (consider removing individual dataframes later for cleaner state):
            "loan_data": pd.read_csv("loan_data.csv"),
            "bureau_data": pd.read_csv("loan_data.csv").copy(),
            "onus_data": pd.read_csv("loan_data.csv").copy(),
            "installments_data": pd.read_csv("loan_data.csv").copy(),

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
                "left_table": "Bureau Data", # You might want to update these default names
                "right_table": "On-Us Data",  # to refer to the keys in raw_datasets
                "how": "inner",
                "on": [],
                "left_on": [],
                "right_on": [],
                "merged_name": "Merged_1",
            }],
            "merged_tables": {},
            "combined_dataset": None,

            # --- Update filter_blocks initialization within the main state ---
            "filter_blocks": [{
                "dataset": "Bureau Data",  # You might want to update this default name
                "feature": "",
                "operation": feat_engg_backend.get_filter_operations()[0], # Default operation from feat_engg_backend
                "value": None, # Initialize value as None
                "output_name": ""
            }],
            # --- Add target column/feature initialization ---
            "target_column": None,
            "target_feature": None,
            "final_dataset_json": None,
        }
        # --- Initialize other session state variables for the first page (outside the main state dict) ---
        st.session_state[f"{first_model_name}_operations_complete"] = {
            "merge": False,
            "recommend": False,
            "accept": False
        }
        st.session_state[f"{first_model_name}_show_filter_data"] = False
        st.session_state[f"{first_model_name}_show_merge"] = False
        st.session_state[f"{first_model_name}_single_transform_success"] = None
        st.session_state[f"{first_model_name}_multi_transform_success"] = None
        st.session_state[f"{first_model_name}_recommended_features"] = pd.DataFrame()
        st.session_state[f"{first_model_name}_final_dataset"] = pd.DataFrame()
        st.session_state[f"{first_model_name}_selected_features"] = []
        st.session_state[f"{first_model_name}_feature_checkboxes"] = {}
        # --- Update the separate Model 1_filter_blocks initialization ---
        st.session_state[f"{first_model_name}_filter_blocks"] = [{
            "dataset": "Bureau Data", # You might want to update this default name
            "feature": "",
            "operation": feat_engg_backend.get_filter_operations()[0], # Default operation from feat_engg_backend
            "value": None, # Default value
            "output_name": ""
        }]
        # --- Initialize other Model 1 specific state variables ---
        st.session_state[f"{first_model_name}_target_column"] = None
        st.session_state[f"{first_model_name}_target_feature"] = None
        st.session_state[f"{first_model_name}_final_dataset_json"] = None


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
installments_name = "Installments Data"

# Store the names and dataframes in a dictionary for easy access
dataset_mapping = {
    bureau_name: model_state["bureau_data"],
    onus_name: model_state["onus_data"],
    installments_name: model_state["installments_data"],
}

with col1:
    st.markdown(
        f"<div style='border: 1px solid #e0e0e0; border-radius: 4px; padding: 10px; text-align: center;'>"
        f"<p style='margin: 0;'>{bureau_name}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"<div style='border: 1px solid #e0e0e0; border-radius: 4px; padding: 10px; text-align: center;'>"
        f"<p style='margin: 0;'>{onus_name}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"<div style='border: 1px solid #e0e0e0; border-radius: 4px; padding: 10px; text-align: center;'>"
        f"<p style='margin: 0;'>{installments_name}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


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
    raw_datasets = model_state.get("raw_datasets", {}) # Get the raw datasets dictionary

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
                    model_state["combined_dataset"] = all_merged_results[final_merged_name].copy() # Store the final result
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
            model_state["operations_complete"]["merge"] = True
            st.session_state[f"{active_model}_state"] = model_state # Update session state

            # Clear the merge blocks after successful application (optional, depends on desired workflow)
            # model_state["merge_blocks"] = []
            # st.session_state[f"{active_model}_state"] = model_state # Update session state

            st.rerun() # Rerun to update the UI with the new state

        except ValueError as ve:
            st.error(f"Merge configuration error: {ve}")
            # Reset merge complete state on error in model-specific state
            model_state["operations_complete"]["merge"] = False
            st.session_state[f"{active_model}_state"] = model_state # Update session state

        except Exception as e:
            st.error(f"Error during merge operations: {str(e)}")
            # Reset merge complete state on error in model-specific state
            model_state["operations_complete"]["merge"] = False
            st.session_state[f"{active_model}_state"] = model_state # Update session state


st.markdown("---")

############################################################################################################################################3

# --- Recommend Features Button ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("✨ Recommend Features", key="recommend_features", use_container_width=True):
        try:
            # Create sample features
            sample_features = [
                "OPB", "interest_rate", "tenure", "credit_score_band", "LTV",
                "age", "income", "employment_length", "debt_to_income", "payment_history",
                "loan_amount", "loan_type", "property_value", "down_payment", "loan_purpose",
                "marital_status", "education", "residence_type", "number_of_dependents", "previous_loans"
            ]

            # Create feature descriptions
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

            # Create sample data with realistic values
            sample_data = pd.DataFrame({
                "OPB": np.random.uniform(10000, 500000, 100),
                "interest_rate": np.random.uniform(3.5, 8.5, 100),
                "tenure": np.random.randint(12, 360, 100),
                "credit_score_band": np.random.choice(["Excellent", "Good", "Fair", "Poor"], 100),
                "LTV": np.random.uniform(0.3, 0.95, 100),
                "age": np.random.randint(18, 75, 100),
                "income": np.random.uniform(30000, 200000, 100),
                "employment_length": np.random.randint(1, 40, 100),
                "debt_to_income": np.random.uniform(0.1, 0.5, 100),
                "payment_history": np.random.uniform(0, 100, 100),
                "loan_amount": np.random.uniform(10000, 500000, 100),
                "loan_type": np.random.choice(["Personal", "Mortgage", "Auto", "Business"], 100),
                "property_value": np.random.uniform(100000, 1000000, 100),
                "down_payment": np.random.uniform(5000, 100000, 100),
                "loan_purpose": np.random.choice(["Home Purchase", "Refinance", "Debt Consolidation", "Business"], 100),
                "marital_status": np.random.choice(["Single", "Married", "Divorced", "Widowed"], 100),
                "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], 100),
                "residence_type": np.random.choice(["Own", "Rent", "Other"], 100),
                "number_of_dependents": np.random.randint(0, 5, 100),
                "previous_loans": np.random.randint(0, 10, 100)
            })

            # Calculate statistics for each feature
            stats = []
            for feature in sample_features:
                data = sample_data[feature]
                if pd.api.types.is_numeric_dtype(data):
                    stats.append({
                        'Feature': feature,
                        'Description': feature_descriptions.get(feature, f"Description for {feature}"),
                        'Min': f"{data.min():.2f}",
                        'Max': f"{data.max():.2f}",
                        'Mean': f"{data.mean():.2f}",
                        'Data Type': 'Numeric'
                    })
                else:
                    stats.append({
                        'Feature': feature,
                        'Description': feature_descriptions.get(feature, f"Description for {feature}"),
                        'Min': 'N/A',
                        'Max': 'N/A',
                        'Mean': 'N/A',
                        'Data Type': 'Categorical'
                    })

            # Create feature info DataFrame with statistics
            feature_info = pd.DataFrame(stats)

            # Store in session state
            st.session_state.recommended_features = sample_data
            st.session_state.feature_info = feature_info
            if f"{active_model}_operations_complete" not in st.session_state:
                st.session_state[f"{active_model}_operations_complete"] = {}
            st.session_state[f"{active_model}_operations_complete"]["recommend"] = True
            st.rerun()

        except Exception as e:
            st.error(f"Error recommending features: {str(e)}")

# Display recommended features if they exist
if st.session_state.get(f"{active_model}_operations_complete", {}).get("recommend", False) and hasattr(st.session_state, 'feature_info'):
    st.markdown("### Recommended Features")
    # Create a dataframe for the features with the same styling as good-to-have section
    features_df = pd.DataFrame({
        "Feature": st.session_state.feature_info["Feature"],
        "Description": st.session_state.feature_info["Description"],
        "Min": st.session_state.feature_info["Min"],
        "Max": st.session_state.feature_info["Max"],
        "Mean": st.session_state.feature_info["Mean"],
        "Data Type": st.session_state.feature_info["Data Type"]
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

# --- Accept Recommended Features Button ---
if st.session_state.get(f"{active_model}_operations_complete", {}).get("recommend", False):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✅ Accept Recommended Features", key="accept_recommended_features", use_container_width=True):
            try:
                if hasattr(st.session_state, 'recommended_features'):
                    # Get the recommended features
                    recommended_features = st.session_state.recommended_features

                    # Store in final dataset
                    st.session_state.final_dataset = recommended_features.copy()

                    # Save to CSV
                    recommended_features.to_csv("recommended_features.csv", index=False)

                    # Store in model state for transformations
                    model_state = st.session_state[f"{active_model}_state"]
                    model_state["recommended_features"] = recommended_features.copy()

                    # Update state
                    if f"{active_model}_operations_complete" not in st.session_state:
                        st.session_state[f"{active_model}_operations_complete"] = {}
                    st.session_state[f"{active_model}_operations_complete"]["accept"] = True

                    # Store success message in session state
                    st.session_state.accept_success = True

                    # Rerun to update the UI
                    st.rerun()
                else:
                    st.warning("No recommended features found. Please click 'Recommend Features' first.")
            except Exception as e:
                st.error(f"Error accepting recommended features: {str(e)}")

# Display success message if it exists in session state
if st.session_state.get("accept_success", False):
    st.success("✅ All recommended features have been selected successfully!")
    # Clear the success message after displaying
    st.session_state.accept_success = False

st.markdown("---")

# --- Data Transformation Buttons ---
st.subheader("Data Actions")
# Create a centered container for the buttons
col1, col2, col3 = st.columns([1, 2, 1])  # Unequal columns to center the buttons
with col2:  # Middle column
    if st.button("🔧 Data Transformation", key="transform_btn", use_container_width=True):
        model_state["show_popup1"] = True
        st.rerun()

# --- Popup 1: Single Feature Transformations ---
if model_state["show_popup1"]:
    st.markdown("### 🔧 Single Feature Transformation")

    # Get the recommended features as input for single feature transformations
    if "recommended_features" in st.session_state and not st.session_state.recommended_features.empty:
        input_features = st.session_state.recommended_features.columns.tolist()
    else:
        input_features = []

    # Initialize transform blocks if empty
    if not model_state["transform_blocks"]:
        model_state["transform_blocks"] = [{
            "feature": input_features[0] if input_features else "",
            "operation": "Addition",
            "value": 1.0,
            "output_name": ""
        }]

    # Show transformation blocks
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
                st.warning("No features available. Please recommend and accept features first.")
                feature = None

        with col3:
            operation = st.selectbox(
                "Operation",
                ["Addition", "Subtraction", "Multiplication", "Division", "Log", "Square Root", "Power", "Absolute Value", "Rename"],
                key=f"single_operation_{i}",
                index=["Addition", "Subtraction", "Multiplication", "Division", "Log", "Square Root", "Power", "Absolute Value", "Rename"].index(block.get("operation", "Addition"))
            )

        with col4:
            freeze_value_ops = ["Rename", "Log", "Square Root", "Absolute Value"]
            if operation in freeze_value_ops:
                default_val = 0 if operation == "Rename" else 1
                value = st.number_input(
                    "Value",
                    value=default_val,
                    key=f"single_value_{i}",
                    disabled=True
                )
            elif operation in ["Addition", "Subtraction", "Multiplication", "Division", "Power"]:
                value = st.number_input(
                    "Value",
                    value=block.get("value", 1.0),
                    key=f"single_value_{i}"
                )
            else:
                value = st.number_input(
                    "Value",
                    value=block.get("value", 1.0),
                    key=f"single_value_{i}"
                )

        with col5:
            if feature:
                if operation == "Rename":
                    suggested_output = f"{feature}_renamed"
                elif value is not None:
                    suggested_output = f"{feature}{operation.replace(' ', '')}{str(value).replace('.', '')}"
                else:
                    suggested_output = f"{feature}{operation.replace(' ', '')}"

                prev_suggestion = block.get("prev_suggestion", "")
                prev_output_name = block.get("output_name", "")

                if not prev_output_name or prev_output_name == prev_suggestion:
                    output_name = suggested_output
                else:
                    output_name = prev_output_name

                output_name = st.text_input(
                    "Output Feature",
                    value=output_name,
                    key=f"single_output_{i}"
                )

                model_state["transform_blocks"][i] = {
                    "feature": feature,
                    "operation": operation,
                    "value": value,
                    "output_name": output_name,
                    "prev_suggestion": suggested_output
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
            st.warning("Please recommend and accept features first before adding transformations.")

    # --- Apply Transformation Button ---
    if st.button("✅ Apply Transformation", key="apply_single_transform"):
        try:
            if not input_features:
                st.warning("No features available. Please recommend and accept features first.")
            else:
                transformed_features = {}
                successful_transformations = []
                # Get the current state of recommended features
                current_recommended_features = st.session_state.recommended_features.copy()

                for block in model_state["transform_blocks"]:
                    feature = block["feature"]
                    operation = block["operation"]
                    value = block["value"]
                    output_name = block["output_name"]
                    if not feature or not operation or not output_name:
                        continue

                    # Apply transformation using the backend function
                    try:
                         # Pass the current recommended features to the backend function
                         current_recommended_features = feat_engg_backend.apply_single_feature_transform(
                             current_recommended_features,
                             block
                         )
                         successful_transformations.append(f"- Applied transformation '{operation}' on '{feature}' to create '{output_name}'")
                    except (ValueError, TypeError, Exception) as e:
                         st.error(f"Error applying transformation block {block}: {e}")
                         continue # Skip to the next block on error

                # Update session state with the new DataFrame containing transformed features
                st.session_state.recommended_features = current_recommended_features

                # Display success message and list of successful transformations
                if successful_transformations:
                    st.session_state.single_transform_success = "✅ Single feature transformations applied successfully!"
                    st.info("Successful transformations:\n" + "\n".join(successful_transformations)) # Display list

                # Clear the transform blocks after successful application
                model_state["transform_blocks"] = []
                st.rerun()
        except Exception as e:
            st.error(f"Error applying transformations: {str(e)}")


    # Display success message if it exists
    if st.session_state.single_transform_success:
        st.success(st.session_state.single_transform_success)
        # The info message is displayed immediately after successful application, so no need to clear here.
        st.session_state.single_transform_success = None

    # --- Multi-Feature Transformation Section ---
    st.markdown("### 🔧 Multiple Features Transformation")

    # Initialize multi transform blocks if empty
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
            if st.button("❌", key=f"remove_multi_{i}"):
                model_state["multi_transform_blocks"].pop(i)
                st.rerun()

        with col2:
            if input_features:
                selected_features = st.multiselect(
                    "Choose Features to Combine",
                    input_features,
                    default=block.get("features", []),
                    key=f"multi_features_{i}"
                )
                model_state["multi_transform_blocks"][i]["features"] = selected_features
            else:
                st.warning("No features available. Please recommend and accept features first.")

        with col3:
            # Use selectbox for predefined multi-feature operations
            operation = st.selectbox(
                 "Select Combination Operation",
                 feat_engg_backend.get_multi_transformation_operations(), # Get operations from backend
                 index=feat_engg_backend.get_multi_transformation_operations().index(block.get("operation", feat_engg_backend.get_multi_transformation_operations()[0])) if block.get("operation") in feat_engg_backend.get_multi_transformation_operations() else 0,
                 key=f"multi_operation_{i}"
            )
            model_state["multi_transform_blocks"][i]["operation"] = operation


        with col4:
            # Suggest a default output name based on selected features and operation
            features_for_name = block.get("features", [])
            operation_for_name = block.get("operation", "").replace(' ', '').lower()

            if features_for_name and operation_for_name:
                 # Create a simple suggested name
                 suggested_output = f"{operation_for_name}_{'_'.join(features_for_name).lower()}"
                 # Limit suggested name length to avoid overly long column names
                 suggested_output = suggested_output[:50] # Arbitrary limit

                 # Check if a name is already saved for this block, if so, use it
                 current_output_name = block.get("output_name", "")

                 # If no name is saved, or the saved name is the old suggestion for this block, use the new suggestion
                 prev_suggestion_key = f"multi_prev_suggestion_{i}"
                 prev_suggestion = st.session_state.get(prev_suggestion_key, "")

                 if not current_output_name or current_output_name == prev_suggestion:
                      output_name_value = suggested_output
                 else:
                      output_name_value = current_output_name

                 output_name = st.text_input(
                     "Name for New Feature",
                     value=output_name_value,
                     key=f"multi_output_{i}"
                 )
                 model_state["multi_transform_blocks"][i]["output_name"] = output_name
                 # Store the current suggestion so we know if the user overwrites it
                 st.session_state[prev_suggestion_key] = suggested_output
            else:
                # If no features or operation selected, provide a default empty input
                output_name = st.text_input(
                    "Name for New Feature",
                    value=block.get("output_name", ""),
                    key=f"multi_output_{i}"
                )
                model_state["multi_transform_blocks"][i]["output_name"] = output_name
                # Clear previous suggestion if features/operation are no longer selected
                prev_suggestion_key = f"multi_prev_suggestion_{i}"
                if prev_suggestion_key in st.session_state:
                     del st.session_state[prev_suggestion_key]


    if st.button("➕ Add New Feature Combination", key="add_multi_transform"):
        if input_features:
            model_state["multi_transform_blocks"].append({
                "features": [],
                "operation": "",
                "output_name": ""
            })
            st.rerun()
        else:
            st.warning("Please recommend and accept features first before adding transformations.")

    if st.button("✅ Apply all transformations", key="apply_multi_transforms"):
        try:
            if not input_features:
                st.warning("No features available. Please recommend and accept features first.")
            else:
                successful_transformations = []
                # Get the current state of recommended features
                current_recommended_features = st.session_state.recommended_features.copy()

                for block in model_state["multi_transform_blocks"]:
                    features = block["features"]
                    operation = block["operation"]
                    output_name = block["output_name"]
                    if not features or not operation or not output_name:
                        continue

                    # Apply multi-feature transformation using the backend function
                    try:
                         current_recommended_features = feat_engg_backend.apply_multi_feature_transform(
                             current_recommended_features,
                             block
                         )
                         successful_transformations.append(f"- Applied multi-feature transformation '{operation}' on {', '.join(features)} to create '{output_name}'")
                    except (ValueError, TypeError, Exception) as e:
                         st.error(f"Error applying multi-feature transformation block {block}: {e}")
                         continue # Skip to the next block on error

                # Update session state with the new DataFrame containing transformed features
                st.session_state.recommended_features = current_recommended_features

                # Save the updated recommended features to a CSV file (optional but good practice)
                combined_dataset_file = f"model_{active_model}_dataset.csv"
                st.session_state.recommended_features.to_csv(combined_dataset_file, index=False)

                # Display success message and list of successful transformations
                if successful_transformations:
                     st.session_state.multi_transform_success = "✅ Multi-feature transformations applied successfully!"
                     st.info("Successful transformations:\n" + "\n".join(successful_transformations)) # Display list


                model_state["multi_transform_blocks"] = []
                st.rerun()
        except Exception as e:
            st.error(f"An unexpected error occurred during multi-feature transformations: {str(e)}")


    if st.session_state.multi_transform_success:
        st.success(st.session_state.multi_transform_success)
        # The info message is displayed immediately after successful application, so no need to clear here.
        st.session_state.multi_transform_success = None


# --- Initialize Session State ---
# Ensure these are initialized for the active model as well
active_model = st.session_state.active_model

if f"{active_model}_final_dataset" not in st.session_state:
    st.session_state[f"{active_model}_final_dataset"] = pd.DataFrame()  # Initialize as an empty DataFrame
if f"{active_model}_recommended_features" not in st.session_state:
    st.session_state[f"{active_model}_recommended_features"] = pd.DataFrame()  # Initialize as an empty DataFrame
# Note: transformed_features and final_transformed_features are now handled within recommended_features
# if f"{active_model}_transformed_features" not in st.session_state:
#     st.session_state[f"{active_model}_transformed_features"] = pd.DataFrame()  # Initialize as an empty DataFrame
# if f"{active_model}_final_transformed_features" not in st.session_state:
#     st.session_state[f"{active_model}_final_transformed_features"] = pd.DataFrame()  # Initialize as an empty DataFrame
if f"{active_model}_selected_features" not in st.session_state:
    st.session_state[f"{active_model}_selected_features"] = []  # Initialize as an empty list
if f"{active_model}_feature_checkboxes" not in st.session_state:
    st.session_state[f"{active_model}_feature_checkboxes"] = {}  # Initialize as an empty dictionary
if f"{active_model}_show_filter" not in st.session_state:
    st.session_state[f"{active_model}_show_filter"] = False
if f"{active_model}_filtered_features" not in st.session_state:
    st.session_state[f"{active_model}_filtered_features"] = []
if f"{active_model}_filter_text" not in st.session_state:
    st.session_state[f"{active_model}_filter_text"] = ""


# --- Data Selection Section ---
st.markdown("### 🔎 Feature Selection")

# Load the combined dataset (recommended features) from the active model's state
combined_data = st.session_state.get(f"{active_model}_recommended_features", pd.DataFrame())

if combined_data.empty:
     # Fallback to loading from file if session state is empty
     try:
         # Assuming the file is named based on the active model
         combined_dataset_file = f"model_{active_model}_dataset.csv"
         if os.path.exists(combined_dataset_file):
             combined_data = pd.read_csv(combined_dataset_file)
             # Store loaded data in session state for the active model
             st.session_state[f"{active_model}_recommended_features"] = combined_data
         else:
             st.warning(f"No combined dataset found for '{active_model}'. Please complete the merging and transformation steps.")
             combined_data = pd.DataFrame() # Ensure combined_data is an empty DataFrame if file not found
     except FileNotFoundError:
         st.warning(f"No combined dataset file found for '{active_model}'. Please complete the merging and transformation steps.")
         combined_data = pd.DataFrame() # Ensure combined_data is an empty DataFrame on error
     except Exception as e:
         st.error(f"Error loading combined dataset for '{active_model}': {str(e)}")
         combined_data = pd.DataFrame() # Ensure combined_data is an empty DataFrame on error


# Define mandatory features
mandatory_features = ["OPB", "interest_rate", "tenure", "credit_score_band", "LTV"]

# Define feature descriptions
feature_descriptions = {
    "OPB": "Outstanding Principal Balance of the customer's loan",
    "interest_rate": "Current interest rate applicable to the customer's loan",
    "tenure": "Duration of the loan in months",
    "credit_score_band": "Customer's credit score category (Excellent, Good, Fair, Poor)",
    "LTV": "Loan-to-Value ratio indicating the risk level of the loan",
}

# Add descriptions for combined features from the loaded data
if not combined_data.empty:
    for feature in combined_data.columns:
        if feature not in feature_descriptions:
            # You might generate a more specific description here if possible
            feature_descriptions[feature] = f"Transformed or engineered feature based on original data."


# Show mandatory features
st.subheader("📌 Mandatory Features")
# Filter mandatory features to only show those present in the combined data
present_mandatory_features = [feat for feat in mandatory_features if feat in combined_data.columns]
if present_mandatory_features:
    st.dataframe(pd.DataFrame({"Mandatory Features": present_mandatory_features}), hide_index=True)
    # Check if all defined mandatory features are present
    if len(present_mandatory_features) == len(mandatory_features):
         st.success("All mandatory attributes are available")
    else:
         missing_mandatory = [feat for feat in mandatory_features if feat not in combined_data.columns]
         st.warning(f"Missing mandatory features: {', '.join(missing_mandatory)}")
else:
    st.warning("No mandatory features found in the combined dataset.")


st.markdown("---")

# Get all available features from the combined dataset
all_features = combined_data.columns.tolist()
# Filter out mandatory features that are present in the combined data from optional features
available_optional_features = [feat for feat in all_features if feat not in present_mandatory_features]


# Initialize feature checkboxes in session state for the active model if not exists
if f"{active_model}_feature_checkboxes" not in st.session_state:
    st.session_state[f"{active_model}_feature_checkboxes"] = {feat: False for feat in available_optional_features}

# Display good-to-have feature selection
st.subheader("✨ Good-to-Have Features")

if available_optional_features:
    # Create a dataframe for the features
    features_df = pd.DataFrame({
        "Feature": available_optional_features,
        "Description": [feature_descriptions.get(feat, "No description available") for feat in available_optional_features],
        # Use the model-specific checkbox state
        "Select": [bool(st.session_state[f"{active_model}_feature_checkboxes"].get(feat, False)) for feat in available_optional_features]
    })

    # Display the features in a dataframe with custom styling
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
        # Use a model-specific key for the data editor
        key=f"{active_model}_feature_editor"
    )

    # Update selected features based on checkboxes for the active model
    st.session_state[f"{active_model}_selected_features"] = [
        feature for feature, is_selected in zip(available_optional_features, edited_df["Select"])
        if is_selected
    ]
else:
    st.info("No optional features available in the combined dataset.")

st.markdown("---")

# Combine and preview
if st.button("📊 Show Selected Attributes"):
    # Create a summary of all selected features
    all_features_summary = []
    feature_types = []

    # Add mandatory features that are present in the data
    for feature in present_mandatory_features:
        all_features_summary.append(feature)
        feature_types.append("Mandatory")

    # Add selected optional features
    # Use the model-specific selected features
    for feature in st.session_state.get(f"{active_model}_selected_features", []):
        all_features_summary.append(feature)
        feature_types.append("Selected")

    if all_features_summary:
        # Create and display the summary dataframe
        summary_df = pd.DataFrame({
            "Feature": all_features_summary,
            "Type": feature_types
        })

        st.subheader("Selected Features Summary")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Store the final dataset in session state for the active model
        working_df = combined_data.copy()
        # Filter working_df to include only the features in all_features_summary
        features_to_include = [f for f in all_features_summary if f in working_df.columns]
        if features_to_include:
            st.session_state[f"{active_model}_final_dataset"] = working_df[features_to_include]
        else:
            st.warning("None of the selected features are present in the combined data.")
            st.session_state[f"{active_model}_final_dataset"] = pd.DataFrame() # Set to empty if no features

    else:
        st.info("No features selected (mandatory or optional).")
        st.session_state[f"{active_model}_final_dataset"] = pd.DataFrame() # Set to empty if no features


# Target Variable Selection
st.subheader("🎯 Target Variable Selection")

# Define the target variable options and their corresponding feature names
target_variable_mapping = {
    "Profitability": "Profitability_GBP",
    "Charge-Off": "COF_EVENT_LABEL",
    "Prepayment": "PREPAYMENT_EVENT_LABEL"
}

# Get the final dataset from session state for the active model
final_dataset = st.session_state.get(f"{active_model}_final_dataset", pd.DataFrame())

if not final_dataset.empty:
    # Allow the user to select a target variable
    target_column = st.selectbox("Select Target Variable", list(target_variable_mapping.keys()), key=f"target_column_select_{active_model}")

    if st.button("Add Target Variable to Dataset", key=f"add_target_btn_{active_model}"):
        try:
            # Get the target feature name from the mapping
            target_feature = target_variable_mapping[target_column]

            # Get the combined dataset from session state for the active model
            combined_data_with_target = st.session_state.get(f"{active_model}_recommended_features", pd.DataFrame()).copy()

            # Add target column to the combined dataset if not already present
            if target_feature not in combined_data_with_target.columns:
                # This is a placeholder. You would typically load this from your raw data or another source.
                # For demonstration, I'll add a dummy column.
                # Replace this with your actual logic to get the target variable data.
                if not combined_data_with_target.empty:
                    combined_data_with_target[target_feature] = np.random.randint(0, 2, len(combined_data_with_target)) # Dummy binary target
                    st.warning(f"Added a dummy '{target_feature}' column. Please ensure this data is correctly sourced.")
                else:
                    st.error("Cannot add target variable to an empty dataset.")
                    combined_data_with_target = pd.DataFrame() # Ensure it's empty on error

            # Update the final dataset in session state to include the target column
            # Ensure only selected features + target are in the final dataset
            selected_features_including_mandatory = st.session_state.get(f"{active_model}_selected_features", []) + present_mandatory_features
            features_for_final_dataset = [f for f in selected_features_including_mandatory if f in combined_data_with_target.columns]
            if target_feature in combined_data_with_target.columns:
                 features_for_final_dataset.append(target_feature)
                 # Remove duplicates while preserving order (optional, but good practice)
                 features_for_final_dataset = list(dict.fromkeys(features_for_final_dataset))


            if features_for_final_dataset and not combined_data_with_target.empty:
                st.session_state[f"{active_model}_final_dataset"] = combined_data_with_target[features_for_final_dataset].copy()
            else:
                st.warning("No features selected to create the final dataset with target.")
                st.session_state[f"{active_model}_final_dataset"] = pd.DataFrame() # Ensure it's empty


            # Store target variable in model-specific session state
            st.session_state[f"{active_model}_target_column"] = target_column
            st.session_state[f"{active_model}_target_feature"] = target_feature

            # Convert final dataset (with target) to JSON and store for Model_develop page
            final_dataset_with_target = st.session_state.get(f"{active_model}_final_dataset", pd.DataFrame())
            if not final_dataset_with_target.empty:
                 final_json = final_dataset_with_target.to_json(orient="records")
                 st.session_state[f"{active_model}_final_dataset_json"] = final_json

                 # Save the JSON file to the backend with model name and target variable
                 # Sanitize model name and target column for filename
                 safe_model_name = re.sub(r'\W+', '_', active_model) # Replace non-alphanumeric with underscore
                 safe_target_column = re.sub(r'\W+', '', target_column) # Remove non-alphanumeric
                 file_name = f"{safe_model_name}_{safe_target_column}.json"

                 # Ensure the model_states directory exists before saving
                 if not os.path.exists("model_states"):
                      os.makedirs("model_states")

                 file_path_to_save = os.path.join("model_states", file_name)

                 with open(file_path_to_save, 'w') as f:
                     f.write(final_json)

                 st.success(f"Target variable '{target_column}' added to the final dataset successfully! Dataset saved as '{file_path_to_save}'.")
            else:
                 st.warning("Final dataset is empty, cannot save JSON.")


        except Exception as e:
            st.error(f"Error adding target variable: {str(e)}")
else:
    st.info("Please select and show your features first to enable target variable selection.")