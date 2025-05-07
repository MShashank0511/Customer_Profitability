import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import re
import uuid
import random
from datetime import datetime, timedelta

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
    st.session_state.models = ["Model 1"]  # Start with Model 1
if "active_model" not in st.session_state:
    st.session_state.active_model = "Model 1"  # Default active model
if "operations_complete" not in st.session_state:
    st.session_state.operations_complete = {
        "merge": False,
        "recommend": False,
        "accept": False
    }

# Function to switch_model
def switch_model(model_name):
    st.session_state.active_model = model_name
    # Always reset state when switching models
    st.session_state[f"{model_name}_state"] = {
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
    }
    
    # Reset operations state for the model
    st.session_state[f"{model_name}_operations_complete"] = {
        "merge": False,
        "recommend": False,
        "accept": False
    }
    
    # Reset other session state variables for the model
    st.session_state[f"{model_name}_show_filter_data"] = False
    st.session_state[f"{model_name}_show_merge"] = False
    st.session_state[f"{model_name}_single_transform_success"] = None
    st.session_state[f"{model_name}_multi_transform_success"] = None
    st.session_state[f"{model_name}_recommended_features"] = pd.DataFrame()
    st.session_state[f"{model_name}_final_dataset"] = pd.DataFrame()
    st.session_state[f"{model_name}_selected_features"] = []
    st.session_state[f"{model_name}_feature_checkboxes"] = {}
    
    st.rerun()

# Function to add a new model
def add_new_model():
    new_model_name = f"Model {len(st.session_state.models) + 1}"
    st.session_state.models.append(new_model_name)
    st.session_state.active_model = new_model_name
    # Reset all state for the new model
    st.session_state[f"{new_model_name}_state"] = {
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
    }
    
    # Initialize operations state for new model
    st.session_state[f"{new_model_name}_operations_complete"] = {
        "merge": False,
        "recommend": False,
        "accept": False
    }
    
    # Initialize other session state variables for new model
    st.session_state[f"{new_model_name}_show_filter_data"] = False
    st.session_state[f"{new_model_name}_show_merge"] = False
    st.session_state[f"{new_model_name}_single_transform_success"] = None
    st.session_state[f"{new_model_name}_multi_transform_success"] = None
    st.session_state[f"{new_model_name}_recommended_features"] = pd.DataFrame()
    st.session_state[f"{new_model_name}_final_dataset"] = pd.DataFrame()
    st.session_state[f"{new_model_name}_selected_features"] = []
    st.session_state[f"{new_model_name}_feature_checkboxes"] = {}
    
    st.rerun()

# --- Display model buttons at the top ---
# Create a row of columns for model buttons
cols = st.columns(len(st.session_state.models) + 1)  # +1 for the add button

# Display model buttons in a row
for i, model_name in enumerate(st.session_state.models):
    with cols[i]:
        if st.button(model_name, key=f"btn_{model_name}", use_container_width=True):
            switch_model(model_name)

# Add button in the last column
with cols[-1]:
    if st.button("➕", key="add_model_btn", use_container_width=True):
        add_new_model()

st.markdown("---")  # Add a separator after the model buttons

# Initialize model state if it doesn't exist
active_model = st.session_state.active_model
if f"{active_model}_state" not in st.session_state:
    st.session_state[f"{active_model}_state"] = {
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
    }

# Use the active model's state for all operations
model_state = st.session_state[f"{active_model}_state"]

# Get model-specific session state variables
operations_complete = st.session_state.get(f"{active_model}_operations_complete", {
    "merge": False,
    "recommend": False,
    "accept": False
})
show_filter_data = st.session_state.get(f"{active_model}_show_filter_data", False)
show_merge = st.session_state.get(f"{active_model}_show_merge", False)
single_transform_success = st.session_state.get(f"{active_model}_single_transform_success", None)
multi_transform_success = st.session_state.get(f"{active_model}_multi_transform_success", None)

# --- Initialize session state ---
if "loan_data" not in model_state:
    model_state["loan_data"] = pd.read_csv("loan_data.csv")

loan_data = model_state["loan_data"]  # Initial load

# --- Initialize the datasets in session state if they don't exist ---
if "bureau_data" not in model_state:
    model_state["bureau_data"] = loan_data.copy()  # Initialize with a copy of loan_data
if "onus_data" not in model_state:
    model_state["onus_data"] = loan_data.copy()  # Initialize with a copy of loan_data
if "installments_data" not in model_state:
    model_state["installments_data"] = loan_data.copy()  # Initialize with a copy of loan_data

# --- Initialize Session State ---
if "show_popup1" not in model_state:
    model_state["show_popup1"] = False  # Default to False
if "transform_blocks" not in model_state:
    model_state["transform_blocks"] = []  # Initialize as an empty list
if "multi_transform_blocks" not in model_state:
    model_state["multi_transform_blocks"] = []  # Initialize as an empty list
if "final_transformed_features" not in model_state:
    model_state["final_transformed_features"] = pd.DataFrame()  # Initialize as an empty DataFrame

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
    """
    # --- Initialize filter blocks in session state ---
    if "filter_blocks" not in st.session_state:
        st.session_state.filter_blocks = [{
            "dataset": bureau_name,  # Default dataset. Use the name, not the data.
            "feature": model_state["bureau_data"].columns[0] if not model_state["bureau_data"].empty else "",
            "operation": "Greater Than",
            "value": 0,
            "output_name": ""
        }]

    # --- Define available operations ---
    OPERATIONS = ["Greater Than", "Less Than", "Equal To", "Not Equal To",
                  "Greater Than or Equal To", "Less Than or Equal To"]

    st.header("Filter Data")

    # Use a container for the filter controls
    filter_container = st.container()

    # Create a mapping of dataset names to the actual DataFrames. Crucial for dynamic access.
    dataset_mapping = {
        bureau_name: model_state["bureau_data"],
        onus_name: model_state["onus_data"],
        installments_name: model_state["installments_data"],
    }

    with filter_container:
        for i, filter_block in enumerate(st.session_state.filter_blocks):
            st.subheader(f"Filter {i + 1}")
            cols = st.columns([1, 2, 2, 2, 2, 2])  # Rearranged columns

            with cols[0]:
                if st.button("❌", key=f"remove_filter_{i}"):
                    st.session_state.filter_blocks.pop(i)
                    st.rerun()
            with cols[1]:
                selected_dataset_name = st.selectbox("Select Table", list(dataset_mapping.keys()),
                                                     index=list(dataset_mapping.keys()).index(filter_block["dataset"]),
                                                     key=f"dataset_{i}")

                # Get the selected DataFrame using the name
                selected_dataset = dataset_mapping[selected_dataset_name]

                # Get the features from the selected DataFrame
                available_features = selected_dataset.columns.tolist() if not selected_dataset.empty else []

                # Update the selected feature if the dataset changes and the previous feature is not in the new dataset
                if filter_block["feature"] not in available_features:
                    filter_block["feature"] = available_features[0] if available_features else ""

            with cols[2]:
                selected_feature = st.selectbox("Select Feature", available_features,
                                                 index=available_features.index(filter_block["feature"]) if filter_block["feature"] in available_features else 0,
                                                 key=f"feature_{i}")
            with cols[3]:
                operation = st.selectbox("Select Operation", OPERATIONS,
                                         index=OPERATIONS.index(filter_block["operation"]),
                                         key=f"operation_{i}")
            with cols[4]:
                value = st.number_input("Select Value", value=filter_block["value"], key=f"value_{i}")
            with cols[5]:
                # Calculate output_name dynamically
                output_name = st.text_input("Output Feature",
                                           value=f"{selected_feature}{operation.replace(' ', '')}",
                                           key=f"output_name_{i}")

            # Update the filter block in session state
            st.session_state.filter_blocks[i] = {
                "dataset": selected_dataset_name,  # Store the name of the dataset
                "feature": selected_feature,
                "operation": operation,
                "value": value,
                "output_name": output_name  # Use the updated output_name
            }

        if st.button("+ Add Filter"):
            st.session_state.filter_blocks.append({
                "dataset": bureau_name,  # Default dataset name
                "feature": model_state["bureau_data"].columns[0] if not model_state["bureau_data"].empty else "",
                "operation": "Greater Than",
                "value": 0,
                "output_name": ""
            })
            st.rerun()

        if st.button("Apply All Filters"):
            # Create a copy to avoid modifying the original dataframes in session state
            filtered_datasets = {
                name: df.copy() for name, df in dataset_mapping.items()
            }

            for filter_block in st.session_state.filter_blocks:
                dataset_name = filter_block["dataset"]  # Get dataset name from filter block
                feature = filter_block["feature"]
                operation = filter_block["operation"]
                value = filter_block["value"]
                output_name = filter_block["output_name"]

                # Get the dataframe to be modified
                df = filtered_datasets[dataset_name]

                # Perform the filtering operation
                if operation == "Greater Than":
                    df[output_name] = df[feature] > value
                elif operation == "Less Than":
                    df[output_name] = df[feature] < value
                elif operation == "Equal To":
                    df[output_name] = df[feature] == value
                elif operation == "Not Equal To":
                    df[output_name] = df[feature] != value
                elif operation == "Greater Than or Equal To":
                    df[output_name] = df[feature] >= value
                elif operation == "Less Than or Equal To":
                    df[output_name] = df[feature] <= value
                else:
                    st.warning(f"Operation '{operation}' not supported. Skipping filter.")
                    continue  # Skip to the next filter block if the operation is unsupported

                # Update the dataset in the dictionary
                filtered_datasets[dataset_name] = df

            # Update the session state with the filtered datasets
            model_state["bureau_data"] = filtered_datasets[bureau_name].copy()  # Make explicit copies
            model_state["onus_data"] = filtered_datasets[onus_name].copy()
            model_state["installments_data"] = filtered_datasets[installments_name].copy()

            st.success("All filters applied!")


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
if "show_merge" not in st.session_state:
    st.session_state.show_merge = False
if "merge_blocks" not in st.session_state:
    # Initialize with one default merge block
    st.session_state.merge_blocks = [{
        "left_table": "Bureau Data (Filtered)", # Default to the filtered data
        "right_table": "On-Us Data (Filtered)", # Default to the filtered data
        "how": "inner",
        "on": [],
        "left_on": [],
        "right_on": [],
        "merged_name": "Merged_1",
    }]
if "merged_tables" not in st.session_state:
    st.session_state.merged_tables = {}
if "combined_dataset" not in st.session_state:
    st.session_state.combined_dataset = None # This will hold the final merged result

# Add this callback function near the top with other callback functions
def show_merge_callback():
    st.session_state.show_merge = not st.session_state.show_merge  # Toggle visibility

# Then modify the Merge Datasets button to use the callback
# --- Merge Datasets Section ---
# Merge Datasets Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.button("🔄 Merge Datasets", key="merge_btn", on_click=show_merge_callback, use_container_width=True)

# Display the merge section if show_merge is True
if st.session_state.show_merge:
    # Removed the title "### 🔄 Merge Datasets"
    
    # Prepare available tables for merging
    # Start with the filtered datasets
    available_tables = {
        "Bureau Data": model_state["bureau_data"],
        "On-Us Data": model_state["onus_data"],
        "Installments Data": model_state["installments_data"],
    }
    # Add any tables previously created by merge operations
    available_tables.update(st.session_state.merged_tables)
    table_names = list(available_tables.keys())

    # Default suffixes for handling duplicate column names after merge
    default_suffixes = ("_x", "_y")

    # --- Display and Configure Merge Operations ---
    # Iterate through each merge block defined in session state
    for i, block in enumerate(st.session_state.merge_blocks):
        st.markdown(f"---")  # Separator for clarity between merge blocks

        # Use two rows of three columns for better layout
        row1_col1, row1_col2, row1_col3 = st.columns([0.2, 1, 1])
        row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])

        with row1_col1:
            # Cross Button to Remove Iteration
            if st.button("❌", key=f"remove_merge_{i}"):
                st.session_state.merge_blocks.pop(i)
                st.rerun()

        with row1_col2:
            # Select the left table for the current merge operation
            left_table = st.selectbox(
                "Left Table",
                table_names,  # Options are all available table names
                key=f"merge_left_table_{i}",
                # Set default based on saved state or fallback to the first available table
                index=table_names.index(block.get("left_table", table_names[0])) if block.get("left_table", table_names[0]) in table_names else 0
            )

        with row1_col3:
            # Select the right table for the current merge operation
            right_table = st.selectbox(
                "Right Table",
                table_names,  # Options are all available table names
                key=f"merge_right_table_{i}",
                # Set default based on saved state or fallback to the first available table
                index=table_names.index(block.get("right_table", table_names[0])) if block.get("right_table", table_names[0]) in table_names else 0
            )

        # Get column names for the selected left and right tables
        left_cols = available_tables[left_table].columns.tolist()
        right_cols = available_tables[right_table].columns.tolist()
        # Find common columns between the two selected tables
        common_cols = list(set(left_cols) & set(right_cols))

        with row2_col1:
            # Select columns present in both tables to join on ('on' parameter)
            on = st.selectbox(
                "Column to Join On",
                common_cols,  # Options are common columns
                index=common_cols.index(block.get("on", [])[0]) if block.get("on") and block["on"][0] in common_cols else 0,
                key=f"merge_on_{i}",
                help="Select a single column present in both tables to join on."
            )

        with row2_col2:
            # Select columns from the left table to join on ('left_on' parameter)
            left_on = st.multiselect(
                "Left Columns",
                left_cols,  # Options are columns from the left table
                default=block.get("left_on", []),  # Default to saved values
                key=f"merge_left_on_{i}",
                help="Select multiple columns from the left table to join on."
            )

        with row2_col3:
            # Select columns from the right table to join on ('right_on' parameter)
            right_on = st.multiselect(
                "Right Columns",
                right_cols,  # Options are columns from the right table
                default=block.get("right_on", []),  # Default to saved values
                key=f"merge_right_on_{i}",
                help="Select multiple columns from the right table to join on."
            )

        # Select the type of join (how)
        how = st.selectbox(
            "Join Type",
            ["inner", "left", "right", "outer", "cross"],
            key=f"merge_how_{i}",
            # Set default based on saved state
            index=["inner", "left", "right", "outer", "cross"].index(block.get("how", "inner"))
        )

        # --- Name the Resulting DataFrame Dynamically ---
        default_merged_name = f"{left_table}merged{right_table}_{how}"
        merged_name = st.text_input(
            "Resulting DataFrame Name",
            value=block.get("merged_name", default_merged_name),  # Default to saved or generated name
            key=f"merge_merged_name_{i}",
            help="Name for the DataFrame resulting from this merge operation."
        )

        # --- Update Session State for the Current Block ---
        st.session_state.merge_blocks[i] = {
            "left_table": left_table,
            "right_table": right_table,
            "how": how,
            "on": [on],  # Store as a list for consistency
            "left_on": left_on,
            "right_on": right_on,
            "merged_name": merged_name,
        }

    # --- Buttons to Manage Merge Operations ---
    if st.button("➕ Add Merge Operation", key="add_merge"):
        last_merged_name = st.session_state.merge_blocks[-1].get("merged_name", "Bureau Data (Filtered)") if st.session_state.merge_blocks else "Bureau Data (Filtered)"
        st.session_state.merge_blocks.append({
            "left_table": last_merged_name,
            "right_table": table_names[0] if table_names else "Bureau Data (Filtered)",
            "how": "inner",
            "on": [],
            "left_on": [],
            "right_on": [],
            "merged_name": f"{last_merged_name}merged{table_names[0] if table_names else 'New'}_{len(st.session_state.merge_blocks) + 1}",
        })
        st.rerun()

    if st.button("✅ Merge Now", key="execute_merges", use_container_width=True):
        try:
            # Set merge operation as complete immediately
            st.session_state.operations_complete["merge"] = True
            st.success("✅ Merge operations completed successfully!")
        except Exception as e:
            st.error(f"Error during merge operations: {str(e)}")


    st.markdown("---")

# Move these sections outside the merge block to make them always visible
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
            st.session_state.operations_complete["recommend"] = True
            st.rerun()
            
        except Exception as e:
            st.error(f"Error recommending features: {str(e)}")

# Display recommended features if they exist
if st.session_state.operations_complete.get("recommend", False) and hasattr(st.session_state, 'feature_info'):
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
if st.session_state.operations_complete.get("recommend", False):
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
                    active_model = st.session_state.active_model
                    model_state = st.session_state[f"{active_model}_state"]
                    model_state["recommended_features"] = recommended_features.copy()
                    
                    # Update state
                    st.session_state.operations_complete["accept"] = True
                    
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

    # Initialize success message in session state if not exists
    if "single_transform_success" not in st.session_state:
        st.session_state.single_transform_success = None
    if "multi_transform_success" not in st.session_state:
        st.session_state.multi_transform_success = None

    # Get the recommended features as input for single feature transformations
    if "recommended_features" in st.session_state and not st.session_state.recommended_features.empty:
        input_features = st.session_state.recommended_features.columns.tolist()
    else:
        input_features = []

    # Initialize transform_blocks if empty
    if not model_state["transform_blocks"]:
        model_state["transform_blocks"] = [{
            "feature": "",
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
                for block in model_state["transform_blocks"]:
                    feature = block["feature"]
                    operation = block["operation"]
                    value = block["value"]
                    output_name = block["output_name"]
                    if not feature or not operation or not output_name:
                        continue
                    data = st.session_state.recommended_features[feature]
                    if operation == "Addition":
                        transformed_features[output_name] = data + value
                        successful_transformations.append(f"- {feature} + {value} → {output_name}")
                    elif operation == "Subtraction":
                        transformed_features[output_name] = data - value
                        successful_transformations.append(f"- {feature} - {value} → {output_name}")
                    elif operation == "Multiplication":
                        transformed_features[output_name] = data * value
                        successful_transformations.append(f"- {feature} × {value} → {output_name}")
                    elif operation == "Division":
                        transformed_features[output_name] = data / value
                        successful_transformations.append(f"- {feature} ÷ {value} → {output_name}")
                    elif operation == "Log":
                        transformed_features[output_name] = np.log1p(data)
                        successful_transformations.append(f"- log({feature}) → {output_name}")
                    elif operation == "Square Root":
                        transformed_features[output_name] = np.sqrt(data)
                        successful_transformations.append(f"- sqrt({feature}) → {output_name}")
                    elif operation == "Power":
                        transformed_features[output_name] = np.power(data, value)
                        successful_transformations.append(f"- {feature} ** {value} → {output_name}")
                    elif operation == "Absolute Value":
                        transformed_features[output_name] = np.abs(data)
                        successful_transformations.append(f"- abs({feature}) → {output_name}")
                    elif operation == "Rename":
                        transformed_features[output_name] = data
                        successful_transformations.append(f"- {feature} renamed to {output_name}")
                # Convert to DataFrame and append to recommended_features
                if transformed_features:
                    transformed_df = pd.DataFrame(transformed_features)
                    st.session_state.recommended_features = pd.concat(
                        [st.session_state.recommended_features, transformed_df], axis=1
                    )
                    st.session_state.single_transform_success = "✅ Single feature transformations applied successfully!"
                    # Clear the transform blocks after successful application
                    model_state["transform_blocks"] = []
                    st.rerun()
        except Exception as e:
            st.error(f"Error applying transformations: {str(e)}")

    # Display success message if it exists
    if st.session_state.single_transform_success:
        st.success(st.session_state.single_transform_success)
        st.session_state.single_transform_success = None

    # --- Multi-Feature Transformation Section ---
    st.markdown("### 🔧 Multiple Features Transformation")

    # Initialize multi_transform_blocks if empty
    if not model_state["multi_transform_blocks"]:
        model_state["multi_transform_blocks"] = [{
            "features": [],
            "operation": "",
            "output_name": ""
        }]

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
            operation_prompt = st.text_input(
                "Describe How to Combine Features",
                value=block.get("operation", ""),
                key=f"multi_operation_{i}"
            )
            model_state["multi_transform_blocks"][i]["operation"] = operation_prompt

        with col4:
            if block.get("features") and block.get("operation"):
                suggested_output = f"{operation_prompt.replace(' ', '').lower()}_{'_'.join(selected_features).lower()}"
                output_name = st.text_input(
                    "Name for New Feature",
                    value=block.get("output_name", suggested_output),
                    key=f"multi_output_{i}"
                )
                model_state["multi_transform_blocks"][i]["output_name"] = output_name

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
                final_transformed_features = {}
                successful_transformations = []
                for block in model_state["multi_transform_blocks"]:
                    features = block["features"]
                    operation = block["operation"]
                    output_name = block["output_name"]
                    if not features or not operation or not output_name:
                        continue
                    feature_data = st.session_state.recommended_features[features]
                    if operation.lower() == "add":
                        final_transformed_features[output_name] = feature_data.sum(axis=1)
                        successful_transformations.append(f"- Combined {', '.join(features)} → {output_name} (Addition)")
                    elif operation.lower() == "multiply":
                        final_transformed_features[output_name] = feature_data.prod(axis=1)
                        successful_transformations.append(f"- Combined {', '.join(features)} → {output_name} (Multiplication)")
                    elif operation.lower() == "average":
                        final_transformed_features[output_name] = feature_data.mean(axis=1)
                        successful_transformations.append(f"- Combined {', '.join(features)} → {output_name} (Average)")
                    elif operation.lower() == "max":
                        final_transformed_features[output_name] = feature_data.max(axis=1)
                        successful_transformations.append(f"- Combined {', '.join(features)} → {output_name} (Maximum)")
                    elif operation.lower() == "min":
                        final_transformed_features[output_name] = feature_data.min(axis=1)
                        successful_transformations.append(f"- Combined {', '.join(features)} → {output_name} (Minimum)")
                final_transformed_features_df = pd.DataFrame(final_transformed_features)
                st.session_state.recommended_features = pd.concat(
                    [st.session_state.recommended_features, final_transformed_features_df], axis=1
                )
                combined_dataset_file = f"model_{active_model}_dataset.csv"
                st.session_state.recommended_features.to_csv(combined_dataset_file, index=False)
                st.session_state.multi_transform_success = "✅ Multi-feature transformations applied successfully!"
                model_state["multi_transform_blocks"] = []
                st.rerun()
        except Exception as e:
            st.error(f"Error creating new features: {str(e)}")

    if st.session_state.multi_transform_success:
        st.success(st.session_state.multi_transform_success)
        st.session_state.multi_transform_success = None


# --- Initialize Session State ---
if "final_dataset" not in st.session_state:
    st.session_state.final_dataset = pd.DataFrame()  # Initialize as an empty DataFrame
if "recommended_features" not in st.session_state:
    st.session_state.recommended_features = pd.DataFrame()  # Initialize as an empty DataFrame
if "transformed_features" not in st.session_state:
    st.session_state.transformed_features = pd.DataFrame()  # Initialize as an empty DataFrame
if "final_transformed_features" not in st.session_state:
    st.session_state.final_transformed_features = pd.DataFrame()  # Initialize as an empty DataFrame
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []  # Initialize as an empty list
if "feature_checkboxes" not in st.session_state:
    st.session_state.feature_checkboxes = {}  # Initialize as an empty dictionary
if "show_filter" not in st.session_state:
    st.session_state.show_filter = False
if "filtered_features" not in st.session_state:
    st.session_state.filtered_features = []
if "filter_text" not in st.session_state:
    st.session_state.filter_text = ""

# --- Data Selection Section ---
st.markdown("### 🔎 Feature Selection")

# Load the combined dataset from the backend if available
if "recommended_features" in st.session_state and not st.session_state.recommended_features.empty:
    combined_data = st.session_state.recommended_features
else:
    try:
        combined_data = pd.read_csv("combined_dataset.csv")
        st.session_state.recommended_features = combined_data
    except FileNotFoundError:
        st.error("No combined dataset found. Please complete the 'Data Transformation' section first.")
        combined_data = pd.DataFrame()

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

# Add descriptions for combined features
for feature in combined_data.columns:
    if feature not in feature_descriptions:
        feature_descriptions[feature] = f"Description for {feature}"

# Show mandatory features
st.subheader("📌 Mandatory Features")
st.dataframe(pd.DataFrame({"Mandatory Features": mandatory_features}), hide_index=True)
st.success("All mandatory attributes are available")

st.markdown("---")

# Get all available features from the combined dataset
all_features = combined_data.columns.tolist()
available_optional_features = [feat for feat in all_features if feat not in mandatory_features]

# Initialize feature checkboxes in session state if not exists
if "feature_checkboxes" not in st.session_state:
    st.session_state.feature_checkboxes = {feat: False for feat in available_optional_features}

# Display good-to-have feature selection
st.subheader("✨ Good-to-Have Features")

# Create a dataframe for the features
features_df = pd.DataFrame({
    "Feature": available_optional_features,
    "Description": [feature_descriptions.get(feat, "No description available") for feat in available_optional_features],
    "Select": [bool(st.session_state.feature_checkboxes.get(feat, False)) for feat in available_optional_features]
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
    key="feature_editor"
)

# Update selected features based on checkboxes
st.session_state.selected_features = [
    feature for feature, is_selected in zip(available_optional_features, edited_df["Select"])
    if is_selected
]

st.markdown("---")

# Combine and preview
if st.button("📊 Show Selected Attributes"):
    # Create a summary of all selected features
    all_features = []
    feature_types = []

    # Add mandatory features
    for feature in mandatory_features:
        all_features.append(feature)
        feature_types.append("Mandatory")

    # Add selected optional features
    for feature in st.session_state.selected_features:
        all_features.append(feature)
        feature_types.append("Selected")

    # Create and display the summary dataframe
    summary_df = pd.DataFrame({
        "Feature": all_features,
        "Type": feature_types
    })

    st.subheader("Selected Features Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Store the final dataset in session state
    working_df = combined_data.copy()
    available_features = [f for f in all_features if f in working_df.columns]
    if available_features:
        st.session_state.final_dataset = working_df[available_features]

# Target Variable Selection
if "final_dataset" in st.session_state and not st.session_state.final_dataset.empty:
    st.subheader("🎯 Target Variable Selection")
    
    # Define the target variable options and their corresponding feature names
    target_variable_mapping = {
        "Profitability": "Profitability_GBP",
        "Charge-Off": "COF_EVENT_LABEL",
        "Prepayment": "PREPAYMENT_EVENT_LABEL"
    }
    
    # Allow the user to select a target variable
    target_column = st.selectbox("Select Target Variable", list(target_variable_mapping.keys()), key="target_column_select")

    if st.button("Add Target Variable to Dataset", key="add_target_btn"):
        try:
            # Get the target feature name from the mapping
            target_feature = target_variable_mapping[target_column]
            
            # Create a copy of the final dataset
            model_dataset = st.session_state.final_dataset.copy()
            
            # Add target column to final dataset if not already present
            if target_feature not in model_dataset.columns:
                # Try to get the target feature from the original data
                if target_feature in st.session_state.recommended_features.columns:
                    model_dataset[target_feature] = st.session_state.recommended_features[target_feature]
                else:
                    # For demo purposes, add a dummy column
                    model_dataset[target_feature] = 0

            # Store target variable in session state
            st.session_state.target_column = target_column
            st.session_state.target_feature = target_feature

            # Convert final dataset (with target) to JSON and store for Model_develop page
            final_json = model_dataset.to_json(orient="records")
            st.session_state.final_dataset_json = final_json

            # Save the JSON file to the backend with model name and target variable
            file_name = f"{active_model}{target_column.replace(' ', '')}.json"
            with open(file_name, "w") as f:
                f.write(final_json)

            st.success(f"✅ Target variable '{target_column}' has been added to your dataset.")
            
        except Exception as e:
            st.error(f"Error adding target variable: {str(e)}")
else:
    st.info("Please select and show your features first to enable target variable selection.")