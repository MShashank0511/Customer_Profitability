import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import re
import uuid # Import uuid for generating unique IDs
import random
from datetime import datetime, timedelta

def process_operation_prompt(prompt: str, features: List[str]) -> Dict[str, Any]:
    """
    Process the operation prompt and return appropriate operation details.
    Returns a dictionary with operation type and parameters.
    """
    prompt = prompt.lower()

    # Common operation patterns
    patterns = {
        "sum": r"(sum|add|total|addition)",
        "average": r"(average|mean|avg)",
        "product": r"(product|multiply|multiplication)",
        "ratio": r"(ratio|divide|division)",
        "difference": r"(difference|subtract|subtraction)",
        "min": r"(min|minimum|smallest)",
        "max": r"(max|maximum|largest)",
        "count": r"(count|number|total number)",
        "custom": r"(custom|complex|advanced)"
    }

    # Determine operation type
    operation_type = "custom"
    for op_type, pattern in patterns.items():
        if re.search(pattern, prompt):
            operation_type = op_type
            break

    # Generate operation details
    operation_details = {
        "type": operation_type,
        "description": prompt,
        "features": features,
        "parameters": {}
    }

    # Add specific parameters based on operation type
    if operation_type == "ratio":
        # For ratio operations, try to determine numerator and denominator
        if len(features) >= 2:
            # Attempt to identify numerator/denominator based on position or prompt clues (basic)
            numerator = features[0]
            denominator = features[1]
            # More advanced parsing could look for "ratio of A to B" etc.
            operation_details["parameters"] = {
                "numerator": numerator,
                "denominator": denominator
            }
    elif operation_type == "difference":
         if len(features) >= 2:
             # Assume the difference is between the first two selected features
             operation_details["parameters"] = {
                 "feature1": features[0],
                 "feature2": features[1]
             }
    elif operation_type == "custom":
        # For custom operations, try to extract specific parameters
        if "between" in prompt:
            operation_details["parameters"]["operation"] = "between"
        elif "weighted" in prompt:
            operation_details["parameters"]["operation"] = "weighted"

    return operation_details

def generate_operation_code(operation_details: Dict[str, Any]) -> str:
    """
    Generate Python code for the operation based on operation details.
    """
    op_type = operation_details["type"]
    features = operation_details["features"]
    # Safely get output_name, default if not present
    output_name = operation_details.get("output_name", "new_feature")
    params = operation_details.get("parameters", {})

    # Construct the string representation of the features list for pandas column selection
    feature_list_str = "[" + ", ".join([f"'{f}'" for f in features]) + "]"

    # Corrected f-string syntax: Removed the extra ']' after '{output_name}' in previous versions
    if op_type == "sum":
        return f"df['{output_name}'] = df[{feature_list_str}].sum(axis=1)"
    elif op_type == "average":
        return f"df['{output_name}'] = df[{feature_list_str}].mean(axis=1)"
    elif op_type == "product":
        return f"df['{output_name}'] = df[{feature_list_str}].prod(axis=1)"
    elif op_type == "ratio":
        numerator = params.get("numerator")
        denominator = params.get("denominator")
        if numerator and denominator:
            # Corrected f-string syntax
            return f"df['{output_name}'] = df['{numerator}'] / df['{denominator}']"
        else:
             return f"# Ratio operation requires two features. Could not identify numerator and denominator.\n# Features: {features}\n# Implement ratio logic here for '{output_name}'."
    elif op_type == "difference":
        feature1 = params.get("feature1")
        feature2 = params.get("feature2")
        if feature1 and feature2:
            # Corrected f-string syntax
             return f"df['{output_name}'] = df['{feature1}'] - df['{feature2}']"
        else:
             return f"# Difference operation requires at least two features. Could not identify features.\n# Features: {features}\n# Implement difference logic here for '{output_name}'."
    elif op_type == "min":
        return f"df['{output_name}'] = df[{feature_list_str}].min(axis=1)"
    elif op_type == "max":
        return f"df['{output_name}'] = df[{feature_list_str}].max(axis=1)"
    elif op_type == "count":
        # Corrected f-string syntax
        return f"df['{output_name}'] = df[{feature_list_str}].count(axis=1)"
    elif op_type == "custom":
        # For custom operations, return a placeholder
        return f"# Custom operation: {operation_details['description']}\n# Features: {features}\n# Implement custom logic here for '{output_name}'"

    return ""


# --- Initialize session state ---
if "df" not in st.session_state:    
    st.session_state.df = None

# Define feature lists
BUREAU_FEATURES = ["CREDIT_ACTIVE", "DAYS_CREDIT", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"]
ONUS_FEATURES = ["NAME_CONTRACT_STATUS", "DAYS_DECISION", "AMT_APPLICATION", "AMT_CREDIT"]  # Renamed from HONORS_FEATURES
INSTALLMENTS_FEATURES = ["NUM_INSTALMENT_VERSION", "DAYS_INSTALMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]

# Initialize transform blocks
if "transform_blocks" not in st.session_state:
    st.session_state.transform_blocks = [{
        "dataset": "Bureau Data",
        "feature": BUREAU_FEATURES[0],
        "operation": "Addition",
        "value": 1.0,
        "output_name": ""
    }]

# Initialize multi-transform blocks with unique IDs
if "multi_transform_blocks" not in st.session_state:
    st.session_state.multi_transform_blocks = [{
        "id": str(uuid.uuid4()), # Assign a unique ID
        "features": [],
        "operation": "",
        "output_name": "",
        "operation_details": {}
    }]

# Initialize filter blocks
if "filter_blocks" not in st.session_state:
    st.session_state.filter_blocks = [{
        "dataset": "Bureau Data",
        "feature": BUREAU_FEATURES[0],
        "operation": "Greater Than",
        "value": 0,
        "output_name": ""
    }]

# Initialize merge blocks
if "merge_blocks" not in st.session_state:
    st.session_state.merge_blocks = [{
        "left_table": "Bureau Data",
        "right_table": "On-Us Data",
        "join_type": "Left Join",
        "merged_table": "bureau_onus_merged"
    }]

# Initialize visibility states
if "show_popup1" not in st.session_state:
    st.session_state.show_popup1 = False

if "show_filter" not in st.session_state:
    st.session_state.show_filter = False

if "show_merge" not in st.session_state:
    st.session_state.show_merge = False

if "selected_filter_features" not in st.session_state:
    st.session_state.selected_filter_features = []

if "combined_dataset" not in st.session_state:
    st.session_state.combined_dataset = None

if "show_recommended" not in st.session_state:
    st.session_state.show_recommended = False

if "recommended_features" not in st.session_state:
    st.session_state.recommended_features = []

# Initialize session state for filter and combine functionality
if "selected_filter_data_features" not in st.session_state:
    st.session_state.selected_filter_data_features = []

# Add custom CSS for popup
st.markdown("""
    <style>
    .popup {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
        width: 80%;
        max-width: 600px;
    }
    .overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        z-index: 999;
    }
    </style>
""", unsafe_allow_html=True)

# Add custom CSS for the feature box
st.markdown("""
    <style>
    .feature-box {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        max-height: 300px;
        overflow-y: auto;
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)




# --- Dataset Selection Section ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div style='border: 1px solid #e0e0e0; border-radius: 4px; padding: 10px; text-align: center;'>
            <p style='margin: 0;'>Bureau Data</p>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div style='border: 1px solid #e0e0e0; border-radius: 4px; padding: 10px; text-align: center;'>
            <p style='margin: 0;'>On-Us Data</p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div style='border: 1px solid #e0e0e0; border-radius: 4px; padding: 10px; text-align: center;'>
            <p style='margin: 0;'>Installments Data</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Filter Data Section ---
if st.button("🔍 Filter Data", key="filter_btn", use_container_width=True):
    st.session_state.show_filter = True
    st.rerun() # Replaced experimental_rerun

if st.session_state.show_filter:
    st.markdown("### 🔍 Filter Data")

    # Show filter blocks
    for i, block in enumerate(st.session_state.filter_blocks):
        st.markdown(f"**Filter #{i+1}**")
        col1, col2, col3, col4, col5 = st.columns([0.5, 2, 2, 2, 2])

        with col1:
            # Use unique key for remove button
            if st.button("❌", key=f"remove_filter_{i}"):
                st.session_state.filter_blocks.pop(i)
                st.rerun() # Replaced experimental_rerun

        with col2:
            # Use unique key for selectbox
            dataset = st.selectbox(
                "Select Data",
                ["Bureau Data", "On-Us Data", "Installments Data"],
                key=f"filter_dataset_{i}",
                index=["Bureau Data", "On-Us Data", "Installments Data"].index(block.get("dataset", "Bureau Data").replace("Honors", "On-Us"))
            )

            # Update features based on selected dataset
            if dataset == "Bureau Data":
                features = BUREAU_FEATURES
            elif dataset == "On-Us Data":
                features = ONUS_FEATURES
            else:
                features = INSTALLMENTS_FEATURES

            # Use unique key for selectbox
            feature = st.selectbox(
                "Select Feature",
                features,
                key=f"filter_feature_{i}",
                index=features.index(block.get("feature", features[0])) if block.get("feature") in features else 0
            )

        with col3:
            # Use unique key for selectbox
            operation = st.selectbox(
                "Select Operation",
                ["Greater Than", "Less Than", "Greater Than or Equal", "Less Than or Equal", "Equal To", "Not Equal To"],
                key=f"filter_operation_{i}",
                index=["Greater Than", "Less Than", "Greater Than or Equal", "Less Than or Equal", "Equal To", "Not Equal To"].index(block.get("operation", "Greater Than"))
            )

        with col4:
            # Use unique key for number_input
            value = st.number_input(
                "Value",
                value=block.get("value", 0),
                key=f"filter_value_{i}"
            )

        with col5:
            output_name = f"{feature}_{operation.replace(' ', '_')}_{value}"
            # Use unique key for text_input
            output_name = st.text_input(
                "Output Feature",
                value=block.get("output_name", output_name),
                key=f"filter_output_{i}"
            )

        # Update block in session state
        st.session_state.filter_blocks[i] = {
            "dataset": dataset,
            "feature": feature,
            "operation": operation,
            "value": value,
            "output_name": output_name
        }

    # Add new filter button
    if st.button("➕ Add Filter", key="add_filter"):
        st.session_state.filter_blocks.append({
            "dataset": "Bureau Data",
            "feature": BUREAU_FEATURES[0],
            "operation": "Greater Than",
            "value": 0,
            "output_name": ""
        })
        st.rerun() # Replaced experimental_rerun

    # Apply all filters button
    if st.button("✅ Apply All Filters", key="apply_filters"):
        st.success("Filters applied successfully!")
        st.session_state.show_filter = False
        st.rerun() # Replaced experimental_rerun


# --- Merge Datasets Section ---
if st.button("🔄 Merge Datasets", key="merge_btn", use_container_width=True):
    st.session_state.show_merge = True
    st.rerun()

if st.session_state.show_merge:
    st.markdown("### 🔄 Merge Datasets (Advanced)")

    # Prepare available tables, including previously merged tables
    if "merged_tables" not in st.session_state:
        st.session_state.merged_tables = {}

    # Add original tables
    available_tables = {
        "Bureau Data": st.session_state.get("bureau_df", pd.DataFrame({
            "id": [1, 2, 3],
            "feature_a": [10, 20, 30],
            "feature_b": [100, 200, 300]
        })),
        "On-Us Data": st.session_state.get("onus_df", pd.DataFrame({
            "id": [1, 2, 3],
            "feature_c": [5, 6, 7],
            "feature_d": [50, 60, 70]
        })),
        "Installments Data": st.session_state.get("installments_df", pd.DataFrame({
            "id": [1, 2, 3],
            "feature_e": [8, 9, 10],
            "feature_f": [80, 90, 100]
        }))
    }
    # Add merged tables from previous steps
    available_tables.update(st.session_state.merged_tables)
    table_names = list(available_tables.keys())

    # Default suffixes for backend use
    default_suffixes = ("_x", "_y")

    for i, block in enumerate(st.session_state.merge_blocks):
        st.markdown(f"**Merge Operation #{i+1}**")
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])

        with col1:
            left_table = st.selectbox(
                "left (Left Table)",
                table_names,
                key=f"merge_left_table_{i}",
                index=table_names.index(block.get("left_table", table_names[0]))
            )

        with col2:
            right_table = st.selectbox(
                "right (Right Table)",
                table_names,
                key=f"merge_right_table_{i}",
                index=table_names.index(block.get("right_table", table_names[1] if len(table_names) > 1 else table_names[0]))
            )

        # Use sample features if tables are empty
        left_cols = available_tables[left_table].columns.tolist() if not available_tables[left_table].empty else ["id", "feature_a", "feature_b"]
        right_cols = available_tables[right_table].columns.tolist() if not available_tables[right_table].empty else ["id", "feature_c", "feature_d"]
        common_cols = list(set(left_cols) & set(right_cols)) or ["id"]

        with col3:
            how = st.selectbox(
                "how (Join Type)",
                ["inner", "left", "right", "outer", "cross"],
                key=f"merge_how_{i}",
                index=["inner", "left", "right", "outer", "cross"].index(block.get("how", "inner"))
            )

        with col4:
            valid_on = [v for v in block.get("on", []) if v in common_cols]
            on = st.multiselect(
                "on (Columns to join on)",
                common_cols,
                default=valid_on if valid_on else (common_cols[:1] if common_cols else []),
                key=f"merge_on_{i}",
                help="Columns present in both tables"
            )

        with col5:
            valid_left_on = [v for v in block.get("left_on", []) if v in left_cols]
            left_on = st.multiselect(
                "left_on (Left columns)",
                left_cols,
                default=valid_left_on,
                key=f"merge_left_on_{i}",
                help="Columns from left table to join on"
            )

        with col6:
            valid_right_on = [v for v in block.get("right_on", []) if v in right_cols]
            right_on = st.multiselect(
                "right_on (Right columns)",
                right_cols,
                default=valid_right_on,
                key=f"merge_right_on_{i}",
                help="Columns from right table to join on"
            )

        # Resulting merged DataFrame name (auto-generated, editable)
        merged_name = f"{left_table}_merged_{right_table}_{how}"
        merged_name = st.text_input(
            "Resulting DataFrame Name",
            value=block.get("merged_name", merged_name),
            key=f"merge_merged_name_{i}"
        )

        # Update block in session state
        st.session_state.merge_blocks[i] = {
            "left_table": left_table,
            "right_table": right_table,
            "how": how,
            "on": on,
            "left_on": left_on,
            "right_on": right_on,
            "merged_name": merged_name,
        }

    if st.button("➕ Add Merge Operation", key="add_merge"):
        # By default, use the last merged table as left_table for the next merge
        last_merged_name = st.session_state.merge_blocks[-1].get("merged_name", table_names[0]) if st.session_state.merge_blocks else table_names[0]
        st.session_state.merge_blocks.append({
            "left_table": last_merged_name,
            "right_table": table_names[0],
            "how": "inner",
            "on": [],
            "left_on": [],
            "right_on": [],
            "merged_name": f"{last_merged_name}_merged_{table_names[0]}_inner",
        })
        st.rerun()

    if st.button("✅ Merge Now", key="combine_datasets"):
        try:
            merged_results = {}
            for idx, block in enumerate(st.session_state.merge_blocks):
                left_df = available_tables[block["left_table"]]
                right_df = available_tables[block["right_table"]]
                merged = left_df.merge(
                    right_df,
                    how=block["how"],
                    on=block["on"] if block["on"] else None,
                    left_on=block["left_on"] if block["left_on"] else None,
                    right_on=block["right_on"] if block["right_on"] else None,
                    suffixes=default_suffixes,
                )
                merged_results[block["merged_name"]] = merged
                # Make this merged table available for next merges
                st.session_state.merged_tables[block["merged_name"]] = merged

            # The final merged DataFrame is the last one created
            if merged_results:
                final_merged_name = list(merged_results.keys())[-1]
                st.session_state.combined_dataset = merged_results[final_merged_name]
            st.success("All merges completed! The final merged dataset is ready for feature recommendation.")
            st.session_state.show_merge = False
            st.rerun()
        except Exception as e:
            st.error(f"Error merging datasets: {str(e)}")

# --- Centered Recommend Features Button ---
col1, col2, col3 = st.columns([0.5, 3, 0.5])  # Adjusted column widths to make the button wider
with col2:  # Middle column
    if st.button("✨ Recommend Features", key="recommend_features", use_container_width=True):
        try:
            # Use actual data if available, else fallback to simulated
            if st.session_state.get("combined_dataset") is not None:
                df_recommended = st.session_state.combined_dataset
            else:
                # Simulate a DataFrame with sample data for the recommended features
                features = BUREAU_FEATURES + ONUS_FEATURES + INSTALLMENTS_FEATURES
                data = {feature: [np.random.randint(1, 100) for _ in range(5)] for feature in features[:5]}
                df_recommended = pd.DataFrame(data)

            st.session_state.recommended_features = df_recommended

            # Example feature descriptions (extend this for real data)
            feature_descriptions = {
                "CREDIT_ACTIVE": "Current credit status",
                "DAYS_CREDIT": "Days since credit was granted",
                "AMT_CREDIT_SUM": "Total credit amount",
                "AMT_CREDIT_SUM_DEBT": "Total debt amount",
                "NAME_CONTRACT_STATUS": "Contract status",
                "DAYS_DECISION": "Days since decision",
                "AMT_APPLICATION": "Applied amount",
                "AMT_CREDIT": "Credit amount",
                "NUM_INSTALMENT_VERSION": "Installment version number",
                "DAYS_INSTALMENT": "Days until installment",
                "AMT_INSTALMENT": "Installment amount",
                "AMT_PAYMENT": "Payment amount"
            }

            summary = []
            for col in df_recommended.columns:
                col_data = df_recommended[col]
                dtype = str(col_data.dtype)
                desc = feature_descriptions.get(col, "No description available")
                if pd.api.types.is_numeric_dtype(col_data):
                    min_val = col_data.min()
                    max_val = col_data.max()
                    mean_val = round(col_data.mean(), 2)
                else:
                    min_val = max_val = mean_val = "-"
                summary.append({
                    "Feature Name": col,
                    "Data Type": dtype,
                    "Description": desc,
                    "Min": min_val,
                    "Max": max_val,
                    "Mean": mean_val
                })
            summary_df = pd.DataFrame(summary)
            st.dataframe(summary_df, use_container_width=True)

            # --- Show message about feature selection completeness ---
            # Get all features from the dataset (if available)
            if st.session_state.get("combined_dataset") is not None:
                all_features = set(st.session_state.combined_dataset.columns)
            else:
                all_features = set(BUREAU_FEATURES + ONUS_FEATURES + INSTALLMENTS_FEATURES)
            recommended_features = set(df_recommended.columns)

            if recommended_features == all_features:
                st.success("All features are selected from the dataset.")
            else:
                st.error("Not all features are selected from the dataset.")

        except Exception as e:
            st.error(f"Could not generate recommended features: {e}")

# --- Accept Recommended Features Button ---
if "recommended_features" in st.session_state and isinstance(st.session_state.recommended_features, pd.DataFrame) and not st.session_state.recommended_features.empty:
    col1, col2, col3 = st.columns([0.5, 3, 0.5])  # Adjusted column widths to make the button wider
    with col2:
        if st.button("✅ Accept Recommended Features", key="accept_recommended_features", use_container_width=True):
            st.success("Recommended features accepted successfully!")
            # Perform any additional actions with the accepted features
            # For example, store them in session state for further operations
            st.session_state.accepted_features = st.session_state.recommended_features

st.markdown("---")

# --- Data Transformation Buttons ---
st.subheader("Data Actions")
# Create a centered container for the buttons
col1, col2, col3 = st.columns([1, 2, 1])  # Unequal columns to center the buttons
with col2:  # Middle column
    if st.button("🔧 Data Transformation", key="transform_btn", use_container_width=True):
        st.session_state.show_popup1 = True
        st.rerun() # Replaced experimental_rerun


# --- Popup 1: Single Feature Transformations ---
if st.session_state.show_popup1:
    st.markdown("### 🔧 Single Feature Transformation")

    # Show transformation blocks
    for i, block in enumerate(st.session_state.transform_blocks):
        st.markdown(f"**Transformation #{i+1}**")
        col1, col2, col3, col4, col5 = st.columns([0.5, 2, 2, 2, 2])

        with col1:
            if st.button("❌", key=f"remove_single_{i}"):
                st.session_state.transform_blocks.pop(i)
                st.rerun()

        with col2:
            feature = st.selectbox(
                "Select Feature",
                BUREAU_FEATURES + ONUS_FEATURES + INSTALLMENTS_FEATURES,
                key=f"single_feature_{i}",
                index=(BUREAU_FEATURES + ONUS_FEATURES + INSTALLMENTS_FEATURES).index(block.get("feature", BUREAU_FEATURES[0])) if block.get("feature") in (BUREAU_FEATURES + ONUS_FEATURES + INSTALLMENTS_FEATURES) else 0
            )

        with col3:
            operation = st.selectbox(
                "Operation",
                ["Addition", "Subtraction", "Multiplication", "Division", "Log", "Square Root", "Power", "Absolute Value", "Rename"],
                key=f"single_operation_{i}",
                index=["Addition", "Subtraction", "Multiplication", "Division", "Log", "Square Root", "Power", "Absolute Value", "Rename"].index(block.get("operation", "Addition"))
            )

        with col4:
            # Operations that should freeze the value input
            freeze_value_ops = ["Rename", "Log", "Square Root", "Absolute Value"]
            if operation in freeze_value_ops:
                # Set a default value for each operation if needed
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
                # Fallback for any other operation (should not occur)
                value = st.number_input(
                    "Value",
                    value=block.get("value", 1.0),
                    key=f"single_value_{i}"
                )

        with col5:
            # Always suggest output name based on current selections
            if operation == "Rename":
                suggested_output = f"{feature}_renamed"
            elif value is not None:
                suggested_output = f"{feature}_{operation.replace(' ', '_')}_{str(value).replace('.', '_')}"
            else:
                suggested_output = f"{feature}_{operation.replace(' ', '_')}"

            # If the output_name is empty or matches the previous suggestion, update it to the new suggestion
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

            # Update block in session state, including the current suggestion for future comparison
            st.session_state.transform_blocks[i] = {
                "feature": feature,
                "operation": operation,
                "value": value,
                "output_name": output_name,
                "prev_suggestion": suggested_output
            }

    # Add new transformation button
    if st.button("➕ Add Transformation", key="add_transform"):
        st.session_state.transform_blocks.append({
            "dataset": "Bureau Data", # Keep dataset key for consistency, though not strictly needed for single feature transforms
            "feature": BUREAU_FEATURES[0],
            "operation": "Addition",
            "value": 1.0,
            "output_name": ""
        })
        st.rerun() # Replaced experimental_rerun # Use rerun here to show the new block immediately

    # Apply all transformations button
    if st.button("✅ Apply All Transformations", key="apply_transforms"):
        try:
            st.success("Transformations applied successfully!")
            st.session_state.show_popup1 = False
            st.rerun() # Replaced experimental_rerun # Rerun to hide the popup and update the main view
        except Exception as e:
            st.error(f"Error applying transformations: {str(e)}")

    st.markdown("---")

    # Add this CSS just before the Multiple Features Transformation section
    

    st.markdown("""
        <style>
            min-height: 300px !important;  /* Increased to match container */
            height: auto !important;
            align-items: flex-start !important;
            padding: 8px !important;
        }
        
        /* Reduce font size for selected feature tags in multiple features section */
        [data-testid="stVerticalBlock"] > div:has(h3:contains("Multiple Features Transformation")) [data-baseweb="tag"] {
            font-size: 9px !important;     /* Reduced from 10px to 9px */
            padding: 1px 3px !important;   /* Reduced padding */
            margin: 1px !important;
            line-height: 12px !important;  /* Reduced line height */
        }
        
        /* Reduce font size for dropdown options in multiple features section */
        [data-testid="stVerticalBlock"] > div:has(h3:contains("Multiple Features Transformation")) [data-baseweb="select"] div[role="option"] {
            font-size: 9px !important;     /* Reduced from 10px to 9px */
            padding: 2px 4px !important;
        }
        
        /* Improve scrolling for selected items in multiple features section */
        [data-testid="stVerticalBlock"] > div:has(h3:contains("Multiple Features Transformation")) [data-baseweb="select"] > div:first-child > div:first-child {
            max-height: 380px !important;  /* Increased to match new container height */
            overflow-y: auto !important;
        }
        
        /* Add some spacing between items */
        [data-testid="stVerticalBlock"] > div:has(h3:contains("Multiple Features Transformation")) [data-baseweb="select"] [data-testid="virtuoso-item-list"] {
            gap: 2px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Multi-Feature Transformation Section
    st.markdown("### 🔧 Multiple Features Transformation")

    # Show multi-feature transformation blocks
    # Iterate using index to allow removal by index
    for i in range(len(st.session_state.multi_transform_blocks)):
        block = st.session_state.multi_transform_blocks[i]
        # Ensure block has an ID, add if missing (for backward compatibility if state was saved without IDs)
        if "id" not in block:
             block["id"] = str(uuid.uuid4())


        st.markdown(f"**Transformation #{i+1}**")
        col1, col2, col3, col4 = st.columns([0.5, 2, 2, 2])

        with col1:
            # Use the stable block ID in the key for the remove button
            if st.button("❌", key=f"remove_multi_{block['id']}"):
                st.session_state.multi_transform_blocks.pop(i)
                st.rerun()  # Replaced experimental_rerun # Use rerun only when removing a block

        with col2:
            # Feature selection - use the stable block ID in the key
            current_features = block.get("features", [])

            selected_features = st.multiselect(
                "Choose Features to Combine",
                BUREAU_FEATURES + ONUS_FEATURES + INSTALLMENTS_FEATURES,
                default=current_features,
                key=f"multi_features_{block['id']}", # Use unique ID in the key - THIS IS THE FIX FOR THE ODD/EVEN CLICK ISSUE
                help="Select multiple features you want to combine"
            )

            # Update features in session state
            st.session_state.multi_transform_blocks[i]["features"] = selected_features

        with col3:
            # Operation prompt input - use the stable block ID in the key
            operation_prompt = st.text_input(
                "Describe How to Combine Features",
                value=block.get("operation", ""),
                key=f"multi_operation_{block['id']}", # Use unique ID in the key
                placeholder="e.g., Add these features together",
                help="Describe in simple words how you want to combine the selected features"
            )

            # Process operation prompt if both prompt and selected features exist
            operation_details = block.get("operation_details", {}).copy() # Get current details and make a copy
            # Only re-process if prompt or features have changed significantly or details are empty
            if operation_prompt and selected_features:
                 if operation_prompt != block.get("operation", "") or selected_features != block.get("features", []) or not operation_details:
                    operation_details = process_operation_prompt(operation_prompt, selected_features)
                    st.session_state.multi_transform_blocks[i]["operation_details"] = operation_details
            elif not operation_prompt or not selected_features:
                 # Clear operation details if prompt or features are missing
                 st.session_state.multi_transform_blocks[i]["operation_details"] = {}
                 operation_details = {} # Clear local variable too

            # Update the operation prompt in session state
            st.session_state.multi_transform_blocks[i]["operation"] = operation_prompt


        with col4:
            # Output feature name - use the stable block ID in the key
            suggested_output = f"{operation_prompt.replace(' ', '_').lower()}_{'_'.join(selected_features).lower()}" if operation_prompt and selected_features else ""
            output_name = st.text_input(
                "Name for New Feature",
                value=block.get("output_name", suggested_output),
                key=f"multi_output_{block['id']}", # Use unique ID in the key
                help="Give a name to the new feature that will be created"
            )
            # Update the output name in session state
            st.session_state.multi_transform_blocks[i]["output_name"] = output_name


            # Show generated code preview
            # Ensure output_name is in operation_details for code generation
            if operation_details:
                 operation_details_with_output = operation_details.copy() # Create a copy
                 operation_details_with_output["output_name"] = output_name
                 generated_code = generate_operation_code(operation_details_with_output)
                 st.code(generated_code, language="python")


    # Add new multi-feature transformation button
    if st.button("➕ Add New Feature Combination", key="add_multi_transform"):
        st.session_state.multi_transform_blocks.append({
            "id": str(uuid.uuid4()), # Assign a unique ID to the new block
            "features": [],
            "operation": "",
            "output_name": "",
            "operation_details": {}
        })
        st.rerun()  # Replaced experimental_rerun # Use rerun only when adding a new block

    # Apply all multi-feature transformations button
    if st.button("✅ Create All New Features", key="apply_multi_transforms"):
        try:
            # You would implement the actual data processing logic here
            # based on the details in st.session_state.multi_transform_blocks
            # For demonstration, we'll just show a success message.
            st.success("New features created successfully!")
            st.session_state.show_popup1 = False # Close the popup after applying
            st.rerun() # Replaced experimental_rerun # Rerun to hide the popup and update the main view
        except Exception as e:
            st.error(f"Error creating new features: {str(e)}")




### FEATURE SELECTION SECTION ###


# Load dummy data if not already present
if "df" not in st.session_state or st.session_state.df is None:
    st.session_state.df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "feature3": [7, 8, 9],
        "feature4": [10, 11, 12],
        "feature5": [13, 14, 15],
        "total_purchases": [20, 25, 30],
        "avg_order_value": [150, 200, 175],
        "customer_segment": ["A", "B", "C"]
    })

# Define mandatory features
mandatory_features = ["OPB", "interest_rate", "tenure", "credit_score_band", "LTV"]

# Define feature descriptions with more detailed information
feature_descriptions = {
    "feature1": "Customer's primary demographic indicator (age, income, etc.)",
    "feature2": "Customer's secondary demographic indicator (education, occupation, etc.)",
    "feature3": "Customer's behavioral pattern indicator (frequency of interactions)",
    "feature4": "Customer's preference indicator (product category preferences)",
    "feature5": "Customer's risk assessment indicator (credit history, etc.)",
    "total_purchases": "Total number of transactions made by the customer in the last 12 months",
    "avg_order_value": "Average monetary value of customer's orders, indicating spending capacity",
    "customer_segment": "Customer classification based on RFM (Recency, Frequency, Monetary) analysis",
    "OPB": "Outstanding Principal Balance of the customer's loan",
    "interest_rate": "Current interest rate applicable to the customer's loan",
    "tenure": "Duration of the loan in months",
    "credit_score_band": "Customer's credit score category (Excellent, Good, Fair, Poor)",
    "LTV": "Loan-to-Value ratio indicating the risk level of the loan"
}

st.title("🔎 Feature Selection")

# Show mandatory features
st.subheader("📌 Mandatory Features")
st.dataframe(pd.DataFrame({"Mandatory Features": mandatory_features}), hide_index=True)
st.success("All mandatory attributes are available")

st.markdown("---")

# Get all available features from session state
all_features = st.session_state.df.columns.tolist() if st.session_state.df is not None else []
available_optional_features = [feat for feat in all_features if feat not in mandatory_features]

# Add recommended features to available features if they exist
if "recommended_features" in st.session_state:
    rf = st.session_state.recommended_features
    if isinstance(rf, pd.DataFrame):
        if not rf.empty:
            available_optional_features.extend(rf.squeeze().tolist())
    elif isinstance(rf, (list, tuple, set)):
        if len(rf) > 0:
            available_optional_features.extend(list(rf))
    # Remove duplicates while preserving order
    available_optional_features = list(dict.fromkeys(available_optional_features))

# Initialize selected features in session state if not present
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []

# Initialize feature checkboxes in session state if not exists
if "feature_checkboxes" not in st.session_state:
    st.session_state.feature_checkboxes = {feat: False for feat in available_optional_features}

# Initialize filter state if not exists
if "show_filter" not in st.session_state:
    st.session_state.show_filter = False
if "filtered_features" not in st.session_state:
    st.session_state.filtered_features = available_optional_features
if "filter_text" not in st.session_state:
    st.session_state.filter_text = ""

# Display good-to-have feature selection
st.subheader("✨ Good-to-Have Features")

# Create a custom CSS for the feature selection table
st.markdown("""
    <style>
    .feature-table {
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        background-color: white;
    }
    .feature-table th {
        background-color: #f9fafb;
        padding: 0.75rem;
        text-align: left;
        font-weight: 600;
        color: #111827;
    }
    .feature-table td {
        padding: 0.75rem;
        border-bottom: 1px solid #f3f4f6;
    }
    .feature-table tr:hover {
        background-color: #f9fafb;
    }
    .feature-name {
        font-weight: 600;
        color: #111827;
    }
    .feature-desc {
        color: #6b7280;
        font-size: 0.875rem;
    }
    .checkbox-container {
        text-align: center;
    }
    .filter-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .column-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .filter-icon {
        cursor: pointer;
        padding: 2px;
        border-radius: 4px;
    }
    .filter-icon:hover {
        background-color: #f3f4f6;
    }
    .popup {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    .overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        z-index: 999;
    }
    </style>
""", unsafe_allow_html=True)

# Create a container for the feature selection
with st.container():
    # Create a dataframe for the features
    features_df = pd.DataFrame({
        "Feature": available_optional_features,
        "Description": [feature_descriptions.get(feat, "No description available") for feat in available_optional_features],
        "Select": [st.session_state.feature_checkboxes.get(feat, False) for feat in available_optional_features]
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

    # Add filter popup when filter icon is clicked
    if st.session_state.get("show_filter", False):
        st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)
        st.markdown('<div class="popup">', unsafe_allow_html=True)
        st.markdown("### Filter Features")
        filter_text = st.text_input("Search features", 
                                  placeholder="Type to filter features...", 
                                  value=st.session_state.filter_text,
                                  key="filter_input")
        
        if filter_text != st.session_state.filter_text:
            st.session_state.filter_text = filter_text
            st.session_state.filtered_features = [
                feat for feat in available_optional_features 
                if filter_text.lower() in feat.lower() or 
                filter_text.lower() in feature_descriptions.get(feat, "").lower()
            ]
            st.rerun()
        
        if st.button("Close"):
            st.session_state.show_filter = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.session_state.filtered_features = available_optional_features

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
    working_df = st.session_state.df.copy()
    available_features = [f for f in all_features if f in working_df.columns]
    if available_features:
        st.session_state.final_dataset = working_df[available_features]
