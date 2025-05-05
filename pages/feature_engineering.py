import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import re
import uuid # Import uuid for generating unique IDs

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

st.markdown(
    """
    <style>
    /* Reduce font size in all dropdown options (affects multiselect feature names) */
    [data-baseweb="select"] div[role="option"] {
        font-size: 12px;
    }
    /* Also reduce font size for the text inside the control */
    [data-baseweb="select"] > div:nth-child(1) {
        font-size: 12px;
    }
    
    /* Increase the height of the multiselect box */
    [data-baseweb="select"] {
        min-height: 60px !important;
    }
    
    /* (Optional) Increase the height of containers that wrap your multi-feature transformation blocks */
    .feature-box {
        min-height: 80px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    st.rerun() # Replaced experimental_rerun

if st.session_state.show_merge:
    st.markdown("### 🔄 Merge Datasets")

    # Show merge blocks
    for i, block in enumerate(st.session_state.merge_blocks):
        st.markdown(f"**Merge Operation #{i+1}**")
        col1, col2, col3, col4, col5 = st.columns([0.5, 2, 2, 2, 2])

        with col1:
            # Use unique key for remove button
            if st.button("❌", key=f"remove_merge_{i}"):
                st.session_state.merge_blocks.pop(i)
                st.rerun() # Replaced experimental_rerun

        with col2:
            # Use unique key for selectbox
            left_table = st.selectbox(
                "Select Left Table",
                ["Bureau Data", "On-Us Data", "Installments Data"],
                key=f"merge_left_table_{i}",
                index=["Bureau Data", "On-Us Data", "Installments Data"].index(block.get("left_table", "Bureau Data").replace("Honors", "On-Us"))
            )

        with col3:
            # Use unique key for selectbox
            right_table = st.selectbox(
                "Select Right Table",
                ["Bureau Data", "On-Us Data", "Installments Data"],
                key=f"merge_right_table_{i}",
                index=["Bureau Data", "On-Us Data", "Installments Data"].index(block.get("right_table", "On-Us Data").replace("Honors", "On-Us"))
            )

        with col4:
            # Use unique key for selectbox
            join_type = st.selectbox(
                "Join Type",
                ["Left Join", "Right Join", "Inner Join", "Outer Join", "Cross Join", "Self Join"],
                key=f"merge_join_type_{i}",
                index=["Left Join", "Right Join", "Inner Join", "Outer Join", "Cross Join", "Self Join"].index(block.get("join_type", "Left Join"))
            )

        with col5:
            # Generate merged table name based on join operation
            left_name = left_table.lower().replace(" data", "").replace("-", "").replace(" ", "")
            right_name = right_table.lower().replace(" data", "").replace("-", "").replace(" ", "")
            join_type_short = join_type.lower().replace(" join", "")
            merged_table = f"{left_name}_{right_name}_{join_type_short}"

            # Use unique key for text_input
            merged_table = st.text_input(
                "Merged Table",
                value=block.get("merged_table", merged_table),
                key=f"merged_table_{i}"
            )

        # Update block in session state
        st.session_state.merge_blocks[i] = {
            "left_table": left_table,
            "right_table": right_table,
            "join_type": join_type,
            "merged_table": merged_table
        }

    # Add new merge operation button
    if st.button("➕ Add Merge Operation", key="add_merge"):
        st.session_state.merge_blocks.append({
            "left_table": "Bureau Data",
            "right_table": "On-Us Data",
            "join_type": "Left Join",
            "merged_table": "bureau_onus_merged"
        })
        st.rerun() # Replaced experimental_rerun

    # Apply all merge operations button
    if st.button("✅ Combine Datasets", key="combine_datasets"):
        try:
            st.success("Datasets combined successfully!")
            # Display the merged tables
            for block in st.session_state.merge_blocks:
                st.info(f"Merged Table: {block['merged_table']} ({block['left_table']} {block['join_type']} {block['right_table']})")
            st.session_state.show_merge = False
            st.rerun() # Replaced experimental_rerun
        except Exception as e:
            st.error(f"Error combining datasets: {str(e)}")

st.markdown("---")

# --- Centered Recommend Features Button ---
col1, col2, col3 = st.columns([0.5, 3, 0.5])  # Adjusted column widths to make the button wider
with col2:  # Middle column
    if st.button("✨ Recommend Features", key="recommend_features", use_container_width=True):
        try:
            # Ensure merged data exists
            if not st.session_state.merge_blocks:
                st.warning("No merged datasets available. Please merge datasets first.")
            else:
                # Use the last merged block for context
                merged_block = st.session_state.merge_blocks[-1]
                left_table = merged_block["left_table"]
                right_table = merged_block["right_table"]
                features = []
                if left_table == "Bureau Data":
                    features += BUREAU_FEATURES
                elif left_table == "On-Us Data":
                    features += ONUS_FEATURES
                elif left_table == "Installments Data":
                    features += INSTALLMENTS_FEATURES
                if right_table == "Bureau Data":
                    features += BUREAU_FEATURES
                elif right_table == "On-Us Data":
                    features += ONUS_FEATURES
                elif right_table == "Installments Data":
                    features += INSTALLMENTS_FEATURES
                features = list(dict.fromkeys(features))

                # Simulate a DataFrame with sample data for the recommended features
                data = {feature: [np.random.randint(1, 100) for _ in range(5)] for feature in features[:5]}
                df_recommended = pd.DataFrame(data)
                st.session_state.recommended_features = df_recommended  # Store the recommended features in session state
                st.dataframe(df_recommended, use_container_width=True)  # Display the DataFrame without styling
        except Exception as e:
            st.warning("Could not generate recommended features automatically. Showing default recommendations.")
            # Simulate a fallback DataFrame with default features and random data
            fallback_data = {
                "CREDIT_ACTIVE": [1, 0, 1, 1, 0],
                "DAYS_CREDIT": [100, 200, 150, 300, 250],
                "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved", "Approved", "Refused"],
                "DAYS_DECISION": [10, 20, 15, 30, 25],
                "AMT_INSTALMENT": [5000, 10000, 7500, 15000, 12500]
            }
            df_fallback = pd.DataFrame(fallback_data)
            st.session_state.recommended_features = df_fallback  # Store fallback data in session state
            st.dataframe(df_fallback, use_container_width=True)  # Display the fallback DataFrame without styling

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
            # Use unique key for remove button
            if st.button("❌", key=f"remove_single_{i}"):
                st.session_state.transform_blocks.pop(i)
                st.rerun() # Replaced experimental_rerun

        with col2:
            # Use unique key for selectbox
            feature = st.selectbox(
                "Select Feature",
                BUREAU_FEATURES + ONUS_FEATURES + INSTALLMENTS_FEATURES,
                key=f"single_feature_{i}",
                index=(BUREAU_FEATURES + ONUS_FEATURES + INSTALLMENTS_FEATURES).index(block.get("feature", BUREAU_FEATURES[0])) if block.get("feature") in (BUREAU_FEATURES + ONUS_FEATURES + INSTALLMENTS_FEATURES) else 0
            )

        with col3:
            # Use unique key for selectbox
            operation = st.selectbox(
                "Operation",
                ["Addition", "Subtraction", "Multiplication", "Division", "Log", "Square Root", "Power", "Absolute Value"],
                key=f"single_operation_{i}",
                index=["Addition", "Subtraction", "Multiplication", "Division", "Log", "Square Root", "Power", "Absolute Value"].index(block.get("operation", "Addition"))
            )

        with col4:
            # Use unique key for number_input
            if operation in ["Addition", "Subtraction", "Multiplication", "Division", "Power"]:
                value = st.number_input(
                    "Value",
                    value=block.get("value", 1.0),
                    key=f"single_value_{i}"
                )
            else:
                value = None

        with col5:
            # Generate suggested output name based on operation
            if value is not None:
                # Added str() and replace('.', '_') for value in output name
                suggested_output = f"{feature}_{operation.replace(' ', '_')}_{str(value).replace('.', '_')}"
            else:
                suggested_output = f"{feature}_{operation.replace(' ', '_')}"

            # Use unique key for text_input
            output_name = st.text_input(
                "Output Feature",
                value=block.get("output_name", suggested_output),
                key=f"single_output_{i}"
            )

        # Update block in session state
        st.session_state.transform_blocks[i] = {
            "feature": feature,
            "operation": operation,
            "value": value,
            "output_name": output_name
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