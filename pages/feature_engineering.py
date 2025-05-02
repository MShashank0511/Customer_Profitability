import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import re

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
            operation_details["parameters"] = {
                "numerator": features[0],
                "denominator": features[1]
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
    
    if op_type == "sum":
        return f"df['{operation_details['output_name']}'] = df[{features}].sum(axis=1)"
    elif op_type == "average":
        return f"df['{operation_details['output_name']}'] = df[{features}].mean(axis=1)"
    elif op_type == "product":
        return f"df['{operation_details['output_name']}'] = df[{features}].prod(axis=1)"
    elif op_type == "ratio":
        params = operation_details["parameters"]
        if "numerator" in params and "denominator" in params:
            return f"df['{operation_details['output_name']}'] = df['{params['numerator']}'] / df['{params['denominator']}']"
    elif op_type == "difference":
        if len(features) >= 2:
            return f"df['{operation_details['output_name']}'] = df['{features[0]}'] - df['{features[1]}']"
    elif op_type == "min":
        return f"df['{operation_details['output_name']}'] = df[{features}].min(axis=1)"
    elif op_type == "max":
        return f"df['{operation_details['output_name']}'] = df[{features}].max(axis=1)"
    elif op_type == "count":
        return f"df['{operation_details['output_name']}'] = df[{features}].count(axis=1)"
    elif op_type == "custom":
        # For custom operations, return a placeholder
        return f"# Custom operation: {operation_details['description']}\n# Features: {features}\n# Implement custom logic here"
    
    return ""

# --- Initialize session state ---
if "df" not in st.session_state:
    st.session_state.df = None

# Define feature lists
BUREAU_FEATURES = ["CREDIT_ACTIVE", "DAYS_CREDIT", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"]
HONORS_FEATURES = ["NAME_CONTRACT_STATUS", "DAYS_DECISION", "AMT_APPLICATION", "AMT_CREDIT"]
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

# Initialize multi-transform blocks
if "multi_transform_blocks" not in st.session_state:
    st.session_state.multi_transform_blocks = [{
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
        "right_table": "Honors Data",
        "join_type": "Left Join",
        "merged_table": "bureau_honors_merged"
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

# Initialize counter for multiselect keys
if "multiselect_counter" not in st.session_state:
    st.session_state.multiselect_counter = 0

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
            <p style='margin: 0;'>Honors Data</p>
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
    st.rerun()

if st.session_state.show_filter:
    st.markdown("### 🔍 Filter Data")
    
    # Show filter blocks
    for i, block in enumerate(st.session_state.filter_blocks):
        st.markdown(f"**Filter #{i+1}**")
        col1, col2, col3, col4, col5 = st.columns([0.5, 2, 2, 2, 2])
        
        with col1:
            if st.button("❌", key=f"remove_filter_{i}"):
                st.session_state.filter_blocks.pop(i)
                st.rerun()
        
        with col2:
            dataset = st.selectbox(
                "Select Data",
                ["Bureau Data", "Honors Data", "Installments Data"],
                key=f"dataset_{i}",
                index=["Bureau Data", "Honors Data", "Installments Data"].index(block.get("dataset", "Bureau Data"))
            )
            
            # Update features based on selected dataset
            if dataset == "Bureau Data":
                features = BUREAU_FEATURES
            elif dataset == "Honors Data":
                features = HONORS_FEATURES
            else:
                features = INSTALLMENTS_FEATURES
            
            feature = st.selectbox(
                "Select Feature",
                features,
                key=f"feature_{i}",
                index=features.index(block.get("feature", features[0])) if block.get("feature") in features else 0
            )
        
        with col3:
            operation = st.selectbox(
                "Select Operation",
                ["Greater Than", "Less Than", "Greater Than or Equal", "Less Than or Equal", "Equal To", "Not Equal To"],
                key=f"operation_{i}",
                index=["Greater Than", "Less Than", "Greater Than or Equal", "Less Than or Equal", "Equal To", "Not Equal To"].index(block.get("operation", "Greater Than"))
            )
        
        with col4:
            value = st.number_input(
                "Value",
                value=block.get("value", 0),
                key=f"value_{i}"
            )
        
        with col5:
            output_name = f"{feature}_{operation}_{value}"
            output_name = st.text_input(
                "Output Feature",
                value=block.get("output_name", output_name),
                key=f"output_{i}"
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
        st.rerun()
    
    # Apply all filters button
    if st.button("✅ Apply All Filters", key="apply_filters"):
        st.success("Filters applied successfully!")
        st.session_state.show_filter = False

# --- Merge Datasets Section ---
if st.button("🔄 Merge Datasets", key="merge_btn", use_container_width=True):
    st.session_state.show_merge = True
    st.rerun()

if st.session_state.show_merge:
    st.markdown("### 🔄 Merge Datasets")
    
    # Show merge blocks
    for i, block in enumerate(st.session_state.merge_blocks):
        st.markdown(f"**Merge Operation #{i+1}**")
        col1, col2, col3, col4, col5 = st.columns([0.5, 2, 2, 2, 2])
        
        with col1:
            if st.button("❌", key=f"remove_merge_{i}"):
                st.session_state.merge_blocks.pop(i)
                st.rerun()
        
        with col2:
            left_table = st.selectbox(
                "Select Left Table",
                ["Bureau Data", "Honors Data", "Installments Data"],
                key=f"left_table_{i}",
                index=["Bureau Data", "Honors Data", "Installments Data"].index(block.get("left_table", "Bureau Data"))
            )
        
        with col3:
            right_table = st.selectbox(
                "Select Right Table",
                ["Bureau Data", "Honors Data", "Installments Data"],
                key=f"right_table_{i}",
                index=["Bureau Data", "Honors Data", "Installments Data"].index(block.get("right_table", "Honors Data"))
            )
        
        with col4:
            join_type = st.selectbox(
                "Join Type",
                ["Left Join", "Right Join", "Inner Join", "Outer Join", "Cross Join", "Self Join"],
                key=f"join_type_{i}",
                index=["Left Join", "Right Join", "Inner Join", "Outer Join", "Cross Join", "Self Join"].index(block.get("join_type", "Left Join"))
            )
        
        with col5:
            # Generate merged table name based on join operation
            left_name = left_table.lower().replace(" data", "")
            right_name = right_table.lower().replace(" data", "")
            join_type_short = join_type.lower().replace(" join", "")
            merged_table = f"{left_name}_{right_name}_{join_type_short}"
            
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
            "right_table": "Honors Data",
            "join_type": "Left Join",
            "merged_table": "bureau_honors_merged"
        })
        st.rerun()
    
    # Apply all merge operations button
    if st.button("✅ Combine Datasets", key="combine_datasets"):
        try:
            st.success("Datasets combined successfully!")
            
            # Display the merged tables
            for block in st.session_state.merge_blocks:
                st.info(f"Merged Table: {block['merged_table']} ({block['left_table']} {block['join_type']} {block['right_table']})")
            
            st.session_state.show_merge = False
                
        except Exception as e:
            st.error(f"Error combining datasets: {str(e)}")

st.markdown("---")

# --- Data Transformation Buttons ---
st.subheader("Data Actions")
# Create a centered container for the buttons
col1, col2, col3 = st.columns([1, 2, 1])  # Unequal columns to center the buttons
with col2:  # Middle column
    if st.button("🔧 Data Transformation", key="transform_btn", use_container_width=True):
        st.session_state.show_popup1 = True

# --- Popup 1: Single Feature Transformations ---
if st.session_state.show_popup1:
    st.markdown("### 🔧 Single Feature Transformation")
    
    # Show transformation blocks
    for i, block in enumerate(st.session_state.transform_blocks):
        st.markdown(f"**Transformation #{i+1}**")
        col1, col2, col3, col4, col5 = st.columns([0.5, 2, 2, 2, 2])
        
        with col1:
            if st.button("❌", key=f"remove_{i}"):
                st.session_state.transform_blocks.pop(i)
                st.rerun()
        
        with col2:
            feature = st.selectbox(
                "Select Feature",
                BUREAU_FEATURES + HONORS_FEATURES + INSTALLMENTS_FEATURES,
                key=f"feature_{i}",
                index=(BUREAU_FEATURES + HONORS_FEATURES + INSTALLMENTS_FEATURES).index(block.get("feature", BUREAU_FEATURES[0])) if block.get("feature") in (BUREAU_FEATURES + HONORS_FEATURES + INSTALLMENTS_FEATURES) else 0
            )
        
        with col3:
            operation = st.selectbox(
                "Operation",
                ["Addition", "Subtraction", "Multiplication", "Division", "Log", "Square Root", "Power", "Absolute Value"],
                key=f"operation_{i}",
                index=["Addition", "Subtraction", "Multiplication", "Division", "Log", "Square Root", "Power", "Absolute Value"].index(block.get("operation", "Addition"))
            )
        
        with col4:
            if operation in ["Addition", "Subtraction", "Multiplication", "Division", "Power"]:
                value = st.number_input(
                    "Value",
                    value=block.get("value", 1.0),
                    key=f"value_{i}"
                )
            else:
                value = None
        
        with col5:
            # Generate suggested output name based on operation
            if value is not None:
                suggested_output = f"{feature}_{operation}_{value}"
            else:
                suggested_output = f"{feature}_{operation}"
            
            output_name = st.text_input(
                "Output Feature",
                value=block.get("output_name", suggested_output),
                key=f"output_{i}"
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
            "feature": BUREAU_FEATURES[0],
            "operation": "Addition",
            "value": 1.0,
            "output_name": ""
        })
        st.rerun()
    
    # Apply all transformations button
    if st.button("✅ Apply All Transformations", key="apply_transforms"):
        try:
            st.success("Transformations applied successfully!")
            st.session_state.show_popup1 = False
        except Exception as e:
            st.error(f"Error applying transformations: {str(e)}")

    st.markdown("---")
    
    # Multi-Feature Transformation Section
    st.markdown("### 🔧 Combine Multiple Features")
    
    # Show multi-feature transformation blocks
    for i, block in enumerate(st.session_state.multi_transform_blocks):
        st.markdown(f"**Transformation #{i+1}**")
        col1, col2, col3, col4 = st.columns([0.5, 2, 2, 2])
        
        with col1:
            if st.button("❌", key=f"remove_multi_{i}"):
                st.session_state.multi_transform_blocks.pop(i)
                st.rerun()
        
        with col2:
            # Feature selection
            current_features = block.get("features", [])
            
            selected_features = st.multiselect(
                "Choose Features to Combine",
                BUREAU_FEATURES + HONORS_FEATURES + INSTALLMENTS_FEATURES,
                default=current_features,
                key=f"multi_features_{i}",
                help="Select multiple features you want to combine"
            )
            
            # Update features in session state
            if selected_features != current_features:
                st.session_state.multi_transform_blocks[i]["features"] = selected_features
                st.session_state.multi_transform_blocks[i]["operation_details"] = {}
        
        with col3:
            # Operation prompt input
            operation_prompt = st.text_input(
                "Describe How to Combine Features",
                value=block.get("operation", ""),
                key=f"multi_operation_{i}",
                placeholder="e.g., Add these features together",
                help="Describe in simple words how you want to combine the selected features"
            )
            
            # Process operation prompt if both prompt and features exist
            if operation_prompt and selected_features:
                operation_details = process_operation_prompt(operation_prompt, selected_features)
                st.session_state.multi_transform_blocks[i]["operation_details"] = operation_details
        
        with col4:
            # Output feature name
            suggested_output = f"{operation_prompt.replace(' ', '_')}_{'_'.join(selected_features)}" if operation_prompt and selected_features else ""
            output_name = st.text_input(
                "Name for New Feature",
                value=block.get("output_name", suggested_output),
                key=f"multi_output_{i}",
                help="Give a name to the new feature that will be created"
            )
            
            # Show generated code preview
            if operation_prompt and selected_features and output_name:
                operation_details = st.session_state.multi_transform_blocks[i].get("operation_details", {})
                if operation_details:
                    operation_details["output_name"] = output_name
                    generated_code = generate_operation_code(operation_details)
                    st.code(generated_code, language="python")
        
        # Update the block in session state
        st.session_state.multi_transform_blocks[i] = {
            "features": selected_features,
            "operation": operation_prompt,
            "output_name": output_name,
            "operation_details": operation_details if operation_prompt and selected_features else {}
        }
    
    # Add new multi-feature transformation button
    if st.button("➕ Add New Feature Combination", key="add_multi_transform"):
        st.session_state.multi_transform_blocks.append({
            "features": [],
            "operation": "",
            "output_name": "",
            "operation_details": {}
        })
        st.rerun()
    
    # Apply all multi-feature transformations button
    if st.button("✅ Create All New Features", key="apply_multi_transforms"):
        try:
            st.success("New features created successfully!")
            st.session_state.show_popup1 = False
        except Exception as e:
            st.error(f"Error creating new features: {str(e)}")

