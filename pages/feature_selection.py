import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

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
if "recommended_features" in st.session_state and st.session_state.recommended_features:
    available_optional_features.extend(st.session_state.recommended_features)
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
