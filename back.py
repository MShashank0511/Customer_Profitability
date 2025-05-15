# back.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
# Import necessary model classes (needed for create_model_instance helper)
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier # Assuming RF is also used
# Import necessary metrics (needed for displaying saved metrics)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_percentage_error, r2_score
)
# Import train_test_split for splitting data
from sklearn.model_selection import train_test_split
import inspect # Needed in create_model_instance

# --- Streamlit Page Configuration ---
# This MUST be the first Streamlit command in back.py


st.title("➡️ Model Confirmation & Data Loader")
st.markdown("""
    <style>
    .stSuccess > div {
        border-left: 0.5rem solid #4CAF50 !important;
    }
    .stWarning > div {
        border-left: 0.5rem solid #ff9800 !important;
    }
     .stError > div {
        border-left: 0.5rem solid #f44336 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Helper to create model instances (Needed for potential future use or inspection) ---
# Copied from previous iteration, though not strictly used for evaluation on THIS page
def create_model_instance(model_name, hyperparameters):
    """Creates an untrained model instance based on name and hyperparameters."""
    try:
        model_classes = {
            "Linear Regression": LinearRegression,
            "Logistic Regression": LogisticRegression,
            "XGBoost Classifier": XGBClassifier, # Use names from Model_develop.py
            "XGBoost Regressor": XGBRegressor,   # Add Regressor if needed
            "LGBM Classifier": LGBMClassifier,   # Use names from Model_develop.py
            "LGBM Regressor": LGBMRegressor,     # Add Regressor if needed
            "Random Forest Classifier": RandomForestClassifier # Added RF
        }
        model_class = model_classes.get(model_name)
        if model_class:
            # Filter hyperparameters to match the model's constructor
            import inspect
            valid_params = inspect.signature(model_class).parameters
            filtered_params = {k: v for k, v in hyperparameters.items() if k in valid_params}

            # Handle specific model requirements (like use_label_encoder in XGBoost)
            if model_name.startswith("XGB") and 'use_label_encoder' in valid_params:
                 # Set consistently with how it was likely trained in Model_develop
                 filtered_params['use_label_encoder'] = False
                 # Add eval_metric if not present and it's a classifier
                 if model_name == "XGBoost Classifier" and 'eval_metric' not in filtered_params:
                      filtered_params['eval_metric'] = 'logloss'
                 elif model_name == "XGBoost Regressor" and 'eval_metric' not in filtered_params:
                      filtered_params['eval_metric'] = 'rmse'


            model = model_class(**filtered_params)
            # Note: This model is NOT trained on this page.
            return model
        else:
            # st.warning(f"Model name '{model_name}' not recognized for instantiation.") # Avoid warning in helper
            return None
    except Exception as e:
        # st.error(f"Error creating model instance {model_name}: {e}") # Avoid error in helper
        return None

# --- Function to load and prepare data for the NEXT page (model_monitoring.py) ---
# This is the function that model_monitoring.py will import
def process_models_from_session():
    """
    Loads confirmed model test data and metadata from session state,
    trains models, generates predictions, calculates evaluation metrics,
    and prepares DataFrames with metadata attributes for the next page.

    Returns:
        list: A list of pandas DataFrames. Each DataFrame corresponds to a
              confirmed model task and contains the test data with metadata
              attached as DataFrame attributes (name, target_variable,
              selected_features, model_name, hyperparameters, Predicted_Target).
              Returns an empty list if no confirmed models are found or
              an error occurs during loading data for any task.
    """
    # Access the confirmed models from the Model Development page's session state
    confirmed_models = st.session_state.get("confirmed_model_outputs", {})

    # If no models have been confirmed, return an empty list
    if not confirmed_models:
        return []

    processed_dataframes = []

    # Iterate through each confirmed task stored in session state
    for task_key, task_info in confirmed_models.items():
        try:
            # Retrieve the necessary information for this task
            test_data_path = task_info.get("test_data_path")
            target_variable = task_info.get("target_variable")
            selected_features = task_info.get("selected_features")
            model_name = task_info.get("model_name")
            hyperparameters = task_info.get("hyperparameters")

            # Check if the test data path is valid and the file exists
            if not test_data_path or not os.path.exists(test_data_path):
                st.error(f"Skipping task '{task_key}' for data loading: Test data file not found at {test_data_path}")
                continue  # Move to the next task if data file is missing

            # Load the specific test data slice saved from the previous step (Model_develop.py)
            df_test = pd.read_parquet(test_data_path)

            # --- Attach Metadata to DataFrame Attributes ---
            # This makes the DataFrame compatible with model_monitoring.py's expectation
            df_test_with_attrs = df_test.copy()
            df_test_with_attrs.attrs["name"] = task_info.get("task_name", task_key)  # Use task_name for display, fallback to key
            df_test_with_attrs.attrs["target_variable"] = target_variable
            df_test_with_attrs.attrs["selected_features"] = selected_features  # Store the list of features
            df_test_with_attrs.attrs["model_name"] = model_name
            df_test_with_attrs.attrs["hyperparameters"] = hyperparameters

            # --- Train the Model and Generate Predictions ---
            features = selected_features
            X = df_test[features]
            y = df_test[target_variable] if target_variable in df_test.columns else None

            # Create the model instance
            model = create_model_instance(model_name, hyperparameters)
            if model is None:
                st.warning(f"Skipping model '{model_name}' for task '{task_key}' due to invalid configuration.")
                continue

            if y is not None:
                # Train/Test Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)

                # Predict on the entire dataset
                y_pred = model.predict(X)
                df_test_with_attrs['Predicted_Target'] = y_pred

                # --- Calculate Evaluation Metrics ---
                if target_variable == "Profitability_GBP":  # Regression
                    r2 = r2_score(y_test, model.predict(X_test))
                    df_test_with_attrs.attrs["evaluation_metrics"] = {"R-squared": r2}
                else:  # Classification
                    try:
                        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                    except Exception:
                        auc = None
                    acc = accuracy_score(y_test, model.predict(X_test))
                    prec = precision_score(y_test, model.predict(X_test), zero_division=0)
                    rec = recall_score(y_test, model.predict(X_test))
                    f1 = f1_score(y_test, model.predict(X_test))
                    df_test_with_attrs.attrs["evaluation_metrics"] = {
                        "AUC": auc,
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1 Score": f1,
                    }
            else:
                # Predict without target variable
                y_pred = model.predict(X)
                df_test_with_attrs['Predicted_Target'] = y_pred
                st.warning(f"Target variable '{target_variable}' not found in task '{task_key}'. Predictions generated without evaluation.")

            # Add the processed DataFrame to the results list
            processed_dataframes.append(df_test_with_attrs)

        except Exception as e:
            # Log any other errors that occur while processing this task
            st.error(f"An error occurred while loading/preparing data for task '{task_key}': {e}")
            # st.exception(e)  # Uncomment to show the full traceback in the Streamlit app

    # Return the list of successfully processed DataFrames
    return processed_dataframes

# --- Main Streamlit Page Logic (Displays Confirmation/Key Results) ---
# This code runs when back.py is the active page
def main_backtesting_page():
    """
    Main function for the Streamlit 'Backtesting & Evaluation' page.
    Displays confirmed models and their key metrics from session state.
    """
    st.subheader("Confirmed Models from Development")
    st.info("Below are the models and iterations you selected and confirmed in the 'Model Development' step.")
    st.info("Click 'Proceed to Backtesting (Next Page)' to view detailed monitoring analysis using the test data from these selections.")


    # Access the confirmed models from the previous page's session state
    confirmed_models = st.session_state.get("confirmed_model_outputs", {})

    if not confirmed_models:
        st.warning("No models have been confirmed yet.")
        st.info("Please return to the 'Model Development' page and confirm a selected iteration for each model task.")
    else:
        st.success(f"Found {len(confirmed_models)} confirmed model tasks.")

        # Iterate through each confirmed task and display its key information
        for task_key, task_info in confirmed_models.items():
            st.markdown(f"---")
            st.markdown(f"### {task_info.get('task_name', task_key)}")

            try:
                target_variable = task_info.get("target_variable")
                model_name = task_info.get("model_name")
                hyperparameters = task_info.get("hyperparameters")
                key_metrics = task_info.get("key_metrics", {}) # Get saved metrics from Model_develop.py
                test_data_path = task_info.get("test_data_path")

                st.markdown(f"**Target Variable:** {target_variable}")
                st.markdown(f"**Model:** {model_name}")
                st.markdown(f"**Test Data File:** `{test_data_path}`") # Display the path for confirmation

                # Display saved hyperparameters
                st.markdown(f"**Hyperparameters:**")
                st.json(hyperparameters)

                # Display the saved key metrics from the previous page
                st.markdown(f"**Key Metrics (from Development Step Test Set):**")
                if key_metrics:
                     # Determine Problem Type to display relevant metrics
                     classification_targets = ["COF_EVENT_LABEL", "PREPAYMENT_EVENT_LABEL"]
                     problem_type = "Classification" if target_variable in classification_targets else "Regression"

                     if problem_type == "Classification":
                          metrics_cols = st.columns(6)
                          metrics_order = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Gini Index"]
                          cols = [metrics_cols[0], metrics_cols[1], metrics_cols[2], metrics_cols[3], metrics_cols[4], metrics_cols[5]] # Use columns
                          for i, metric_name in enumerate(metrics_order):
                               value = key_metrics.get(metric_name, 'N/A')
                               display_value = f"{value:.4f}" if isinstance(value, (int, float, np.float64)) else value
                               cols[i].metric(metric_name, display_value)

                     elif problem_type == "Regression":
                          metrics_cols = st.columns(3)
                          metrics_order = ["Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "Mean Absolute Percentage Error (MAPE)"]
                          cols = [metrics_cols[0], metrics_cols[1], metrics_cols[2]] # Use columns
                          for i, metric_name in enumerate(metrics_order):
                                value = key_metrics.get(metric_name, 'N/A')
                                display_value = f"{value:.4f}" if isinstance(value, (int, float, np.float64)) else value
                                cols[i].metric(metric_name, display_value)
                else:
                     st.info("No key metrics were saved for this confirmed iteration.")

            except Exception as e:
                st.error(f"An error occurred while displaying confirmation for task {task_key}: {e}")
                # st.exception(e) # Uncomment for traceback


        st.markdown("---")
        st.success("All confirmed models are ready.")
        # Add a button to proceed to the next page (Model Monitoring)
        # In a multi-page app, this button would trigger navigation
        if st.button("Proceed to Model Prediction Monitoring (Next Page)"):
             st.info("Navigating to Model Prediction Monitoring...")
             # Add logic here to switch page, e.g.:
             # st.switch_page("pages/model_monitoring.py") # Assuming model_monitoring is in a 'pages' directory
             # For demonstration, just show a success message
             st.success("Ready to proceed! The next page will load data using `process_models_from_session`.")


# --- Entry point for the Streamlit page ---
if __name__ == "__main__":
    # When back.py is run directly, execute the main page logic
    main_backtesting_page()

