# back.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import inspect

DEFAULT_DATA_DIR = "default_data"

# --- Helper to create model instances ---
def create_model_instance(model_name, hyperparameters):
    try:
        model_classes = {
            "Linear Regression": LinearRegression,
            "Logistic Regression": LogisticRegression,
            "XGBoost Classifier": XGBClassifier,
            "XGBoost Regressor": XGBRegressor,
            "LGBM Classifier": LGBMClassifier,
            "LGBM Regressor": LGBMRegressor,
            "Random Forest Classifier": RandomForestClassifier
        }
        model_class = model_classes.get(model_name)
        if model_class:
            valid_params = inspect.signature(model_class).parameters
            filtered_params = {k: v for k, v in hyperparameters.items() if k in valid_params}
            return model_class(**filtered_params)
        else:
            print(f"Warning: Model name '{model_name}' not recognized.")
            return None
    except Exception as e:
        print(f"Error creating model instance {model_name}: {e}")
        return None

# --- Function to process models ---
def process_models_from_session():
    confirmed_models_data = st.session_state.get("confirmed_model_outputs")
    source_of_config = "User Confirmed (Model Development)"  # Default assumption

    # New block: Try loading specific parquet files from data_registry/selected_iterations dynamically
    if not confirmed_models_data:
        st.info("No user-confirmed models found. Attempting to load predefined parquet files from data_registry/selected_iterations...")
        
        DATA_REGISTRY_BASE_DIR = "data_registry"
        SELECTED_ITERATIONS_SUBFOLDER = "selected_iterations"
        
        predefined_filenames = {
            "COF_EVENT_LABEL": "COF_EVENT_LABEL_test_data_task_Charge-Off_Model_for_COF_task_iter_1.parquet",
            "PREPAYMENT_EVENT_LABEL": "PREPAYMENT_EVENT_LABEL_test_data_task_Prepayment_Model_for_Prepayment_task_iter_1.parquet",
            "Profitability_GBP": "Profitability_GBP_test_data_task_Forecast_Model_for_Profitability_task_iter_1.parquet"
        }
        
        loaded_models = {}
        
        for key, filename in predefined_filenames.items():
            constructed_path = os.path.join(DATA_REGISTRY_BASE_DIR, SELECTED_ITERATIONS_SUBFOLDER, filename)
            if os.path.exists(constructed_path):
                try:
                    df = pd.read_parquet(constructed_path)
                    loaded_models[key] = {
                        "test_data_path": constructed_path,
                        "target_variable": None,       # Update these if you know the target variables
                        "selected_features": list(df.columns),
                        "model_name": None,            # Update if known
                        "hyperparameters": {}
                    }
                except Exception as e:
                    st.warning(f"Failed to load {constructed_path}: {e}")
            else:
                st.warning(f"File not found: {constructed_path}")
        
        if loaded_models:
            confirmed_models_data = loaded_models
            source_of_config = "Predefined Parquet Files"

    # If still no data, fallback to default config
    if not confirmed_models_data:
        st.info("Attempting to generate and use default configurations.")
        confirmed_models_data = generate_default_confirmed_outputs()
        source_of_config = "Default Configuration"
        if not confirmed_models_data:
            st.error("Failed to generate default model configurations. Cannot proceed.")
            return []

    processed_dataframes_list = []
    for task_key_proc, task_info_proc in confirmed_models_data.items():
        try:
            test_data_path_proc = task_info_proc.get("test_data_path")
            target_variable_proc = task_info_proc.get("target_variable")
            selected_features_proc = task_info_proc.get("selected_features")
            model_name_proc = task_info_proc.get("model_name")
            hyperparams_proc = task_info_proc.get("hyperparameters")

            # Basic validation of retrieved info
            if not all([test_data_path_proc, target_variable_proc, isinstance(selected_features_proc, list), model_name_proc, isinstance(hyperparams_proc, dict)]):
                st.error(f"Task '{task_key_proc}' has incomplete or malformed configuration. Skipping.")
                continue
            if not os.path.exists(test_data_path_proc):
                st.error(f"Skipping task '{task_key_proc}': Source data file not found at {test_data_path_proc}")
                continue

            df_source_for_task = pd.read_parquet(test_data_path_proc)

            # Validate that features and target exist in the loaded DataFrame
            missing_feats_proc = [f for f in selected_features_proc if f not in df_source_for_task.columns]
            if missing_feats_proc:
                st.error(f"Task '{task_key_proc}': Selected features {missing_feats_proc} not found in source data. Skipping.")
                continue
            if target_variable_proc not in df_source_for_task.columns:
                st.error(f"Task '{task_key_proc}': Target variable '{target_variable_proc}' not found in source data. Skipping.")
                continue

            # --- Preprocess Data ---
            df_source_for_task = preprocess_data(df_source_for_task)

            X_data = df_source_for_task[selected_features_proc]
            y_data = df_source_for_task[target_variable_proc]

            # Validate input data
            if X_data.empty or y_data.empty:
                st.error(f"Task '{task_key_proc}': Input features or target variable are empty after preprocessing. Skipping.")
                continue

            model_instance_proc = create_model_instance(model_name_proc, hyperparameters=hyperparams_proc)
            if model_instance_proc is None:
                st.warning(f"Skipping model processing for '{model_name_proc}' (task '{task_key_proc}') due to model instantiation failure.")
                continue

            # Split data for training and evaluation
            X_train_here, X_test_here, y_train_here, y_test_here = train_test_split(
                X_data, y_data, test_size=0.25, random_state=42
            )

            model_instance_proc.fit(X_train_here, y_train_here)

            # Predictions on the *entire* X_data
            if model_name_proc in ["Logistic Regression", "XGBoost Classifier", "LGBM Classifier", "Random Forest Classifier"]:
                try:
                    y_proba_full_task = model_instance_proc.predict_proba(X_data)[:, 1]
                    df_source_for_task['Predicted_Probability'] = y_proba_full_task
                except AttributeError:
                    st.warning(f"Model '{model_name_proc}' does not support probability prediction. Using predicted labels instead.")
                    y_pred_full_task = model_instance_proc.predict(X_data)
                    df_source_for_task['Predicted_Probability'] = y_pred_full_task
            elif model_name_proc in ["Linear Regression", "XGBoost Regressor", "LGBM Regressor"]:
                y_pred_full_task = model_instance_proc.predict(X_data)
                df_source_for_task['Predicted_Target'] = y_pred_full_task

            # Remove irrelevant columns based on task type
            if model_name_proc in ["Logistic Regression", "XGBoost Classifier", "LGBM Classifier", "Random Forest Classifier"]:
                if 'Predicted_Target' in df_source_for_task.columns:
                    df_source_for_task.drop(columns=['Predicted_Target'], inplace=True)
            elif model_name_proc in ["Linear Regression", "XGBoost Regressor", "LGBM Regressor"]:
                if 'Predicted_Probability' in df_source_for_task.columns:
                    df_source_for_task.drop(columns=['Predicted_Probability'], inplace=True)

            # Add attributes for model name and target variable
            df_source_for_task.attrs['name'] = f"{model_name_proc} ({target_variable_proc})"
            df_source_for_task.attrs['target_variable'] = target_variable_proc

            processed_dataframes_list.append(df_source_for_task)

        except Exception as e_task_proc:
            continue

    return processed_dataframes_list


# --- Main Streamlit Page Logic (Displays Confirmation/Key Results) ---
def main_backtesting_page():
    st.header("Step 3: Backtesting Setup & Confirmation")

    confirmed_models_to_display = st.session_state.get("confirmed_model_outputs")
    is_using_defaults_for_display = False

    if not confirmed_models_to_display:
        st.info("No models were confirmed in the 'Model Development' step. Displaying default model configurations for demonstration.")
        confirmed_models_to_display = generate_default_confirmed_outputs()
        is_using_defaults_for_display = True
        if not confirmed_models_to_display: # If default generation also failed
            st.error("Failed to generate default model configurations. Please ensure 'default_on_us_data.parquet' is available and contains necessary columns, or confirm models in the 'Model Development' page.")
            return 
    
    if is_using_defaults_for_display:
        st.markdown("> :information_source: **You are viewing default model configurations.** Metrics displayed under 'Key Metrics (from Development Step)' are placeholders. Actual evaluation for these defaults will occur if you proceed to Model Monitoring, using a new train/test split of the `default_on_us_data.parquet` file.")
    else:
        st.info("These are the models and iterations you selected in the 'Model Development' step. Their test data slices and configurations are listed below.")
    
    st.markdown("Clicking 'Prepare Data and Proceed...' will finalize these configurations for the Model Monitoring page. This involves re-training each model on a portion of its specified data and generating predictions on the entirety of that data.")


    if not confirmed_models_to_display:
        st.warning("No model configurations available to display (neither user-confirmed nor default).")
        return
    
    st.success(f"Ready to display {len(confirmed_models_to_display)} model task configurations {'(Defaults)' if is_using_defaults_for_display else '(User-Confirmed)'}.")

    for task_key_disp, task_info_disp in confirmed_models_to_display.items():
        st.markdown(f"---")
        st.markdown(f"#### Task: {task_info_disp.get('task_name', task_key_disp)}")
        try:
            target_disp = task_info_disp.get("target_variable")
            model_name_disp = task_info_disp.get("model_name")
            hyperparams_disp = task_info_disp.get("hyperparameters")
            dev_metrics_disp = task_info_disp.get("key_metrics", {}) 
            data_path_disp = task_info_disp.get("test_data_path") # This is the source data for this task
            features_disp = task_info_disp.get("selected_features", [])
            iter_num_disp = task_info_disp.get("iteration_number", "N/A (Default)")

            st.markdown(f"**Target Variable:** `{target_disp}` | **Model:** `{model_name_disp}` | **Source Iteration:** `{iter_num_disp}`")
            st.markdown(f"**Data Source for this task:** `{data_path_disp}`")
            
            with st.expander("View Selected Features & Hyperparameters"):
                st.markdown(f"**Selected Features ({len(features_disp)}):**")
                st.json(features_disp) 
                st.markdown(f"**Hyperparameters:**")
                st.json(hyperparams_disp)

            st.markdown(f"**Key Metrics (from Development Step Test Set {'- N/A for Defaults' if is_using_defaults_for_display and dev_metrics_disp.get('Info') else ''}):**")
            if dev_metrics_disp and not dev_metrics_disp.get("Info"): # Check if not placeholder
                 problem_type_for_metrics = "Regression" if target_disp == "Profitability_GBP" else "Classification"
                 num_cols_for_metrics = 3 if problem_type_for_metrics == "Regression" else 5
                 metric_cols_on_page = st.columns(num_cols_for_metrics)
                 
                 metrics_order_display = []
                 if problem_type_for_metrics == "Classification":
                      metrics_order_display = ["ROC-AUC", "F1 Score", "Accuracy", "Precision", "Recall"]
                 elif problem_type_for_metrics == "Regression":
                      metrics_order_display = ["Root Mean Squared Error (RMSE)", "Mean Squared Error (MSE)", "R-squared"]

                 metric_idx_disp = 0
                 for metric_name_on_page in metrics_order_display:
                      val_on_page = dev_metrics_disp.get(metric_name_on_page)
                      if val_on_page is not None:
                           val_str_on_page = f"{val_on_page:.4f}" if isinstance(val_on_page, (int, float, np.float64)) else str(val_on_page)
                           metric_cols_on_page[metric_idx_disp % num_cols_for_metrics].metric(metric_name_on_page, val_str_on_page)
                           metric_idx_disp +=1
                 if metric_idx_disp == 0:
                     st.caption("Metrics from development step either not applicable or not found in the standard list.")
                     st.json(dev_metrics_disp) # Show raw if no standard metrics found
            elif is_using_defaults_for_display:
                st.caption("Metrics from the 'Model Development' step are not applicable for default configurations. New metrics will be calculated upon proceeding.")
            else:
                 st.caption("No key metrics were explicitly passed from the Model Development step for this iteration, or they were placeholders.")

        except Exception as e_main_page_disp:
            st.error(f"An error occurred while displaying configuration details for task {task_key_disp}: {e_main_page_disp}")

    st.markdown("---")
    if st.button("Prepare Data and Proceed to Model Prediction Monitoring (Next Page)", key="proceed_to_model_monitoring_page"):
         st.info("Preparation for Model Monitoring initiated. When you navigate to the 'Model Monitoring' page, it will use the configurations processed by this step.")
         st.session_state["backtesting_setup_complete_flag"] = True # Flag for next page if needed
         st.success("Setup complete! You can now navigate to 'Model Monitoring' via the sidebar.")
         # If using st.navigation or st.switch_page:
         # st.switch_page("pages/model_monitoring.py") # Example navigation


# --- Generate Default Confirmed Outputs ---
def generate_default_confirmed_outputs():
    """
    Generates default configurations for models using the default data file.
    Returns:
        dict: A dictionary containing default configurations for each task.
    """
    default_data_path = os.path.join(DEFAULT_DATA_DIR, "default_on_us_data.parquet")
    
    if not os.path.exists(default_data_path):
        st.error(f"Default data file not found at {default_data_path}. Please ensure the file exists.")
        return None

    try:
        # Load the default data
        default_data = pd.read_parquet(default_data_path)

        # Validate required columns
        required_columns = ['Profitability_GBP', 'COF_EVENT_LABEL', 'PREPAYMENT_EVENT_LABEL']
        missing_columns = [col for col in required_columns if col not in default_data.columns]
        if missing_columns:
            st.error(f"Default data is missing required columns: {missing_columns}.")
            return None

        # Generate default configurations for each task
        default_configurations = {
            "Model 1 (Profitability_GBP)": {
                "test_data_path": default_data_path,
                "target_variable": "Profitability_GBP",
                "selected_features": [col for col in default_data.columns if col != "Profitability_GBP"],
                "model_name": "Linear Regression",
                "hyperparameters": {}
            },
            "Model 2 (COF_EVENT_LABEL)": {
                "test_data_path": default_data_path,
                "target_variable": "COF_EVENT_LABEL",
                "selected_features": [col for col in default_data.columns if col != "COF_EVENT_LABEL"],
                "model_name": "Logistic Regression",
                "hyperparameters": {}
            },
            "Model 3 (PREPAYMENT_EVENT_LABEL)": {
                "test_data_path": default_data_path,
                "target_variable": "PREPAYMENT_EVENT_LABEL",
                "selected_features": [col for col in default_data.columns if col != "PREPAYMENT_EVENT_LABEL"],
                "model_name": "Logistic Regression",
                "hyperparameters": {}
            }
        }

        return default_configurations

    except Exception as e:
        st.error(f"An error occurred while generating default configurations: {e}")
        return None

# --- Data Preprocessing Function ---
def preprocess_data(df):
    """
    Preprocesses the DataFrame by handling missing values, removing duplicates, and ensuring data readiness.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # --- 1. Handle Missing Values ---
    null_percentage = df.isnull().sum() / len(df) * 100
    high_null_columns = null_percentage[null_percentage > 30].index.tolist()
    if high_null_columns:
        df = df.drop(columns=high_null_columns)  # Drop columns with >30% null values

    # Impute remaining null values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                # Impute categorical columns with mode
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                # Impute numerical columns with mean
                df[col].fillna(df[col].mean(), inplace=True)

    # --- 2. Remove Duplicates ---
    df = df.drop_duplicates()

    # --- 3. Ensure Data Readiness ---
    # Convert categorical columns to numeric if necessary
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.factorize(df[col])[0]

    return df

# --- Entry point for the Streamlit page ---
if __name__ == "__main__":
    # Retrieve processed dataframes from session state
    processed_dataframes = st.session_state.get("processed_dataframes", [])

    if not processed_dataframes:
        st.error("No processed dataframes found. Please ensure the Model Development step is completed.")
    else:
        # Create a dictionary with model names and their corresponding DataFrames
        available_data_sources = {df.attrs.get('name', f"Unnamed Model {i}"): df for i, df in enumerate(processed_dataframes)}

        # Display the dropdown for Data Source Selection
        data_source_name = st.selectbox(
            "Select Data Source (Model)", options=list(available_data_sources.keys())
        )

        selected_df = available_data_sources[data_source_name]
        st.session_state.data_source = data_source_name

        # Display the selected DataFrame
        st.write(f"Selected Data Source: {data_source_name}")
        st.dataframe(selected_df.head())

    main_backtesting_page()
