import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor # If you have it installed
from lightgbm import LGBMClassifier, LGBMRegressor
import glob
import streamlit as st # Import streamlit


def create_model(model_name, hyperparameters):
    """
    Creates an instance of the specified model with the given hyperparameters.

    Args:
        model_name (str): The name of the model.
        hyperparameters (dict): The hyperparameters for the model.

    Returns:
        An instance of the model, or None if the model name is invalid.
    """
    try:
        # Map model names to their classes.
        model_classes = {
            "Linear Regression": LinearRegression,
            "Logistic Regression": LogisticRegression,
            "XGBoost Classifier": XGBClassifier,  # Requires xgboost
            "XGBoost Regressor": XGBRegressor,    # Requires xgboost
            "LGBM Classifier": LGBMClassifier,
            "LGBM Regressor": LGBMRegressor
        }

        model_class = model_classes.get(model_name)  # Get the class, None if not found.
        if model_class:
            return model_class(**hyperparameters)  # Instantiate with hyperparameters.
        else:
            print(f"Warning: Model name '{model_name}' not recognized. Skipping.")
            return None
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")
        return None



def process_json_and_predict():
    """
    Processes JSON files based on a fixed naming convention, loads the data,
    trains models, generates predictions, and returns the results as a list of DataFrames.

    Returns:
        list of pd.DataFrame:  A list of DataFrames, or None on error.
                              Each DataFrame contains the original data plus a 'Predicted_Target' column.
    """
    dataframes = []
    # Define the JSON file naming convention
    file_names = [
        "LGBM Classifier_COF_EVENT_LABEL.json",
        "LGBM Classifier_PREPAYMENT_EVENT_LABEL.json",
        "LGBM Classifier_Profitability_GBP.json",
        "Linear Regression_Profitability_GBP.json",
        "Logistic Regression_COF_EVENT_LABEL.json",  # Addded Logistic Regression
        "Logistic Regression_PREPAYMENT_EVENT_LABEL.json", # Added Logistic Regression
        "XGBoost Classifier_COF_EVENT_LABEL.json",
        "XGBoost Classifier_PREPAYMENT_EVENT_LABEL.json",
        "XGBoost Classifier_Profitability_GBP.json",
        "XGBoost Regressor_COF_EVENT_LABEL.json",
        "XGBoost Regressor_PREPAYMENT_EVENT_LABEL.json",
        "XGBoost Regressor_Profitability_GBP.json"
    ]

    for json_file_name in file_names:
        json_file_path = json_file_name # Directly use the filename.  Assumes files are in same directory.

        if not os.path.exists(json_file_path):
            print(f"Warning: File not found: {json_file_path}. Skipping.")
            continue  # Skip to the next file if this one doesn't exist.

        try:
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_file_path}")
            return None

        # 1. Load Data from JSON
        rows = json_data.get("dataset")
        if not rows:
            print(f"Error: \'dataset\' key not found in JSON data from file {json_file_path}")
            return None

        # Convert rows data to a Pandas DataFrame
        df = pd.DataFrame(rows)
        df['source_file'] = os.path.basename(json_file_path)  # Store source filename

        target_variable = json_data.get("target_variable")
        if not target_variable:
            print(f"Error: 'target_variable' is missing in JSON data from file {json_file_path}")
            return None

        model_name = json_data.get("model_name")
        if not model_name:
            print(f"Error: 'model_name' is missing in JSON data from file {json_file_path}")
            return None

        hyperparameters = json_data.get("hyperparameters")
        if not hyperparameters:
            print(f"Error: 'hyperparameters' is missing in JSON data from file {json_file_path}")
            return None

        model_code = json_data.get("model_code")  # Added model_code.  Not used, but kept for compatibility.
        if not model_code:
            print(f"Error: 'model_code' is missing in JSON data from file {json_file_path}")
            return None

        # 2. Separate Features and Target Variable
        features = [col for col in df.columns if col != target_variable and df[col].dtype in [np.float64, np.int64]]
        X = df[features]
        y = df[target_variable] if target_variable in df.columns else None

        # 3. Create Model
        model = create_model(model_name, hyperparameters)
        if model is None:
            print(f"Skipping file {json_file_path} due to invalid model.")
            continue  # Process the next file

        # 4. Train-Test Split and Train Model (Only if target variable is provided)
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            print(f"Model {model_name} trained successfully for target variable {target_variable} from file {json_file_path}")

            # 5. Make Predictions
            y_pred = model.predict(X)
            df['Predicted_Target'] = y_pred  # Store predictions in the DataFrame

            # 6. Evaluate the model and print metrics
            print(f"Evaluation Metrics for {model_name} - {target_variable} - {json_file_path}:")
            if target_variable == "Profitability_GBP":
                r2 = r2_score(y_test, y_pred[X_test.index])
                print(f"  R-squared: {r2:.4f}")
            else:
                try:
                    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                    print(f"  AUC: {auc:.4f}")
                except Exception as e:
                    print(f"  AUC not calculated: {e}")
                acc = accuracy_score(y_test, y_pred[X_test.index])
                prec = precision_score(y_test, y_pred[X_test.index], zero_division=0)
                rec = recall_score(y_test, y_pred[X_test.index])
                f1 = f1_score(y_test, y_pred[X_test.index])
                print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        else:
            y_pred = model.predict(X)
            df['Predicted_Target'] = y_pred
            print(
                f"Trained model not evaluated because the target variable is not provided in the input data from file {json_file_path}")

        df.name = f"{target_variable}_{model_name}"  # Set DataFrame name
        dataframes.append(df)  # Append the processed DataFrame to the list

    return dataframes  # Return the list of DataFrames



if __name__ == "__main__":


    # Process the JSON data and generate predictions
    processed_dataframes = process_json_and_predict()

    if processed_dataframes:
        for df in processed_dataframes:
            print(f"\n--- DataFrame: {df.name} ---")
            print(df.head())
