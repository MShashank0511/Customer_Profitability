import numpy as np

# Patch for deprecated np.bool
if not hasattr(np, 'bool'):
    np.bool = np.bool_
from xgboost import XGBClassifier
import plotly.express as px
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error
)
from sklearn.preprocessing import LabelEncoder
import shap
import json

st.set_page_config(page_title="Model Development", layout="wide")
st.title("🔧 Model Development")

# Load the CSV file
csv_file_path = "loan_data.csv"  # Replace with the actual path to your CSV file
df_full = pd.read_csv(csv_file_path)

# Ensure the target variables exist in the dataset
target_variables = ["Profitability_GBP", "COF_EVENT_LABEL", "PREPAYMENT_EVENT_LABEL"]
for target in target_variables:
    if target not in df_full.columns:
        raise ValueError(f"Target variable '{target}' not found in the dataset.")

# List of datasets
datasets = [
    {"name": "Dataset 1 (Profitability_GBP)", "target": "Profitability_GBP"},
    {"name": "Dataset 2 (COF_EVENT_LABEL)", "target": "COF_EVENT_LABEL"},
    {"name": "Dataset 3 (PREPAYMENT_EVENT_LABEL)", "target": "PREPAYMENT_EVENT_LABEL"},
]

# Session state for selected iterations
if "selected_iterations" not in st.session_state:
    st.session_state.selected_iterations = {}

# Step 1: Dropdown for Dataset Selection
selected_json = st.selectbox(
    "Select Dataset to Work On",
    [d["name"] for d in datasets]
)
current_dataset = next(d for d in datasets if d["name"] == selected_json)
target_column = current_dataset["target"]

# Work on a copy of the full dataframe
df = df_full.copy()

# Step 2: Sub-sampling
st.subheader("🔍 Sub-sampling")
sample_frac = st.slider("Select sub-sample fraction", 0.1, 1.0, 1.0, key=f"sample_frac_{target_column}")
df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

# Step 3: Target Variable Selection & Distribution
st.subheader("🎯 Target Variable Selection")
if df[target_column].nunique() < 10:
    target_type = "Categorical"
else:
    target_type = "Continuous"
st.markdown(f"**Target Variable:** {target_column}")
st.markdown(f"**Target Variable Type:** {target_type}")

if target_type == "Continuous":
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[target_column], kde=True, ax=ax, color="blue")
    ax.set_title(f"Distribution of {target_column}")
    ax.set_xlabel(target_column)
    st.pyplot(fig)
else:
    event_rate = (df[target_column].value_counts(normalize=True) * 100).to_dict()
    st.metric(
        label="Event Rate (%)",
        value=f"{event_rate.get(1, 0):.2f}%" if 1 in event_rate else "N/A",
        help="Percentage of positive events in the target variable."
    )
    frequency_df = df[target_column].value_counts().reset_index()
    frequency_df.columns = [target_column, "count"]
    fig = px.bar(
        frequency_df,
        x=target_column,
        y="count",
        text="count",
        labels={target_column: target_column, "count": "Frequency"},
        title=f"Frequency of {target_column}"
    )
    fig.update_traces(textposition="inside")
    st.plotly_chart(fig, use_container_width=True)

# Step 4: Train/Test Split
st.subheader("🧪 Train/Test Split")
test_size = st.slider("Select test size", 0.1, 0.5, 0.2, key=f"test_size_{target_column}")
feature_columns = st.multiselect("📊 Select Features", [col for col in df.columns if col != target_column], key=f"features_{target_column}")

if feature_columns:
    X = df[feature_columns]
    y = df[target_column]
    y = y.ravel()

    # Determine problem type
    if len(np.unique(y)) < 3:
        problem_type = "Classification"
    else:
        problem_type = "Regression"
    st.markdown(f"**Problem Type:** {problem_type}")

    # Encode target for classification
    if problem_type == "Classification" and y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Step 5: Model Selection
    st.subheader("📚 Select Model")
    if problem_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression,
            "LGBM Classifier": LGBMClassifier,
            "Random Forest Classifier": RandomForestClassifier,
            "XGBoost Classifier": XGBClassifier
        }
    else:
        models = {"Linear Regression": LinearRegression}

    selected_model = st.selectbox("Select Model", list(models.keys()), key=f"model_{target_column}")
    model_class = models[selected_model]

    # Step 6: Run Model
    if st.button("Run Model", key=f"run_{target_column}"):
        st.subheader("🏆 Best 5 Iterations")
        param_dist = {
            "Logistic Regression": {"C": [0.1, 1.0, 10.0], "solver": ["liblinear"]},
            "LGBM Classifier": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]},
            "Random Forest Classifier": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7]},
            "XGBoost Classifier": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]},
            "Linear Regression": {"fit_intercept": [True, False]}
        }
        sampled_params = list(ParameterSampler(param_dist[selected_model], n_iter=5, random_state=42))
        best_iterations = []

        for i, params in enumerate(sampled_params):
            try:
                model = model_class(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = {}
                if problem_type == "Classification":
                    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                    metrics = {
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred, average='binary', zero_division=0),
                        "Recall": recall_score(y_test, y_pred, average='binary', zero_division=0),
                        "F1 Score": f1_score(y_test, y_pred, average='binary', zero_division=0),
                    }
                    if y_proba is not None:
                        metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba[:, 1])
                        metrics["Gini Index"] = 2 * metrics["ROC-AUC"] - 1
                    metrics["Confusion Matrix"] = confusion_matrix(y_test, y_pred)
                else:
                    metrics = {
                        "Mean Squared Error": mean_squared_error(y_test, y_pred),
                        "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, y_pred)),
                    }
                best_iterations.append({"params": params, "metrics": metrics})
            except Exception as e:
                st.warning(f"⚠️ Skipping iteration due to error: {e}")

        st.session_state[f"best_iterations_{target_column}"] = best_iterations

        # Display Best Iterations
        for i, iteration in enumerate(best_iterations):
            with st.expander(f"Iteration {i+1}"):
                st.markdown("### 🔧 Hyperparameters")
                st.json(iteration["params"])

                st.markdown("### 📊 Metrics")

                if target_type == "Categorical":
                    # Metrics for Categorical Targets
                    col1, col2, col3, col4, col5, col6 = st.columns(6)

                    col1.metric(
                        "Accuracy",
                        f"{iteration['metrics'].get('Accuracy', 'N/A'):.4f}" if isinstance(iteration['metrics'].get('Accuracy'), (int, float)) else iteration['metrics'].get('Accuracy', 'N/A')
                    )
                    col2.metric(
                        "Precision",
                        f"{iteration['metrics'].get('Precision', 'N/A'):.4f}" if isinstance(iteration['metrics'].get('Precision'), (int, float)) else iteration['metrics'].get('Precision', 'N/A')
                    )
                    col3.metric(
                        "Recall",
                        f"{iteration['metrics'].get('Recall', 'N/A'):.4f}" if isinstance(iteration['metrics'].get('Recall'), (int, float)) else iteration['metrics'].get('Recall', 'N/A')
                    )
                    col4.metric(
                        "F1 Score",
                        f"{iteration['metrics'].get('F1 Score', 'N/A'):.4f}" if isinstance(iteration['metrics'].get('F1 Score'), (int, float)) else iteration['metrics'].get('F1 Score', 'N/A')
                    )
                    col5.metric(
                        "ROC-AUC",
                        f"{iteration['metrics'].get('ROC-AUC', 'N/A'):.4f}" if isinstance(iteration['metrics'].get('ROC-AUC'), (int, float)) else iteration['metrics'].get('ROC-AUC', 'N/A')
                    )
                    col6.metric(
                        "Gini Index",
                        f"{iteration['metrics'].get('Gini Index', 'N/A'):.4f}" if isinstance(iteration['metrics'].get('Gini Index'), (int, float)) else iteration['metrics'].get('Gini Index', 'N/A')
                    )

                    # Confusion Matrix
                    if "Confusion Matrix" in iteration["metrics"]:
                        st.markdown("#### Confusion Matrix")
                        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                        sns.heatmap(iteration["metrics"]["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                        ax_cm.set_xlabel("Predicted")
                        ax_cm.set_ylabel("Actual")
                        st.pyplot(fig_cm)

                    # SHAP Summary Plot
                    st.markdown("#### SHAP Summary Plot")
                    try:
                        explainer = shap.Explainer(model, X_train)
                        shap_values = explainer(X_test)
                        shap.summary_plot(shap_values, X_test, show=False)
                        st.pyplot(plt.gcf())
                    except Exception as e:
                        st.warning(f"⚠️ Could not generate SHAP summary plot: {e}")

                elif target_type == "Continuous":
                    # Metrics for Continuous Targets
                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Root Mean Squared Error (RMSE)",
                        f"{iteration['metrics'].get('Root Mean Squared Error', 'N/A'):.4f}" if isinstance(iteration['metrics'].get('Root Mean Squared Error'), (int, float)) else iteration['metrics'].get('Root Mean Squared Error', 'N/A')
                    )
                    col2.metric(
                        "Mean Squared Error (MSE)",
                        f"{iteration['metrics'].get('Mean Squared Error', 'N/A'):.4f}" if isinstance(iteration['metrics'].get('Mean Squared Error'), (int, float)) else iteration['metrics'].get('Mean Squared Error', 'N/A')
                    )
                    col3.metric(
                        "Mean Absolute Percentage Error (MAPE)",
                        f"{iteration['metrics'].get('MAPE', 'N/A'):.4f}" if isinstance(iteration['metrics'].get('MAPE'), (int, float)) else iteration['metrics'].get('MAPE', 'N/A')
                    )

                    # SHAP Summary Plot
                    st.markdown("#### SHAP Summary Plot")
                    try:
                        explainer = shap.Explainer(model, X_train)
                        shap_values = explainer(X_test)
                        shap.summary_plot(shap_values, X_test, show=False)
                        st.pyplot(plt.gcf())
                    except Exception as e:
                        st.warning(f"⚠️ Could not generate SHAP summary plot: {e}")

    # Step 7: Select Iteration and Save JSON
    best_iterations = st.session_state.get(f"best_iterations_{target_column}", [])
    if best_iterations:
        selected_iteration = st.selectbox(
            f"Select Iteration for {selected_json}",
            [f"Iteration {i+1}" for i in range(len(best_iterations))],
            key=f"iteration_select_{target_column}"
        )
        if st.button(f"Confirm Selection for {selected_json}", key=f"confirm_{target_column}"):
            iteration_index = int(selected_iteration.split(" ")[1]) - 1
            iteration_details = best_iterations[iteration_index]
            json_data = {
                "dataset": df[feature_columns].to_dict(orient="records"),
                "target_variable": target_column,
                "model_name": selected_model,
                "model_code": f"{model_class.__name__}(**{iteration_details['params']})",
                "hyperparameters": iteration_details["params"],
            }
            json_file_name = f"{selected_model}_{target_column}.json"
            with open(json_file_name, "w") as json_file:
                json.dump(json_data, json_file, indent=4)
            st.session_state.selected_iterations[selected_json] = json_file_name
            st.success(f"✅ {selected_iteration} selected for {selected_json}! JSON file created: {json_file_name}")

# Step 8: Proceed to Next Page
if len(st.session_state.selected_iterations) == len(datasets):
    if st.button("Proceed to Next Page"):
        st.success("✅ All datasets have been processed! You can proceed to the next page.")
else:
    st.info("Please complete the selection for all datasets before proceeding.")
