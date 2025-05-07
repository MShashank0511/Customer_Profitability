import numpy as np

# Patch for deprecated np.bool
if not hasattr(np, 'bool'):
    np.bool = np.bool_

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

st.set_page_config(page_title="Model Development", layout="wide")
st.title("🔧 Model Development")

# Load the CSV file
csv_file_path = "loan_data.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file_path)

# Ensure the target variables exist in the dataset
target_variables = ["Profitability_GBP", "COF_EVENT_LABEL", "PREPAYMENT_EVENT_LABEL"]
for target in target_variables:
    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in the dataset.")

# Create JSON files for each target variable
df_profitability = df.copy()
df_profitability.to_json("Model1_Profitability_GBP.json", orient="records")

df_cof_event = df.copy()
df_cof_event.to_json("Model1_COF_EVENT_LABEL.json", orient="records")

df_prepayment_event = df.copy()
df_prepayment_event.to_json("Model1_PREPAYMENT_EVENT_LABEL.json", orient="records")

print("JSON files created successfully!")

# Step 1: Dropdown for JSON Selection
json_options = ["Dataset 1 (Profitability_GBP)", "Dataset 2 (COF_EVENT_LABEL)", "Dataset 3 (PREPAYMENT_EVENT_LABEL)"]
selected_json = st.selectbox("Select Dataset", json_options)

# Step 2: Load Dataset Based on Selection
try:
    if selected_json == "Dataset 1 (Profitability_GBP)":
        df = pd.read_json("Model1_Profitability_GBP.json")
        st.success("✅ Dataset 1 (Profitability_GBP) Loaded Successfully!")
    elif selected_json == "Dataset 2 (COF_EVENT_LABEL)":
        df = pd.read_json("Model1_COF_EVENT_LABEL.json")
        st.success("✅ Dataset 2 (COF_EVENT_LABEL) Loaded Successfully!")
    elif selected_json == "Dataset 3 (PREPAYMENT_EVENT_LABEL)":
        df = pd.read_json("Model1_PREPAYMENT_EVENT_LABEL.json")
        st.success("✅ Dataset 3 (PREPAYMENT_EVENT_LABEL) Loaded Successfully!")
    else:
        st.error("❌ Invalid dataset selection.")
        st.stop()

    # Display the first few rows of the dataset
    st.dataframe(df.head(), use_container_width=True)

except FileNotFoundError as e:
    st.error(f"❌ Required JSON file not found: {e}")
    st.stop()

# Step 3: Sub-sampling
st.subheader("🔍 Sub-sampling")
sample_frac = st.slider("Select sub-sample fraction", 0.1, 1.0, 1.0)
df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

# Step 4: Target Variable Selection
st.subheader("🎯 Target Variable Selection")

# Extract the target variable from the JSON file name
if selected_json == "Dataset 1 (Profitability_GBP)":
    target_column = "Profitability_GBP"
elif selected_json == "Dataset 2 (COF_EVENT_LABEL)":
    target_column = "COF_EVENT_LABEL"
elif selected_json == "Dataset 3 (PREPAYMENT_EVENT_LABEL)":
    target_column = "PREPAYMENT_EVENT_LABEL"
else:
    st.error("❌ Invalid dataset selection.")
    st.stop()

# Determine if the target variable is categorical or continuous
if df[target_column].nunique() < 10:  # Arbitrary threshold for categorical
    target_type = "Categorical"
else:
    target_type = "Continuous"

# Display the target variable and its type
st.markdown(f"**Target Variable:** {target_column}")
st.markdown(f"**Target Variable Type:** {target_type}")

# Visualize the target variable
if target_type == "Continuous":
    # Display histogram with KDE
    st.subheader(f"Distribution of {target_column}")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[target_column], kde=True, ax=ax, color="blue")
    ax.set_title(f"Distribution of {target_column}")
    ax.set_xlabel(target_column)
    st.pyplot(fig)
else:
    # Display event rate as a card
    event_rate = (df[target_column].value_counts(normalize=True) * 100).to_dict()
    st.metric(
        label="Event Rate (%)",
        value=f"{event_rate.get(1, 0):.2f}%" if 1 in event_rate else "N/A",
        help="Percentage of positive events in the target variable."
    )

    # Display bar plot for frequency
    frequency_df = df[target_column].value_counts().reset_index()
    frequency_df.columns = [target_column, "count"]

    fig = px.bar(
        frequency_df,
        x=target_column,
        y="count",
        text="count",  # Add count as text inside the bars
        labels={target_column: target_column, "count": "Frequency"},
        title=f"Frequency of {target_column}"
    )

    # Update layout to display text inside the bars
    fig.update_traces(textposition="inside")  # Position text inside the bars
    st.plotly_chart(fig, use_container_width=True)

# Step 5: Train/Test Split
st.subheader("🧪 Train/Test Split")
test_size = st.slider("Select test size", 0.1, 0.5, 0.2)

# Step 6: Model Training
st.subheader("⚙️ Model Training")
feature_columns = st.multiselect("📊 Select Features", [col for col in df.columns if col != target_column])

if feature_columns:
    X = df[feature_columns]
    y = df[target_column]

    # Ensure y is a 1D array
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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Step 7: Model Selection
    st.subheader("📚 Select Model")
    if problem_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression,
            "LGBM Classifier": LGBMClassifier,
            "Random Forest Classifier": RandomForestClassifier
        }
    else:
        models = {"Linear Regression": LinearRegression}

    selected_model = st.selectbox("Select Model", list(models.keys()))
    model_class = models[selected_model]

    # Step 8: Run Model
    if st.button("Run Model"):
        st.subheader("🏆 Best 5 Iterations")
        param_dist = {
            "Logistic Regression": {"C": [0.1, 1.0, 10.0], "solver": ["liblinear"]},
            "LGBM Classifier": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]},
            "Random Forest Classifier": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7]},
            "Linear Regression": {"fit_intercept": [True, False]}
        }

        sampled_params = list(ParameterSampler(param_dist[selected_model], n_iter=5, random_state=42))
        best_iterations = []

        for i, params in enumerate(sampled_params):
            try:
                # Initialize the model with sampled hyperparameters
                model = model_class(**params)

                # Fit the model
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Metrics
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

                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    metrics["Confusion Matrix"] = cm
                else:
                    metrics = {
                        "Mean Squared Error": mean_squared_error(y_test, y_pred),
                        "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, y_pred)),
                    }

                # Append results
                best_iterations.append({"params": params, "metrics": metrics})

            except Exception as e:
                st.warning(f"⚠️ Skipping iteration due to error: {e}")

        # Display Best Iterations
        for i, iteration in enumerate(best_iterations):
            with st.expander(f"Iteration {i+1}"):
                st.markdown("### 🔧 Hyperparameters")
                st.json(iteration["params"])

                st.markdown("### 📊 Metrics")

                if target_type == "Categorical":
                    # Metrics for Categorical Targets
                    col1, col2, col3, col4 = st.columns(4)

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

                        # Create a new figure for the SHAP summary plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, X_test, show=False, plot_type="dot", ax=ax)
                        st.pyplot(fig)
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

                        # Create a new figure for the SHAP summary plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, X_test, show=False, plot_type="dot", ax=ax)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"⚠️ Could not generate SHAP summary plot: {e}")

        # Step 8: Select Final Iteration
        st.subheader("✅ Select Final Iteration")
        selected_iteration = st.selectbox("Select Iteration", [f"Iteration {i+1}" for i in range(len(best_iterations))])
        if st.button("Pass Selected Iteration to Next Page"):
            st.success(f"Selected {selected_iteration} for the next page!")
