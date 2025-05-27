import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
# Set up the Gemini API key
load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]


genai.configure(api_key=GEMINI_API_KEY)

# def generate_feature_description_from_metadata(metadata_file_path: str) -> str:
#     """
#     Reads a CSV metadata file and returns formatted feature descriptions.

#     Args:
#         metadata_file_path (str): Path to the metadata CSV file.
#                                   It must have 'feature_name' in column 0 and 'description' in column 2.

#     Returns:
#         str: A string with feature descriptions formatted for prompting.
#     """
#     df = pd.read_csv(metadata_file_path)

#     if df.shape[1] < 3:
#         raise ValueError("Metadata file must contain at least 3 columns with 'feature_name' in col 0 and 'description' in col 2.")

#     lines = []
#     for idx, row in df.iterrows():
#         feature = str(row.iloc[0]).strip()
#         description = str(row.iloc[2]).strip()
#         if feature and description:
#             lines.append(f"- **{feature}**: {description}")

#     return "\n".join(lines)
def summarize_dataset_columns(df: pd.DataFrame) -> str:
    """Generates a textual description of the dataset based on column names and data types."""
    summary_lines = []
    for col in df.columns:
        dtype = df[col].dtype
        summary_lines.append(f"- {col} (type: {dtype})")
    return "\n".join(summary_lines)

def get_recommended_features_gemini(dataset_description: str):
    """
    Calls the Gemini API to recommend engineered features for loan profitability prediction.
    """
    llm_prompt = (
        "You are an expert data science assistant specializing in financial risk modeling.\n\n"
        "I have a loan dataset intended for profitability prediction. The profitability is calculated by "
        "estimating the probabilities of two key events:\n"
        "1. Charge-Off (COF_EVENT_LABEL) – when a loan is written off.\n"
        "2. Prepayment (PREPAYMENT_EVENT_LABEL) – when a loan is paid off early.\n"
        "The combined probabilities from these events are used to compute the expected profitability for each loan.\n\n"
        "Here is a description of the available features in my raw dataset:\n"
        f"{dataset_description}\n\n"
        "Objective:\n"
        "My goal is to use machine learning models (Logistic Regression, LightGBM, XGBoost, Random Forest Classifiers) "
        "to predict these events. Based on the provided raw dataset features, I need you to perform **feature engineering** "
        "and recommend a ranked list of **new, engineered features** that would significantly enhance the predictive power "
        "of my models for Charge-Off, Prepayment, and overall loan profitability.\n\n"
        "Instructions:\n"
        "1. Identify Feature Engineering Opportunities from the features.\n"
        "2. Suggest only high-value, business-relevant engineered features.\n"
        "3. Indicate if they are most relevant for Charge-Off, Prepayment, or Both.\n"
        "4. Provide a Ranked List in this format:\n"
        "- Engineered Feature Name\n"
        "- Derivation\n"
        "- Justification\n"
        "- Primary Event Impact\n\n"
        "Now, please provide the ranked list of engineered features based on the dataset description."
    )

    model = genai.GenerativeModel('gemini-1.5-flash')

    try:
        response = model.generate_content(
            llm_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    if GEMINI_API_KEY == "YOUR_API_KEY":
        print("Please replace 'YOUR_API_KEY' with your actual Google AI Studio API key.")
    else:
        csv_path = input("Enter the path to your dataset CSV file: ").strip()
        if not csv_path or not os.path.isfile(csv_path):
            print("File not found. Please check the path.")
        else:
            try:
                df = pd.read_csv(csv_path, nrows=5)  # Only need a few rows to infer structure
                dataset_description = summarize_dataset_columns(df)
                print("\nDataset Feature Summary:\n")
                print(dataset_description)

                print("\nGetting feature recommendations from Gemini...\n")
                recommendations = get_recommended_features_gemini(dataset_description)
                print("Recommended Features for Loan Profitability Prediction:\n")
                print(recommendations)
            except Exception as e:
                print(f"Error reading CSV: {str(e)}")

