import os
import pandas as pd
import google.generativeai as genai

# Set up the Gemini API key
try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
except KeyError:
    GEMINI_API_KEY = 'AIzaSyACQ67N4ATAjqUIO7xLGTxw0zQ4Xkzqbjk'  # Replace with your actual key

genai.configure(api_key=GEMINI_API_KEY)

def generate_feature_description_from_metadata(metadata_file_path: str) -> str:
    """
    Reads a CSV metadata file and returns formatted feature descriptions.

    Args:
        metadata_file_path (str): Path to the metadata CSV file.
                                  It must have 'feature_name' in column 0 and 'description' in column 2.

    Returns:
        str: A string with feature descriptions formatted for prompting.
    """
    df = pd.read_csv(metadata_file_path)

    if df.shape[1] < 3:
        raise ValueError("Metadata file must contain at least 3 columns with 'feature_name' in col 0 and 'description' in col 2.")

    lines = []
    for idx, row in df.iterrows():
        feature = str(row.iloc[0]).strip()
        description = str(row.iloc[2]).strip()
        if feature and description:
            lines.append(f"- **{feature}**: {description}")

    return "\n".join(lines)

def get_recommended_features_gemini(dataset_description: str) -> str:
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
        "Instructions for your analysis and recommendations:\n"
        "1.  **Identify Feature Engineering Opportunities**: Analyze the provided raw features and propose creative "
        "    and impactful new features that can be derived or combined from them. Think about ratios, interactions, "
        "    differences, aggregations, or transformations that could uncover deeper insights into loan behavior.\n"
        "2.  **Focus on Predictive Utility**: For each *newly engineered feature*, consider its strong business relevance "
        "    (how it intuitively affects charge-off or prepayment behavior) and its potential technical/predictive value "
        "    (e.g., likely predictive power, importance in tree-based models, utility in linear models after transformation). "
        "    Only suggest features that are likely to add substantial value.\n"
        "3.  **Distinguish Event Impact**: For each engineered feature, specify if it is likely more relevant for "
        "    predicting Charge-Offs, Prepayments, or both.\n"
        "4.  **Provide a Ranked List**: Present the output as a ranked list of **only the newly engineered features**. "
        "    The ranking should reflect the overall expected importance and impact of these engineered features for the combined objective.\n"
        "5.  **Detailed Justification**: For each engineered feature, provide a clear, descriptive name and a brief "
        "    justification. The justification should explain:\n"
        "    a.  How it's derived from the raw features.\n"
        "    b.  Its business impact on Charge-Offs and/or Prepayments.\n"
        "    c.  Its general predictive utility for loan profitability.\n\n"
        "Expected Output Format:\n"
        "A ranked list of engineered features. Each item should include:\n"
        "- Engineered Feature Name: (e.g., Debt-to-Income Ratio % Change)\n"
        "- Derivation: (Explain how it's created from existing features, e.g., 'annual_inc / loan_amount')\n"
        "- Justification: (Explanation covering business impact on COF/Prepayment, and predictive utility)\n"
        "- Primary Event Impact: (Charge-Off / Prepayment / Both)\n\n"
        "Now, please provide the ranked list of recommended **engineered features**, specifically leveraging the insights from your expert knowledge and the provided dataset description."
    )

    model = genai.GenerativeModel('gemini-2.0-flash')

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
        metadata_csv_path = input("Enter the path to your metadata CSV file: ").strip()
        if not metadata_csv_path or not os.path.isfile(metadata_csv_path):
            print("File not found. Please check the path.")
        else:
            try:
                dataset_description = generate_feature_description_from_metadata(metadata_csv_path)
                print("\nDataset Feature Descriptions:\n")
                print(dataset_description)

                print("\nGetting feature recommendations from Gemini...\n")
                recommendations = get_recommended_features_gemini(dataset_description)
                print("Recommended Engineered Features for Loan Profitability Prediction:\n")
                print(recommendations)
            except Exception as e:
                print(f"Error processing metadata file: {str(e)}")
