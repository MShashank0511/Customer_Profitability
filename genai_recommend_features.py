# genai_utils.py

import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import re

# Load environment variables and configure Gemini API
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_KEY_CONFIGURED = False

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_API_KEY_CONFIGURED = True
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        GEMINI_API_KEY_CONFIGURED = False
else:
    print("GEMINI_API_KEY not found in environment variables. Please set it in your .env file or environment.")

def summarize_dataset_columns(df: pd.DataFrame) -> str:
    """Generates a textual description of the dataset based on column names, data types, and examples."""
    summary_lines = []
    for col in df.columns:
        dtype = df[col].dtype
        example_values = df[col].dropna().unique()[:3]  # Show a few unique examples
        example_str = ", ".join(map(str, example_values))
        summary_lines.append(f"- {col} (type: {dtype}, examples: [{example_str}])")
    return "\n".join(summary_lines)

def get_recommended_features_gemini(dataset_description: str):
    """
    Calls the Gemini API to recommend engineered features for loan profitability prediction.
    """
    if not GEMINI_API_KEY_CONFIGURED:
        return "Error: Gemini API key not configured. Cannot get recommendations."

    llm_prompt = (
        "You are an expert data science assistant specializing in financial risk modeling.\n\n"
        "I have a loan dataset intended for profitability prediction. The profitability is calculated by "
        "estimating the probabilities of two key events:\n"
        "1. Charge-Off (COF_EVENT_LABEL) – when a loan is written off.\n"
        "2. Prepayment (PREPAYMENT_EVENT_LABEL) – when a loan is paid off early.\n"
        "The combined probabilities from these events are used to compute the expected profitability for each loan.\n\n"
        "Here is a summary of the available features in my raw dataset:\n"
        f"{dataset_description}\n\n"
        "Objective:\n"
        "My goal is to use machine learning models (Logistic Regression, LightGBM, XGBoost, Random Forest Classifiers) "
        "to predict these events. Based on the provided raw dataset features, I need you to perform **feature engineering** "
        "and recommend a ranked list of **new, engineered features** that would significantly enhance the predictive power "
        "of my models for Charge-Off, Prepayment, and overall loan profitability.\n\n"
        "Instructions:\n"
        "1. Identify Feature Engineering Opportunities from the features.\n"
        "2. Suggest only high-value, business-relevant engineered features.\n"
        "3. For each engineered feature, clearly state its name, how it's derived, the justification for its inclusion, "
        "and whether it's most relevant for Charge-Off, Prepayment, or Both.\n"
        "4. Provide a Ranked List in this exact format for each feature (ensure to strictly follow this multi-line format for each feature):\n"
        "- Engineered Feature Name: [Name]\n"
        "- Derivation: [Explanation of how to compute the feature]\n"
        "- Justification: [Why this feature is valuable]\n"
        "- Primary Event Impact: [Charge-Off / Prepayment / Both]\n\n"
        "Begin the list directly. Do not include any preamble before the first feature. Ensure each feature block is separated by a double newline if providing multiple features."
    )

    model = genai.GenerativeModel('gemini-1.5-flash') # Or your preferred Gemini model

    try:
        response = model.generate_content(
            llm_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        return response.text
    except Exception as e:
        # Log the error or handle it more gracefully
        print(f"Gemini API Error during generation: {str(e)}")
        return f"An error occurred while contacting Gemini: {str(e)}"

def parse_gemini_recommendations(text: str) -> pd.DataFrame:
    """
    Parses the text output from Gemini into a DataFrame.
    It expects features to be separated by double newlines or be identifiable by '- Engineered Feature Name:'.
    """
    features_list = []
    
    # Attempt to split by double newlines first, as it's a common separator for distinct blocks.
    # Then, refine by looking for the specific feature name marker.
    # Using a regex to split by one or more empty lines, then filtering out empty strings.
    raw_feature_blocks = [block for block in re.split(r'\n\s*\n', text.strip()) if block.strip()]

    if not raw_feature_blocks or not raw_feature_blocks[0].strip().startswith("- Engineered Feature Name:"):
        # If the above split doesn't work well, fallback to splitting by the feature name itself.
        # This helps if the LLM doesn't use double newlines consistently.
        # Positive lookahead `(?=...)` keeps the delimiter as part of the next split.
        raw_feature_blocks = [block for block in re.split(r'(?=- Engineered Feature Name:)', text.strip()) if block.strip()]


    for block_idx, block_content in enumerate(raw_feature_blocks):
        block = block_content.strip()
        if not block:
            continue

        feature_data = {}
        try:
            name_match = re.search(r"- Engineered Feature Name:\s*(.*)", block)
            derivation_match = re.search(r"- Derivation:\s*(.*)", block, re.DOTALL)
            justification_match = re.search(r"- Justification:\s*(.*)", block, re.DOTALL)
            impact_match = re.search(r"- Primary Event Impact:\s*(.*)", block)

            if name_match:
                feature_data['Feature'] = name_match.group(1).strip()
            else:
                # print(f"Skipping block (index {block_idx}), couldn't find Feature Name: '{block[:100]}...'")
                continue # Essential field missing

            def clean_multiline_field(text_block_content, field_text_match_group):
                if not field_text_match_group: return "N/A"
                # Split the field's text by lines
                lines = field_text_match_group.strip().split('\n')
                cleaned_lines = []
                for line in lines:
                    # Stop if we hit the start of another known field for the *current* feature block
                    if line.strip().startswith("- Derivation:") and "Derivation" not in feature_data or \
                       line.strip().startswith("- Justification:") and "Justification" not in feature_data or \
                       line.strip().startswith("- Primary Event Impact:") and "Primary Event Impact" not in feature_data :
                        if field_text_match_group.strip() != line.strip(): # make sure it's not the current line itself
                             break
                    cleaned_lines.append(line.strip())
                return " ".join(cleaned_lines).strip()

            derivation_text = derivation_match.group(1) if derivation_match else "N/A"
            justification_text = justification_match.group(1) if justification_match else "N/A"
            
            feature_data['Derivation'] = clean_multiline_field(block, derivation_text)
            feature_data['Justification'] = clean_multiline_field(block, justification_text)
            feature_data['Primary Event Impact'] = impact_match.group(1).strip() if impact_match else "N/A"
            
            feature_data['Description'] = f"Derivation: {feature_data.get('Derivation', 'N/A')}. Justification: {feature_data.get('Justification', 'N/A')}"
            
            feature_data['Min'] = 'N/A'
            feature_data['Max'] = 'N/A'
            feature_data['Mean'] = 'N/A'
            feature_data['Data Type'] = 'Engineered (Proposed)'
            
            features_list.append(feature_data)
        except Exception as e:
            print(f"Error parsing a feature block (index {block_idx}): {e}. Block content: '{block[:100]}...'")
            continue
            
    if not features_list:
        # This case will be handled in the Streamlit app, but good to be aware of
        pass

    return pd.DataFrame(features_list)