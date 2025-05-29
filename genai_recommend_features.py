# genai_utils.py
import ast 
import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import re
import numpy as np
import streamlit as st
from typing import List, Dict, Any
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
        "1. Always use the provided raw dataset features as the basis for your recommendations. Do not assume any external features.\n"
        "2. Identify Feature Engineering Opportunities from the features.\n"
        "3. Suggest only high-value, business-relevant engineered features.\n"
        "4. For each engineered feature, clearly state its name, how it's derived, the justification for its inclusion, "
        "and whether it's most relevant for Charge-Off, Prepayment, or Both.\n"
        "5. Provide a Ranked List in this exact format for each feature (ensure to strictly follow this multi-line format for each feature):\n"
        "- Engineered Feature Name: [Name]\n"
        "- Description: [Brief description of the feature]\n"
        "- List of Raw Features Used: [List of raw features used to create this engineered feature]\n"
        "- Derivation: [Explanation of how to compute the feature]\n"
        "- Code Snippet: [Python code snippet to compute the feature]\n"
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
            description_match = re.search(r"- Description:\s*(.*)", block, re.DOTALL)
            raw_data_match = re.search(r"- List of Raw Features Used:\s*(.*)", block, re.DOTALL)
            code_snippet_match = re.search(r"- Code Snippet:\s*(.*)", block, re.DOTALL)

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
                       line.strip().startswith("- Primary Event Impact:") and "Primary Event Impact" not in feature_data or \
                       line.strip().startswith("- List of Raw Features Used:") and "Raw Features" not in feature_data or \
                       line.strip().startswith("- Code Snippet:") and "Code Snippet" not in feature_data :
                        if field_text_match_group.strip() != line.strip(): # make sure it's not the current line itself
                             break
                    cleaned_lines.append(line.strip())
                return " ".join(cleaned_lines).strip()

            derivation_text = derivation_match.group(1) if derivation_match else "N/A"
            justification_text = justification_match.group(1) if justification_match else "N/A"
            description_text = description_match.group(1) if description_match else "N/A"
            feature_data['Derivation'] = clean_multiline_field(block, derivation_text)
            feature_data['Justification'] = clean_multiline_field(block, justification_text)
            feature_data['Primary Event Impact'] = impact_match.group(1).strip() if impact_match else "N/A"
            feature_data['Raw Features'] = clean_multiline_field(block, raw_data_match.group(1) if raw_data_match else "N/A")
            feature_data['Code Snippet'] = clean_multiline_field(block, code_snippet_match.group(1) if code_snippet_match else "N/A")
            feature_data['Description'] = clean_multiline_field(block, description_text)
            
            feature_data['Min'] = 'N/A'
            feature_data['Max'] = 'N/A'
            feature_data['Mean'] = 'N/A'
            feature_data['Data Type'] = 'Engineered (Proposed)'
            
            features_list.append(feature_data)
             # Debugging point to inspect feature_data
        except Exception as e:
            print(f"Error parsing a feature block (index {block_idx}): {e}. Block content: '{block[:100]}...'")
            continue
            
    if not features_list:
        # This case will be handled in the Streamlit app, but good to be aware of
        pass
    print(pd.DataFrame(features_list))
    return pd.DataFrame(features_list)

def apply_recommended_features(current_dataset: pd.DataFrame, recommended_features: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Applies recommended features to the current dataset by executing dynamically generated code.

    Args:
        current_dataset: The input pandas DataFrame to which recommended features will be added.
        recommended_features: A list of dictionaries containing recommended features and their generation logic.

    Returns:
        A new DataFrame with the recommended features appended.

    Raises:
        ValueError: If the generated code fails or the feature cannot be created.
    """
    if current_dataset.empty:
        raise ValueError("The input dataset is empty. Cannot apply recommended features.")

    # Create a copy of the dataset to avoid modifying the original
    updated_dataset = current_dataset.copy()

    for feature_info in recommended_features:
        feature_name = feature_info.get("Feature")
        generated_code = feature_info.get("Code Snippet")  # Assume this contains the code to generate the feature

        if not feature_name or not generated_code:
            raise ValueError(f"Invalid feature recommendation: {feature_info}")

        try:
            # Prepare execution context
            execution_context = {
                'df': updated_dataset,  # Pass the dataset into the context
                'np': np,
                'pd': pd
            }

            # Execute the generated code to create the feature
            exec(generated_code, {}, execution_context)

            # Check if the feature was successfully created
            if feature_name not in updated_dataset.columns:
                raise ValueError(f"Feature '{feature_name}' was not created by the generated code: {generated_code}")

            st.success(f"Successfully created feature '{feature_name}' using AI-generated code.")

        except Exception as e:
            st.error(f"Failed to create feature '{feature_name}'. Error: {e}")
            raise ValueError(f"Error applying recommended feature '{feature_name}': {e}")

    return updated_dataset

def validate_code_snippet(code: str) -> bool:
    """
    Validates the generated code snippet to ensure it is valid Python code.
    Args:
        code (str): The code snippet to validate.
    Returns:
        bool: True if the code is valid, False otherwise.
    """
    try:
        ast.parse(code)  # Parse the code to check for syntax errors
        return True
    except SyntaxError as e:
        st.error(f"Syntax error in generated code: {e}")
        return False

def sanitize_code_snippet(code: str) -> str:
    """
    Extracts and sanitizes the generated code snippet to fix common issues like unterminated string literals.
    Args:
        code (str): The code snippet to sanitize.
    Returns:
        str: The sanitized code snippet.
    """
    try:
        # Extract the code between triple quotes
        if code.startswith("'''") and code.endswith("'''"):
            start_index = code.find("'''") + len("'''")
            end_index = code.rfind("'''")
            extracted_code = code[start_index:end_index].strip()
        else:
            raise ValueError("Code snippet does not start and end with ''' or is improperly formatted.")

        # Format the extracted code to fix indentation issues
        formatted_code = "\n".join([line.strip() for line in extracted_code.splitlines()])

        return formatted_code
    except Exception as e:
        st.error(f"Error sanitizing code snippet: {e}")
        return ""
    
def _dedent_code(code_str):
    """
    Remove common leading whitespace from each line in code_str.
    """
    lines = code_str.strip().split("\n")
    min_indent = float("inf")
    for line in lines:
        if line.strip():  # Ignore empty lines
            min_indent = min(min_indent, len(line) - len(line.lstrip()))
    return "\n".join(line[min_indent:] if line.strip() else "" for line in lines)
