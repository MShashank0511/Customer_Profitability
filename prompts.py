new_feature_transformation_prompt ='''You are an expert data science assistant specializing in financial risk modeling.

Context:
You are working with a loan dataset intended to predict loan profitability. Profitability is calculated by estimating the probabilities of two critical events:
1. Charge-Off (COF_EVENT_LABEL) – when a loan is written off.
2. Prepayment (PREPAYMENT_EVENT_LABEL) – when a loan is paid off early.

These probabilities are combined to compute the expected profitability for each loan.

Here is a summary of the available features in my raw dataset:
{dataset_description}

Objective:
You need to perform **feature engineering** using only the provided raw dataset features and recommend a **ranked list of high-value engineered features** that enhance the predictive power of the machine learning models for Charge-Off, Prepayment, and overall Profitability.

Instructions:
1. Use **only the raw dataset features** for engineering. Do not introduce any external features.
2. Identify high-value, **business-relevant** feature engineering opportunities.
3. For each feature you recommend, include:
   - A clear, descriptive name
   - The raw features used to derive it
   - A brief explanation of the derivation logic
   - A Python code snippet (single-line Pandas operation using df) for reproducibility
   - Justification for its inclusion
   - The primary event (Charge-Off / Prepayment / Both) the feature impacts

4. Output Format:
Return your result as a **JSON list of objects** with the following keys for each feature:
json

[
  {
    "Engineered Feature Name": "string",
    "Description": "string",
    "List of Raw Features Used": ["feature_1", "feature_2"],
    "Derivation": "string",
    "Code Snippet": "[[[df['new_feature'] = ...]]]",
    "Justification": "string",
    "Primary Event Impact": "Charge-Off / Prepayment / Both"
  },
  ...
]'''

prompt = '''"You are an expert data science assistant specializing in financial risk modeling.\n\n"
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
        "4. For each engineered feature, clearly state its name, how it's derived,python code to derive the engineered feature, the justification for its inclusion, "
        "and whether it's most relevant for Charge-Off, Prepayment, or Both.\n"
        "5. Provide a Ranked List in this exact format for each feature (ensure to strictly follow this multi-line format for each feature):\n"
        "- Engineered Feature Name: [Name]\n"
        "- Description: [Brief description of the feature]\n"
        "- List of Raw Features Used: [List of raw features used to create this engineered feature]\n"
        "- Derivation: [Explanation of how to compute the feature]\n"
        "- Code Snippet: [Python code snippet to compute the feature], give only the pandas operation code without defining extra functions ,my pandas df is df - just 1 line per feature put code inside [[[code]]] everything except code to be in comments (justification,primary impact, etc..)\n"
        "- Justification: [Why this feature is valuable]\n"
        "- Primary Event Impact: [Charge-Off / Prepayment / Both]\n\n"
        "Begin the list directly. Do not include any preamble before the first feature. Ensure each feature block is separated by a double newline if providing multiple features."
    )'''


def get_code_snippet_gemini(code_snippet: str):
    """
    Calls the Gemini API to recommend engineered features for loan profitability prediction.
    """
    if not GEMINI_API_KEY_CONFIGURED:
        return "Error: Gemini API key not configured. Cannot get recommendations."

    llm_prompt = '''You will be given a section labeled as a "Code Snippet" that may contain non-code text or formatting artifacts. Extract and return only the valid, executable Python code found within it. The output must be in the following JSON format:
    {"code": "<cleaned_python_code>"}
    
    Rules:

    1.Only include actual Python code that can be executed with exec().
    2.Strip away any formatting markers (e.g., [[[code]]], [[[/code]]], markdown formatting, or non-code descriptions).
    3.Ensure the returned code is syntactically correct.
    4.Do not include explanations, comments, or additional text — only the JSON object containing the clean Python code.

    Example Input:

    [[[code]]]df['NEW_FEATURE'] = df['COL1'] / df['COL2'][[[/code]]]

    Expected Output:

    {"code": "df['NEW_FEATURE'] = df['COL1'] / df['COL2']"}

    Now process the following input:
    

    ''' + code_snippet
        
    model = genai.GenerativeModel('gemini-2.0-flash') # Or your preferred Gemini model

    try:
        response = model.generate_content(
            llm_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        import pdb ;pdb.set_trace()
        return response.text
    except Exception as e:
        # Log the error or handle it more gracefully
        print(f"Gemini API Error during generation: {str(e)}")
        return f"An error occurred while contacting Gemini: {str(e)}"


