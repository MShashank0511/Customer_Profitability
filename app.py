import streamlit as st
import sys
import os

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Profitability Prediction Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all pages
from pages.data_engineering import show_page as data_engineering_page
from pages.feature_engineering import show_page as feature_engineering_page
from pages.Model_Development import show_page as model_dev_page
from pages.results import show_page as results_page
from pages.Model_Validation import show_page as monitor_page

# Import navbar
from utils.navbar import create_sidebar

def main():
    # Initialize session state for current page if not exists
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Create sidebar navigation
    create_sidebar()
    
    # Main content based on current page
    if st.session_state.current_page == "Home":
        show_home_page()
    elif st.session_state.current_page == "Data Engineering":
        data_engineering_page()
    elif st.session_state.current_page == "Feature Engineering":
        feature_engineering_page()
    elif st.session_state.current_page == "Model Development":
        model_dev_page()
    elif st.session_state.current_page == "Results":
        results_page()
    elif st.session_state.current_page == "Monitor":
        monitor_page()

def show_home_page():
    # Main content
    st.title("Profitability Prediction Platform")
    st.markdown("---")
    
    # Overview Section
    st.header("Overview")
    st.markdown("""
    Welcome to the Profitability Prediction Platform, a comprehensive tool designed to analyze and predict 
    loan profitability. This platform helps financial institutions make data-driven decisions by providing 
    insights into loan performance and potential profitability.
    """)
    
    # Key Metrics Section
    st.header("Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Loans", "1,234", "+5%")
    with col2:
        st.metric("Average Profit", "$2,500", "+3.2%")
    with col3:
        st.metric("Default Rate", "2.5%", "-0.5%")
    
    # Profitability Under Loans Section
    st.header("Profitability Under Loans")
    st.markdown("""
    ### Understanding Loan Profitability
    
    Loan profitability is influenced by several key factors:
    
    1. **Interest Rates and Fees**
       - Primary source of revenue from loans
       - Must balance competitive rates with profitability
       - Fee structures impact overall returns
    
    2. **Risk Assessment**
       - Credit scores and borrower history
       - Loan-to-value ratios
       - Industry and economic conditions
    
    3. **Operational Efficiency**
       - Processing costs
       - Collection efficiency
       - Default management
    
    4. **Market Conditions**
       - Interest rate environment
       - Economic cycles
       - Competitive landscape
    
    ### Key Performance Indicators
    
    - **Net Interest Margin (NIM)**: Difference between interest earned and interest paid
    - **Return on Assets (ROA)**: Measures overall profitability
    - **Loan Loss Provision**: Funds set aside for potential defaults
    - **Cost of Funds**: Interest paid on deposits and borrowings
    
    ### Risk Management
    
    Effective risk management is crucial for maintaining profitability:
    
    - **Credit Risk**: Assessing borrower's ability to repay
    - **Market Risk**: Impact of interest rate changes
    - **Operational Risk**: Internal processes and controls
    - **Compliance Risk**: Regulatory requirements and costs
    
    ### Data-Driven Decision Making
    
    Our platform leverages advanced analytics to:
    
    - Predict loan performance
    - Identify profitable customer segments
    - Optimize pricing strategies
    - Manage risk effectively
    - Forecast future profitability
    """)

if __name__ == "__main__":
    main()
