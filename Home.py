import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Mortage Analytics Tool",
    page_icon="🏦",
    layout="wide",
)
import html

def data_loaded_file_uploader(show: bool = True):
    if show:
        # Custom CSS + container with bubble style
        st.markdown(
            """
            <style>
            .bubble-container {
                background-color: #E9F5FE;
                border-radius: 18px;
                padding: 15px 20px;
                margin: 10px 0px;
                max-width: 600px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .bubble-label {
                font-weight: bold;
                margin-bottom: 8px;
                color: #0A2540;
                font-size: 18px;
            }
            </style>
            <div class="bubble-container">
                <div class="bubble-label">Data Loaded</div>
            """,
            unsafe_allow_html=True,
        )
        
        # Show file uploader inside this container
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
        
        # Closing the div container
        st.markdown("</div>", unsafe_allow_html=True)
        return uploaded_file
    else:
        # If show is False, don't show anything
        return None
    
# Path to logo image
# logo_path = os.path.join("assets", "static/logo.png")
# print(f"Logo path exists: {os.path.exists(logo_path)}")

# Create three columns: left (logo), center (header), right (contact bubble)
col1, col2, col3 = st.columns([1, 20, 5])

with col1:
    st.image("cropped-Sigmoid_logo_3x.png", width=100)

with col2:
    st.markdown(
        """
        <style>
        .dynamic-title {
            text-align: center;
            margin-left: auto;
            margin-right: auto;
            font-size: 32px; /* Reduced font size */
            font-weight: bold;
        }
        </style>
        <h1 class="dynamic-title"> Sigmoid Profitability Analytics Tool</h1>
        """,
        unsafe_allow_html=True,
    )

with col3:
    # Dropdown-style contact bubble
    st.markdown("""
    <style>
    .dropdown {
        position: relative;
        display: inline-block;
        margin-top: 10px;
        float: right;
    }

    .dropdown-button {
        background-color: #E9F5FE;
        color: #0A2540;
        padding: 10px 16px;
        font-size: 14px;
        border: none;
        cursor: pointer;
        border-radius: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .dropdown-content {
        display: none;
        position: absolute;
        right: 0;
        background-color: #F9FAFB;
        min-width: 180px;
        padding: 10px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        z-index: 1;
    }

    .dropdown:hover .dropdown-content {
        display: block;
    }

    .dropdown-content a {
        color: #0A2540;
        text-decoration: none;
        display: block;
        font-size: 14px;
        margin-top: 5px;
    }
    </style>

    <div class="dropdown">
      <button class="dropdown-button">📞 Contact Us</button>
      <div class="dropdown-content">
        <b>Ravi Bajagur</b><br>
        <a href="tel:8959896843">8959896843</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Adjust spacing dynamically based on sidebar visibility
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] + div {
        margin-left: 150px; /* Adjust this value to control spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
    <style>
    .intro-text {
        font-size: 21px;
        font-weight: 600;
        color: #000000;
        line-height: 1.7;
        background-color: #f5f7fa;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

#<!-- # Layout using columns
# left, right = st.columns([1.4, 1]) -->

# <!-- with left: -->
st.markdown("""
    <div class="intro-text">
        Welcome to the Profitability Analytics Tool – Your intelligent assistant for loan profitability monitoring and performance optimization.
This platform leverages data science to compare predicted and actual outcomes, identify model drift, and fine-tune decisions in real time.
Visualize profitability, charge-off, and prepayment trends with ease—filtered by any feature combination.
Upload your data, ask questions, and get instant clarity through conversational intelligence.
    </div>
""", unsafe_allow_html=True)

# Add buttons for MVP and Customized Version
  # Add a horizontal line for separation

# Use columns for button layout and apply custom CSS for button style
button_css = """
<style>
.stButton > button {
    font-size: 35px;
    font-weight: bold;
    color: #000000;
    background-color: white;
    border: none;
    border-radius: 12px;
    padding: 15px 30px;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    transition: background-color 0.3s ease, color 0.3s ease;
    margin: 0 10px;
}
.stButton > button:hover {
    background-color: #000000;
    color: white;
}
</style>
"""
st.markdown(button_css, unsafe_allow_html=True)


st.markdown("---")

# MVP Version description and button
st.markdown("""
#### Ready to Use Model
Provides a streamlined workflow that focuses on core functionalities, allowing users to quickly calculate profitability using the available input data with minimal processing.
""")
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
if st.button("Ready to Use Model", key="mvp_version_btn", use_container_width=True):
    st.session_state["app_version"] = "MVP"
    st.switch_page("pages/Exploratory_Data_Analysis.py")  # Adjust path if needed

st.markdown("---")

# Customized Version description and button
st.markdown("""
#### Customized Version
Offers the complete set of functionalities across all pages, including transformations, AI recommendations, detailed analysis, and advanced profitability simulation.
""")
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
if st.button("Customized Version", key="custom_version_btn", use_container_width=True):
    st.session_state["app_version"] = "Customized"
    st.switch_page("pages/Exploratory_Data_Analysis.py")  # Adjust path if needed