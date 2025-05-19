import streamlit as st

def create_sidebar():
    # Create a container for the sidebar
    with st.sidebar:
        st.markdown("""
        <style>
            .sidebar { width: 100%; }
            .sidebar .stButton button {
                width: 100%;
                text-align: left;
                padding: 10px;
                margin: 8px 0;
                border-radius: 5px;
                background-color: transparent;
                border: 1px solid #e0e0e0;
            }
            .sidebar .stButton button:hover {
                background-color: #f0f2f6;
            }
            .sidebar .stButton button[data-baseweb="button"] {
                justify-content: flex-start;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("### Navigation")
        
        # Create navigation buttons for all pages
        if st.button("🏠 Home", key="home_btn"):
            st.session_state.current_page = "Home"
            st.rerun()
            
        if st.button("🛠️ Data Engineering", key="data_eng_btn"):
            st.session_state.current_page = "Data Engineering"
            st.rerun()
            
        if st.button("🧪 Feature Engineering", key="feature_eng_btn"):
            st.session_state.current_page = "Feature Engineering"
            st.rerun()
            
        if st.button("🤖 Model Development", key="model_dev_btn"):
            st.session_state.current_page = "Model Development"
            st.rerun()
            
        if st.button("📈 Results", key="results_btn"):
            st.session_state.current_page = "Results"
            st.rerun()
            
        if st.button("👁️ Monitor", key="monitor_btn"):
            st.session_state.current_page = "Monitor"
            st.rerun()

def create_settings():
    pass  # Remove settings functionality
