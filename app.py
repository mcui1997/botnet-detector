import streamlit as st
from data_handler import show_data_tab
from ml_models import show_ml_tab
from dl_models import show_dl_tab
from model_comparison import show_compare_tab

# Page configuration
st.set_page_config(
    page_title="IoT Botnet Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load custom CSS
def load_css():
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'ml_trained' not in st.session_state:
    st.session_state.ml_trained = False
if 'dl_trained' not in st.session_state:
    st.session_state.dl_trained = False
if 'is_sample' not in st.session_state:
    st.session_state.is_sample = False

def main():
    # Main title
    st.markdown('<h1 class="main-title" data-testid="main-title">🛡️ IoT Botnet Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Create tabs with navbar on top
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔧 ML Model", "🧠 DL Model", "⚖️ Compare & Test"])
    
    with tab1:
        show_data_tab()
    
    with tab2:
        show_ml_tab()
    
    with tab3:
        show_dl_tab()
    
    with tab4:
        show_compare_tab()

if __name__ == "__main__":
    main()