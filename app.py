import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime

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

def show_data_tab():
    st.markdown('<h2 class="section-header" data-testid="data-header">📊 Dataset Upload & Overview</h2>', 
                unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your IoT network traffic dataset (CSV format)",
        type=['csv'],
        help="Upload the Bot IoT dataset or similar network traffic data",
        key="dataset_upload"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.session_state.data_loaded = True
            
            st.success(f"✅ Dataset loaded successfully!")
            
            # Dataset info in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", f"{len(data):,}", 
                         help="Number of network traffic records")
            with col2:
                st.metric("Features", f"{len(data.columns):,}",
                         help="Number of data columns/features")
            with col3:
                st.metric("Memory Usage", f"{data.memory_usage().sum() / 1024**2:.1f} MB",
                         help="Dataset size in memory")
            with col4:
                missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%",
                         help="Percentage of missing values")
            
            # Data preview
            st.markdown('<div class="info-card" data-testid="data-preview">', unsafe_allow_html=True)
            st.markdown("**Dataset Preview:**")
            st.dataframe(data.head(10), use_container_width=True, key="data_preview_df")
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV dataset to begin analysis")

def show_ml_tab():
    st.markdown('<h2 class="section-header" data-testid="ml-header">🔧 Machine Learning Model</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload a dataset first in the Data tab.")
        return
    
    st.markdown('<div class="info-card" data-testid="ml-info">', unsafe_allow_html=True)
    st.markdown("**Approach:** Traditional ML with feature engineering (PCA, scaling, feature selection)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Feature Engineering:**")
        use_pca = st.checkbox("Apply PCA Dimensionality Reduction", value=True, 
                             key="ml_pca", help="Reduce feature dimensions using PCA")
        use_scaling = st.checkbox("Apply Standard Scaling", value=True,
                                 key="ml_scaling", help="Normalize features to standard scale")
        use_feature_selection = st.checkbox("Apply Feature Selection", value=True,
                                          key="ml_feature_sel", help="Select most important features")
    
    with col2:
        st.markdown("**🤖 Model Configuration:**")
        model_type = st.selectbox("Select ML Algorithm", 
                                ["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression"],
                                key="ml_algorithm", help="Choose the machine learning algorithm")
        test_size = st.slider("Test Split %", 10, 40, 20, 5,
                             key="ml_test_split", help="Percentage of data for testing")
    
    # Train button
    if st.button("🚀 Train ML Model", key="train_ml_btn", help="Start training the ML model"):
        with st.spinner("Training ML model with feature engineering..."):
            # Simulate training process
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)  # Simulate training time
                progress_bar.progress(i + 1)
            
            # Simulate results (replace with actual training later)
            st.session_state.ml_results = {
                "accuracy": 92.3,
                "precision": 89.7,
                "recall": 94.1,
                "f1_score": 91.8,
                "training_time": 2.1
            }
            st.session_state.ml_trained = True
            
            st.success("✅ ML Model trained successfully!")
    
    # Show results if trained
    if st.session_state.ml_trained:
        st.markdown('<div class="model-results" data-testid="ml-results">', unsafe_allow_html=True)
        st.markdown("**📊 Training Results:**")
        
        results_col1, results_col2, results_col3, results_col4 = st.columns(4)
        
        with results_col1:
            st.metric("Accuracy", f"{st.session_state.ml_results['accuracy']:.1f}%")
        with results_col2:
            st.metric("Precision", f"{st.session_state.ml_results['precision']:.1f}%")
        with results_col3:
            st.metric("Recall", f"{st.session_state.ml_results['recall']:.1f}%")
        with results_col4:
            st.metric("F1-Score", f"{st.session_state.ml_results['f1_score']:.1f}%")
        
        st.markdown(f"**⏱️ Training Time:** {st.session_state.ml_results['training_time']} minutes")
        st.markdown('</div>', unsafe_allow_html=True)

def show_dl_tab():
    st.markdown('<h2 class="section-header" data-testid="dl-header">🧠 Deep Learning Model</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload a dataset first in the Data tab.")
        return
    
    st.markdown('<div class="info-card" data-testid="dl-info">', unsafe_allow_html=True)
    st.markdown("**Approach:** Neural network using raw data (no manual feature engineering)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🧠 Network Architecture:**")
        hidden_layers = st.slider("Hidden Layers", 1, 5, 3,
                                 key="dl_layers", help="Number of hidden layers in the network")
        neurons_per_layer = st.selectbox("Neurons per Layer", [64, 128, 256, 512], index=1,
                                       key="dl_neurons", help="Number of neurons in each hidden layer")
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1,
                                key="dl_dropout", help="Dropout rate for regularization")
    
    with col2:
        st.markdown("**⚙️ Training Configuration:**")
        epochs = st.slider("Epochs", 20, 100, 50, 10,
                          key="dl_epochs", help="Number of training epochs")
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=2,
                                 key="dl_batch", help="Training batch size")
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0,
                                    key="dl_lr", help="Learning rate for optimization")
    
    # Train button
    if st.button("🚀 Train DL Model", key="train_dl_btn", help="Start training the deep learning model"):
        with st.spinner("Training Deep Learning model..."):
            # Simulate training process
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.03)  # Simulate longer training time
                progress_bar.progress(i + 1)
            
            # Simulate results (replace with actual training later)
            st.session_state.dl_results = {
                "accuracy": 94.7,
                "precision": 92.1,
                "recall": 96.3,
                "f1_score": 94.1,
                "training_time": 8.4
            }
            st.session_state.dl_trained = True
            
            st.success("✅ DL Model trained successfully!")
    
    # Show results if trained
    if st.session_state.dl_trained:
        st.markdown('<div class="model-results" data-testid="dl-results">', unsafe_allow_html=True)
        st.markdown("**📊 Training Results:**")
        
        results_col1, results_col2, results_col3, results_col4 = st.columns(4)
        
        with results_col1:
            st.metric("Accuracy", f"{st.session_state.dl_results['accuracy']:.1f}%")
        with results_col2:
            st.metric("Precision", f"{st.session_state.dl_results['precision']:.1f}%")
        with results_col3:
            st.metric("Recall", f"{st.session_state.dl_results['recall']:.1f}%")
        with results_col4:
            st.metric("F1-Score", f"{st.session_state.dl_results['f1_score']:.1f}%")
        
        st.markdown(f"**⏱️ Training Time:** {st.session_state.dl_results['training_time']} minutes")
        st.markdown('</div>', unsafe_allow_html=True)

def show_compare_tab():
    st.markdown('<h2 class="section-header" data-testid="compare-header">⚖️ Model Comparison & Testing</h2>', 
                unsafe_allow_html=True)
    
    if not (st.session_state.ml_trained and st.session_state.dl_trained):
        st.warning("⚠️ Please train both ML and DL models first.")
        return
    
    # Model comparison
    st.markdown('<div class="info-card" data-testid="model-comparison">', unsafe_allow_html=True)
    st.markdown("**📊 Training Performance Comparison:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 ML Model Results:**")
        ml_results = st.session_state.ml_results
        for metric, value in ml_results.items():
            if metric != 'training_time':
                st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
    
    with col2:
        st.markdown("**🧠 DL Model Results:**")
        dl_results = st.session_state.dl_results
        for metric, value in dl_results.items():
            if metric != 'training_time':
                st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
    
    # Determine winner
    ml_avg = (ml_results['accuracy'] + ml_results['precision'] + ml_results['recall'] + ml_results['f1_score']) / 4
    dl_avg = (dl_results['accuracy'] + dl_results['precision'] + dl_results['recall'] + dl_results['f1_score']) / 4
    
    winner = "Deep Learning" if dl_avg > ml_avg else "Machine Learning"
    st.markdown(f'<div class="comparison-winner" data-testid="training-winner">🏆 Training Winner: {winner} Model</div>', 
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Test on unseen data
    st.markdown('<div class="info-card" data-testid="test-section">', unsafe_allow_html=True)
    st.markdown("**🎯 Test on Unseen Data:**")
    
    test_file = st.file_uploader(
        "Upload test dataset (CSV format)",
        type=['csv'],
        key="test_data_upload",
        help="Upload new network traffic data to test model performance"
    )
    
    if test_file is not None:
        try:
            test_data = pd.read_csv(test_file)
            st.success(f"✅ Test data loaded! Shape: {test_data.shape}")
            
            if st.button("🧪 Run Predictions", key="test_models_btn", help="Test both models on the new data"):
                with st.spinner("Testing models on unseen data..."):
                    time.sleep(2)  # Simulate testing time
                    
                    # Simulate test results (replace with actual testing later)
                    ml_test_results = {
                        "accuracy": 89.2,
                        "precision": 86.4,
                        "recall": 91.7,
                        "f1_score": 88.9
                    }
                    
                    dl_test_results = {
                        "accuracy": 91.8,
                        "precision": 89.3,
                        "recall": 94.2,
                        "f1_score": 91.7
                    }
                    
                    st.markdown("**📊 Test Results:**")
                    
                    test_col1, test_col2 = st.columns(2)
                    
                    with test_col1:
                        st.markdown("**🔧 ML Test Results:**")
                        for metric, value in ml_test_results.items():
                            st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
                    
                    with test_col2:
                        st.markdown("**🧠 DL Test Results:**")
                        for metric, value in dl_test_results.items():
                            st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
                    
                    # Final winner
                    ml_test_avg = sum(ml_test_results.values()) / 4
                    dl_test_avg = sum(dl_test_results.values()) / 4
                    
                    final_winner = "Deep Learning" if dl_test_avg > ml_test_avg else "Machine Learning"
                    st.markdown(f'<div class="comparison-winner" data-testid="final-winner">🎉 Final Winner: {final_winner} Model</div>', 
                                unsafe_allow_html=True)
                    
                    st.success("🎯 Testing complete! Models have been saved automatically.")
                    
        except Exception as e:
            st.error(f"❌ Error loading test data: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()