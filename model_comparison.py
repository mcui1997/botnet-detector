"""
IoT Botnet Detection - Model Comparison & Testing Module
This module handles comparison between trained ML and DL models and testing on unseen data.
Key feature: Properly hides labels during testing to simulate real-world deployment.

Testing Pipeline:
1. Load test data with labels (for evaluation)
2. Extract true labels and convert to binary (Normal=0, Attack=1)
3. Remove all label columns to simulate unseen data
4. Apply same preprocessing as training (same feature filtering)
5. Test both models on unlabeled data
6. Compare predictions to true labels for performance evaluation

Author: IoT Security Research
Last Updated: 2025
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

def show_compare_tab():
    """
    Main function for the model comparison and testing tab.
    Shows training comparison and allows testing on new data.
    """
    st.markdown('<h2 class="section-header" data-testid="compare-header">⚖️ Model Comparison & Testing</h2>', 
                unsafe_allow_html=True)
    
    # Check if both models are trained
    if not (st.session_state.get('ml_trained', False) and st.session_state.get('dl_trained', False)):
        st.warning("⚠️ Please train both ML and DL models first.")
        return
    
    # ===== TRAINING PERFORMANCE COMPARISON =====
    show_training_comparison()
    
    # ===== TEST ON NEW DATA SECTION =====
    show_testing_section()

def show_training_comparison():
    """Display the training performance comparison between ML and DL models"""
    
    st.markdown("### 📊 Training Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    # ML Results
    with col1:
        st.markdown("**🔧 ML Model Results:**")
        ml_results = st.session_state.ml_results
        for metric, value in ml_results.items():
            if metric != 'training_time':
                st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
    
    # DL Results
    with col2:
        st.markdown("**🧠 DL Model Results:**")
        dl_results = st.session_state.dl_results
        for metric, value in dl_results.items():
            if metric != 'training_time':
                st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
    
    # Determine training winner
    ml_avg = (ml_results['accuracy'] + ml_results['precision'] + ml_results['recall'] + ml_results['f1_score']) / 4
    dl_avg = (dl_results['accuracy'] + dl_results['precision'] + dl_results['recall'] + dl_results['f1_score']) / 4
    
    winner = "Deep Learning" if dl_avg > ml_avg else "Machine Learning"
    st.success(f"🏆 Training Winner: {winner} Model ({max(ml_avg, dl_avg):.1f}% avg)")

def show_testing_section():
    """Handle the test dataset selection and testing functionality"""
    
    st.markdown("### 🎯 Test Models on Unseen Data")
    st.info("🤖 **Test real botnet detection capability on new network traffic**")
    
    # Get available test files
    test_files = get_available_test_files()
    
    if not test_files:
        st.error("❌ No test files found in 'testing_datasets/' folder")
        return
    
    # Dropdown to select test file
    selected_file = st.selectbox(
        "Choose test dataset:",
        options=test_files,
        format_func=lambda x: x,
        key="test_file_selector",
        help="Select which test dataset to use for botnet detection"
    )
    
    if selected_file:
        # Load button
        if st.button("📂 Load Test Dataset", 
                    key="load_test_data", 
                    help="Load the selected test dataset",
                    use_container_width=True):
            
            # Load and validate the test data
            test_data = load_and_validate_test_data(selected_file)
            
            if test_data is not None:
                # Store in session state
                st.session_state.current_test_data = test_data
                st.session_state.current_test_file = selected_file
                
        # Show test data info and run button if data is loaded
        if hasattr(st.session_state, 'current_test_data') and st.session_state.current_test_data is not None:
            test_data = st.session_state.current_test_data
            
            # Simple file info
            st.success(f"✅ Data loaded: {test_data.shape[0]:,} network traffic samples")
            
            # Big test button
            st.markdown("---")
            if st.button("🚀 TEST BOTNET DETECTION", 
                        key="run_tests", 
                        help="Run botnet detection on both ML and DL models",
                        use_container_width=True):
                
                # Run the actual testing
                run_model_tests(test_data)

def get_available_test_files():
    """
    Scan the testing_datasets folder for available CSV files.
    Returns a list of filenames.
    """
    import os
    
    test_folder = "testing_datasets"
    
    if not os.path.exists(test_folder):
        st.error(f"❌ Testing folder '{test_folder}' not found!")
        return []
    
    try:
        # Get all CSV files in the testing folder
        files = [f for f in os.listdir(test_folder) if f.endswith('.csv')]
        files.sort()
        return files
        
    except Exception as e:
        st.error(f"❌ Error reading testing folder: {e}")
        return []

def load_and_validate_test_data(filename):
    """
    Load test data and assign proper column names (same as training).
    Returns the dataframe if successful, None if failed.
    """
    try:
        file_path = f"testing_datasets/{filename}"
        
        # Define same column names as training data (35 columns)
        column_names = [
            'pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport',
            'pkts', 'bytes', 'state', 'ltime', 'seq', 'dur', 'mean', 'stddev',
            'res_bps_payload', 'res_pps_payload', 'res_bps_ratio', 'res_pps_ratio',
            'ar_bps_payload', 'ar_pps_payload', 'ar_bps_ratio', 'ar_pps_ratio',
            'ar_bps_delta', 'trans_depth', 'response_body_len', 'ct_srv_src',
            'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
            'ct_dst_sport_ltm', 'category', 'subcategory'
        ]
        
        # Load data without headers (same as training)
        test_data = pd.read_csv(file_path, sep=',', header=None, names=column_names)
        
        # Validate we have required columns
        if 'category' not in test_data.columns:
            st.error("❌ Invalid test file format!")
            return None
        
        return test_data
        
    except Exception as e:
        st.error(f"❌ Error loading test data: {str(e)}")
        return None

def run_model_tests(test_data):
    """
    Run both ML and DL models on test data and display results.
    This is where the actual model testing happens with proper label hiding.
    """
    try:
        with st.spinner("🔍 Analyzing network traffic for botnet activity..."):
            
            # Extract true labels for evaluation (before hiding)
            true_categories = test_data['category'].copy()
            true_binary = (true_categories != 'Normal').astype(int)
            
            # Remove ALL label information (simulate unseen data)
            unlabeled_data = test_data.drop(['category', 'subcategory'], axis=1, errors='ignore')
            
            # Test ML Model
            ml_test_results, ml_predictions = test_ml_model(unlabeled_data, true_binary)
            
            # Test DL Model
            dl_test_results, dl_predictions = test_dl_model(unlabeled_data, true_binary)
            
            # Display results
            display_botnet_detection_results(ml_test_results, dl_test_results, ml_predictions, dl_predictions, true_categories, true_binary)
            
    except Exception as e:
        st.error(f"❌ Error during testing: {str(e)}")

def test_ml_model(unlabeled_data, true_binary):
    """
    Test the trained ML model on unlabeled data.
    Applies the same preprocessing pipeline as training.
    """
    # Get the saved ML pipeline
    ml_pipeline = st.session_state.ml_pipeline
    model = ml_pipeline["model"]
    scaler = ml_pipeline["scaler"] 
    selector = ml_pipeline["selector"]
    pca = ml_pipeline["pca"]
    
    # Preprocess test data exactly like training data (no labels!)
    X_test = preprocess_data_for_ml(unlabeled_data)
    
    # Apply the saved ML pipeline transformations
    X_test_final = apply_ml_pipeline(X_test, scaler, selector, pca)
    
    # Make predictions (model sees only features, no labels!)
    y_pred = model.predict(X_test_final)
    
    # Calculate and return metrics
    return calculate_metrics(true_binary, y_pred), y_pred

def test_dl_model(unlabeled_data, true_binary):
    """
    Test the trained DL model on unlabeled data.
    Applies the same preprocessing pipeline as training.
    """
    # Get the saved DL pipeline
    dl_pipeline = st.session_state.dl_pipeline
    model = dl_pipeline["model"]
    scaler = dl_pipeline["scaler"]
    label_encoders = dl_pipeline.get("label_encoders", {})
    
    # Preprocess test data exactly like training data (no labels!)
    X_test = preprocess_data_for_dl(unlabeled_data, label_encoders)
    
    # Apply scaling
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test.values
    
    # Make predictions (model sees only features, no labels!)
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate and return metrics
    return calculate_metrics(true_binary, y_pred), y_pred

def preprocess_data_for_ml(unlabeled_data):
    """
    Preprocess unlabeled test data exactly like ML training data.
    Uses the exact same features that the ML model was trained on.
    Returns X_test (features only, no labels).
    """
    # Copy data
    data = unlabeled_data.copy()
    
    # Get the exact feature columns used during ML training
    ml_pipeline = st.session_state.ml_pipeline
    trained_numerical_cols = ml_pipeline.get("numerical_cols", [])
    trained_categorical_cols = ml_pipeline.get("categorical_cols", [])
    
    # Create feature matrix using EXACT same features as training
    X_test = pd.DataFrame()
    
    # Add numerical features (must match training exactly)
    for col in trained_numerical_cols:
        if col in data.columns:
            # Convert to numeric and clean
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Fill missing values with median
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].median())
            
            X_test[col] = data[col]
        else:
            # Create dummy column with zeros if missing
            X_test[col] = 0
    
    # Add categorical features (encode same as training)
    for col in trained_categorical_cols:
        if col in data.columns:
            try:
                # Handle missing values
                data[col] = data[col].fillna('unknown')
                
                # Simple label encoding (same as training)
                le = LabelEncoder()
                encoded_values = le.fit_transform(data[col].astype(str))
                X_test[f'{col}_encoded'] = encoded_values
            except:
                X_test[f'{col}_encoded'] = 0
        else:
            X_test[f'{col}_encoded'] = 0
    
    # Ensure we have exact same features as training
    expected_features = trained_numerical_cols + [f'{col}_encoded' for col in trained_categorical_cols]
    
    # Add any missing features with zeros
    for feature in expected_features:
        if feature not in X_test.columns:
            X_test[feature] = 0
    
    # Remove any extra features
    extra_features = [col for col in X_test.columns if col not in expected_features]
    if extra_features:
        X_test = X_test.drop(columns=extra_features)
    
    # Reorder columns to match training order
    X_test = X_test[expected_features]
    
    return X_test

def preprocess_data_for_dl(unlabeled_data, label_encoders):
    """
    Preprocess unlabeled test data exactly like DL training data.
    Uses the exact same features that the DL model was trained on.
    Returns X_test (features only, no labels).
    """
    # Copy data
    data = unlabeled_data.copy()
    
    # Get the exact feature columns used during DL training
    dl_pipeline = st.session_state.dl_pipeline
    trained_numerical_cols = dl_pipeline.get("numerical_cols", [])
    trained_categorical_cols = dl_pipeline.get("categorical_cols", [])
    
    # Create feature matrix using EXACT same features as training
    X_test = pd.DataFrame()
    
    # Add numerical features (must match training exactly)
    for col in trained_numerical_cols:
        if col in data.columns:
            # Convert to numeric and clean
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Fill missing values with median
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].median())
            
            X_test[col] = data[col]
        else:
            # Create dummy column with zeros if missing
            X_test[col] = 0
    
    # Add categorical features (use saved encoders if available)
    for col in trained_categorical_cols:
        if col in data.columns:
            try:
                # Handle missing values
                data[col] = data[col].fillna('unknown')
                
                if label_encoders and col in label_encoders:
                    # Use saved encoder
                    le = label_encoders[col]
                    unique_values = set(le.classes_)
                    test_values = data[col].astype(str)
                    test_values = test_values.apply(lambda x: x if x in unique_values else 'unknown')
                    
                    if 'unknown' not in unique_values:
                        le.classes_ = np.append(le.classes_, 'unknown')
                    
                    encoded_values = le.transform(test_values)
                    X_test[f'{col}_encoded'] = encoded_values
                else:
                    # Create new encoder
                    le = LabelEncoder()
                    encoded_values = le.fit_transform(data[col].astype(str))
                    X_test[f'{col}_encoded'] = encoded_values
            except:
                X_test[f'{col}_encoded'] = 0
        else:
            X_test[f'{col}_encoded'] = 0
    
    # Ensure we have exact same features as training
    expected_features = trained_numerical_cols + [f'{col}_encoded' for col in trained_categorical_cols]
    
    # Add any missing features with zeros
    for feature in expected_features:
        if feature not in X_test.columns:
            X_test[feature] = 0
    
    # Remove any extra features
    extra_features = [col for col in X_test.columns if col not in expected_features]
    if extra_features:
        X_test = X_test.drop(columns=extra_features)
    
    # Reorder columns to match training order
    X_test = X_test[expected_features]
    
    return X_test

def apply_ml_pipeline(X_test, scaler, selector, pca):
    """
    Apply the saved ML pipeline transformations to test data.
    Returns the final feature matrix ready for prediction.
    """
    X_processed = X_test.copy()
    
    # Apply scaling
    if scaler is not None:
        X_scaled = scaler.transform(X_processed)
        X_processed = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)
    
    # Apply feature selection
    if selector is not None:
        X_selected = selector.transform(X_processed)
        X_processed = pd.DataFrame(X_selected, index=X_processed.index)
    
    # Apply PCA
    if pca is not None:
        X_final = pca.transform(X_processed)
    else:
        X_final = X_processed.values
    
    return X_final

def display_botnet_detection_results(ml_results, dl_results, ml_predictions, dl_predictions, true_categories, true_binary):
    """Display the botnet detection results in a user-friendly format"""
    
    st.markdown("---")
    st.markdown("## 🤖 **BOTNET DETECTION RESULTS**")
    
    # Analyze what's in the test data
    category_counts = true_categories.value_counts()
    total_samples = len(true_binary)
    attack_samples = (true_binary == 1).sum()
    normal_samples = (true_binary == 0).sum()
    
    # Determine if this is botnet traffic
    is_botnet_dataset = attack_samples > normal_samples
    attack_percentage = (attack_samples / total_samples) * 100
    
    # Dataset analysis
    if is_botnet_dataset:
        st.error(f"🚨 **BOTNET TRAFFIC DETECTED** - {attack_percentage:.1f}% malicious activity")
        dominant_attack = category_counts.index[0] if category_counts.index[0] != 'Normal' else category_counts.index[1]
        st.warning(f"⚠️ **Primary threat**: {dominant_attack} attacks")
    else:
        st.success(f"✅ **NORMAL NETWORK TRAFFIC** - {100-attack_percentage:.1f}% legitimate activity")
    
    st.info(f"📊 **Analysis**: {total_samples:,} network flows analyzed")
    
    # Model performance comparison
    st.markdown("### 🏆 **Model Performance Comparison**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 ML Model Detection:**")
        ml_detected = (ml_predictions == 1).sum()
        ml_detection_rate = (ml_detected / attack_samples) * 100 if attack_samples > 0 else 0
        
        if is_botnet_dataset:
            st.metric("Botnet Detection Rate", f"{ml_detection_rate:.1f}%")
            st.metric("Threats Detected", f"{ml_detected:,} / {attack_samples:,}")
        else:
            false_positives = (ml_predictions == 1).sum()
            st.metric("False Alarms", f"{false_positives:,}")
        
        st.metric("Overall Accuracy", f"{ml_results['accuracy']:.1f}%")
    
    with col2:
        st.markdown("**🧠 DL Model Detection:**")
        dl_detected = (dl_predictions == 1).sum()
        dl_detection_rate = (dl_detected / attack_samples) * 100 if attack_samples > 0 else 0
        
        if is_botnet_dataset:
            st.metric("Botnet Detection Rate", f"{dl_detection_rate:.1f}%")
            st.metric("Threats Detected", f"{dl_detected:,} / {attack_samples:,}")
        else:
            false_positives = (dl_predictions == 1).sum()
            st.metric("False Alarms", f"{false_positives:,}")
            
        st.metric("Overall Accuracy", f"{dl_results['accuracy']:.1f}%")
    
    # Determine winner
    ml_avg = sum(ml_results.values()) / len(ml_results)
    dl_avg = sum(dl_results.values()) / len(dl_results)
    
    final_winner = "Machine Learning" if ml_avg > dl_avg else "Deep Learning"
    winning_score = max(ml_avg, dl_avg)
    
    st.success(f"""
    🏆 **Best Performer: {final_winner} Model**  
    📈 **Detection Accuracy**: {winning_score:.1f}%
    """)
    
    # Performance interpretation
    if winning_score > 90:
        st.success("✅ **Excellent Detection** - Models demonstrate strong botnet identification capability")
    elif winning_score > 80:
        st.info("📊 **Good Detection** - Models show solid performance on this traffic type")
    elif winning_score > 70:
        st.warning("⚠️ **Moderate Detection** - Models may need additional training")
    else:
        st.error("❌ **Poor Detection** - Models struggle with this traffic pattern")
    
    # Store results for later use
    st.session_state.ml_test_results = ml_results
    st.session_state.dl_test_results = dl_results
    
    st.success("🎯 **Analysis Complete!** Botnet detection testing finished successfully.")

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }