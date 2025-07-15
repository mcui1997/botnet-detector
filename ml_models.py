"""
IoT Botnet Detection - Machine Learning Models Module
Traditional machine learning approaches for IoT botnet detection with comprehensive 
feature engineering and strict filtering to prevent data leakage.
"""
import streamlit as st
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

def show_ml_tab():
    """
    Display the machine learning model training interface.
    """
    st.markdown('<h2 class="section-header" data-testid="ml-header">🔧 Machine Learning Model</h2>', 
                unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'ml_trained' not in st.session_state:
        st.session_state.ml_trained = False
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = {}
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load a dataset first in the Data tab.")
        return
    
    st.markdown("**Approach:** Traditional ML with feature engineering (PCA, scaling, feature selection)")
    
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
                                ["Random Forest", "Gradient Boosting", "Logistic Regression"],
                                key="ml_algorithm", help="Choose the machine learning algorithm")
        test_size = st.slider("Test Split %", 10, 40, 20, 5,
                             key="ml_test_split", help="Percentage of data for testing")
    
    # Train button
    if st.button("🚀 Train ML Model", key="train_ml_btn"):
        with st.spinner("Training ML model..."):
            try:
                data = st.session_state.data.copy()
                
                if data is None or data.empty:
                    st.error("❌ Data is empty. Please reload your dataset.")
                    return
                
                if 'attack' not in data.columns:
                    st.error("❌ Attack column not found! Load data through the Data tab first.")
                    return
                
                # Feature filtering and preprocessing
                st.info("🔒 Applying feature filtering...")
                
                # Exclude metadata and high-cardinality features
                exclude_cols = [
                    'attack', 'category', 'subcategory', 'pkSeqID', 'stime', 'ltime',
                    'saddr', 'daddr', 'sport', 'dport'
                ]
                
                all_feature_cols = [col for col in data.columns if col not in exclude_cols]
                
                # Remove constant and near-constant features
                viable_features = []
                for col in all_feature_cols:
                    unique_count = data[col].nunique()
                    total_count = len(data[col].dropna())
                    
                    if unique_count > 1 and unique_count / total_count >= 0.01:
                        viable_features.append(col)
                
                # Classify feature types
                force_numerical = [
                    'pkts', 'bytes', 'seq', 'dur', 'mean', 'stddev', 
                    'trans_depth', 'response_body_len',
                    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 
                    'ct_src_dport_ltm', 'ct_dst_sport_ltm'
                ]
                
                allow_categorical = ['proto', 'flgs', 'state']
                
                numerical_cols = []
                categorical_cols = []
                
                for col in viable_features:
                    if col in force_numerical:
                        try:
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                            numerical_cols.append(col)
                        except:
                            continue
                    elif col in allow_categorical:
                        unique_count = data[col].nunique()
                        if unique_count <= 50:
                            categorical_cols.append(col)
                    else:
                        try:
                            numeric_data = pd.to_numeric(data[col], errors='coerce')
                            non_null_ratio = numeric_data.notna().sum() / len(data[col])
                            
                            if non_null_ratio > 0.8:
                                data[col] = numeric_data
                                numerical_cols.append(col)
                            else:
                                unique_count = data[col].nunique()
                                if unique_count <= 20:
                                    categorical_cols.append(col)
                        except:
                            continue
                
                # Create feature matrix
                X = data[numerical_cols].copy()
                
                # Clean numerical features
                for col in numerical_cols:
                    missing_ratio = X[col].isnull().sum() / len(X[col])
                    if missing_ratio > 0.5:
                        X = X.drop(columns=[col])
                        continue
                    
                    if X[col].isnull().sum() > 0:
                        X[col] = X[col].fillna(X[col].median())
                
                # Add categorical features
                if categorical_cols:
                    for col in categorical_cols:
                        try:
                            data[col] = data[col].fillna('unknown')
                            unique_count = data[col].nunique()
                            if unique_count <= 20:
                                le = LabelEncoder()
                                encoded_values = le.fit_transform(data[col].astype(str))
                                X[f'{col}_encoded'] = encoded_values
                        except:
                            continue
                
                # Variance filtering
                variance_threshold = 0.01
                selector = VarianceThreshold(threshold=variance_threshold)
                X_variance_filtered = selector.fit_transform(X)
                selected_features = selector.get_support()
                remaining_feature_names = [col for i, col in enumerate(X.columns) if selected_features[i]]
                X = pd.DataFrame(X_variance_filtered, columns=remaining_feature_names, index=X.index)
                
                # Get target and align indices
                y = data['attack']
                common_indices = X.index.intersection(y.index)
                X = X.loc[common_indices]
                y = y.loc[common_indices]
                
                # Check class distribution
                class_counts = y.value_counts()
                if len(class_counts) < 2:
                    st.error("❌ Only one class found in target variable!")
                    return
                
                # Create balanced dataset
                minority_class = class_counts.idxmin()
                majority_class = class_counts.idxmax()
                minority_count = class_counts.min()
                
                minority_mask = y == minority_class
                majority_mask = y == majority_class
                
                minority_indices = y[minority_mask].index
                majority_indices = y[majority_mask].index
                
                np.random.seed(42)
                majority_sampled_indices = np.random.choice(
                    majority_indices, minority_count, replace=False
                )
                
                balanced_indices = np.concatenate([minority_indices, majority_sampled_indices])
                X_balanced = X.loc[balanced_indices].copy()
                y_balanced = y.loc[balanced_indices].copy()
                
                balanced_counts = y_balanced.value_counts()
                st.success(f"✅ Balanced dataset: Normal: {balanced_counts.get(0, 0):,}, Attacks: {balanced_counts.get(1, 0):,}")
                
                # Train/test split
                test_split = st.session_state.ml_test_split / 100
                X_train, X_test, y_train, y_test = train_test_split(
                    X_balanced, y_balanced, test_size=test_split, random_state=42, stratify=y_balanced
                )
                
                # Feature Engineering Pipeline
                progress_bar = st.progress(0)
                X_train_processed = X_train.copy()
                X_test_processed = X_test.copy()
                
                # Standard Scaling
                scaler = None
                if st.session_state.ml_scaling:
                    scaler = StandardScaler()
                    X_train_processed = pd.DataFrame(
                        scaler.fit_transform(X_train_processed), 
                        columns=X_train_processed.columns, 
                        index=X_train_processed.index
                    )
                    X_test_processed = pd.DataFrame(
                        scaler.transform(X_test_processed), 
                        columns=X_test_processed.columns, 
                        index=X_test_processed.index
                    )
                
                progress_bar.progress(33)
                
                # Feature Selection
                selector = None
                selected_feature_names = list(X_train_processed.columns)
                
                if st.session_state.ml_feature_sel:
                    k_features = min(8, X_train_processed.shape[1])
                    selector = SelectKBest(score_func=f_classif, k=k_features)
                    X_train_selected = selector.fit_transform(X_train_processed, y_train)
                    X_test_selected = selector.transform(X_test_processed)
                    
                    selected_indices = selector.get_support(indices=True)
                    selected_feature_names = [X_train_processed.columns[i] for i in selected_indices]
                    
                    X_train_processed = pd.DataFrame(
                        X_train_selected, 
                        index=X_train_processed.index
                    )
                    X_test_processed = pd.DataFrame(
                        X_test_selected, 
                        index=X_test_processed.index
                    )
                
                progress_bar.progress(66)
                
                # PCA
                pca = None
                if st.session_state.ml_pca:
                    n_components = min(5, X_train_processed.shape[1])
                    pca = PCA(n_components=n_components)
                    X_train_final = pca.fit_transform(X_train_processed)
                    X_test_final = pca.transform(X_test_processed)
                    
                    explained_variance = pca.explained_variance_ratio_.sum()
                    st.info(f"📊 PCA explains {explained_variance:.1%} of variance")
                else:
                    X_train_final = X_train_processed.values
                    X_test_final = X_test_processed.values
                
                progress_bar.progress(80)
                
                # Model Training
                if model_type == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=50,
                        random_state=42, 
                        max_depth=8,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        max_features='sqrt',
                        n_jobs=-1
                    )
                elif model_type == "Gradient Boosting":
                    model = GradientBoostingClassifier(
                        n_estimators=50,
                        random_state=42, 
                        max_depth=5,
                        learning_rate=0.1,
                        min_samples_split=10
                    )
                else:  # Logistic Regression
                    model = LogisticRegression(
                        random_state=42, 
                        max_iter=1000,
                        C=0.1,
                        penalty='l2'
                    )
                
                start_time = time.time()
                model.fit(X_train_final, y_train)
                training_time = time.time() - start_time
                
                progress_bar.progress(100)
                
                # Model Evaluation
                y_pred = model.predict(X_test_final)
                
                # Calculate accuracy and check performance
                accuracy = accuracy_score(y_test, y_pred)
                
                # Calculate all metrics
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
                f1 = f1_score(y_test, y_pred, average='binary', zero_division=0) * 100
                
                # Store results
                st.session_state.ml_results = {
                    "accuracy": accuracy * 100,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "training_time": training_time
                }
                
                # Store pipeline for export
                st.session_state.ml_pipeline = {
                    "model": model,
                    "scaler": scaler,
                    "selector": selector,
                    "pca": pca,
                    "feature_cols": selected_feature_names,
                    "model_type": model_type,
                    "numerical_cols": [col for col in numerical_cols if col in X.columns],
                    "categorical_cols": categorical_cols
                }
                
                st.session_state.ml_trained = True
                st.success("✅ ML Model trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Training error: {str(e)}")
    
    # Results Display
    if st.session_state.ml_trained:
        st.markdown("### 📊 Training Results")
        
        results_col1, results_col2 = st.columns(2)
        
        with results_col1:
            st.metric("Accuracy", f"{st.session_state.ml_results['accuracy']:.1f}%")
        with results_col2:
            st.metric("F1-Score", f"{st.session_state.ml_results['f1_score']:.1f}%")
        
        st.info(f"⏱️ Training Time: {st.session_state.ml_results['training_time']:.2f} seconds")
        