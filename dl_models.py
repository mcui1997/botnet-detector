"""
IoT Botnet Detection - Deep Learning Models Module
This module implements neural network approaches for IoT botnet detection using raw data 
with minimal feature engineering. Contrasts with ML approach by letting the network 
learn feature representations automatically.

Deep Learning Approach:
- Raw data processing with minimal preprocessing
- Simple normalization (MinMax scaling) only
- No PCA, no feature selection, no manual feature engineering
- Neural network learns patterns and feature representations automatically
- Focus on network architecture and hyperparameter optimization

Training Pipeline:
1. Basic data cleaning and type conversion
2. Simple categorical encoding for non-numeric features
3. Class balancing for imbalanced datasets (same as ML approach)
4. Raw data normalization (0-1 scaling)
5. Neural network architecture configuration
6. Model training with callbacks for optimization
7. Performance evaluation and comparison with ML approach

Author: IoT Security Research
Last Updated: 2025
"""
import streamlit as st
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import plotly.graph_objects as go

def show_dl_tab():
    """
    Display the deep learning model training interface.
    """
    st.markdown('<h2 class="section-header" data-testid="dl-header">🧠 Deep Learning Model</h2>', 
                unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'dl_trained' not in st.session_state:
        st.session_state.dl_trained = False
    if 'dl_results' not in st.session_state:
        st.session_state.dl_results = {}
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load a dataset first in the Data tab.")
        return
    
    st.markdown("**Approach:** Neural network using raw data (minimal feature engineering)")
    st.info("🧠 **DL Philosophy**: Let the network learn feature representations automatically from raw data")
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🧠 Network Architecture:**")
        hidden_layers = st.slider("Hidden Layers", 1, 5, 3,
                                 key="dl_layers", help="Number of hidden layers")
        neurons_per_layer = st.selectbox("Neurons per Layer", [64, 128, 256, 512], index=1,
                                       key="dl_neurons", help="Neurons in each hidden layer")
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1,
                                key="dl_dropout", help="Dropout for regularization")
        activation = st.selectbox("Activation Function", ["relu", "tanh", "sigmoid"], index=0,
                                key="dl_activation", help="Hidden layer activation")
    
    with col2:
        st.markdown("**⚙️ Training Configuration:**")
        epochs = st.slider("Epochs", 20, 100, 50, 10,
                          key="dl_epochs", help="Training epochs")
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=2,
                                 key="dl_batch", help="Training batch size")
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0,
                                    key="dl_lr", help="Learning rate")
        test_size = st.slider("Test Split %", 10, 40, 20, 5,
                             key="dl_test_split", help="Test data percentage")
    
    # Train button
    if st.button("🚀 Train DL Model", key="train_dl_btn"):
        with st.spinner("Training Deep Learning model..."):
            try:
                data = st.session_state.data.copy()
                
                if data is None or data.empty:
                    st.error("❌ Data is empty. Please reload your dataset.")
                    return
                
                if 'attack' not in data.columns:
                    st.error("❌ Attack column not found! Load data through the Data tab first.")
                    return
                
                # Raw data processing with minimal feature engineering
                st.info("🔍 Processing raw data...")
                
                # Exclude metadata and high-cardinality features that cause leakage
                exclude_cols = [
                    'attack', 'category', 'subcategory', 'pkSeqID', 'stime', 'ltime',
                    'saddr', 'daddr', 'sport', 'dport'
                ]
                
                all_feature_cols = [col for col in data.columns if col not in exclude_cols]
                
                # Filter constant and near-constant features
                viable_features = []
                for col in all_feature_cols:
                    unique_count = data[col].nunique()
                    total_count = len(data[col].dropna())
                    
                    if unique_count > 1 and unique_count / total_count >= 0.01:
                        viable_features.append(col)
                
                # Feature type classification
                numerical_cols = []
                categorical_cols = []
                
                # Force numerical features
                force_numerical = [
                    'pkts', 'bytes', 'seq', 'dur', 'mean', 'stddev', 
                    'trans_depth', 'response_body_len',
                    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 
                    'ct_src_dport_ltm', 'ct_dst_sport_ltm'
                ]
                
                # Allow limited categorical features
                allow_categorical = ['proto', 'flgs', 'state']
                
                for col in viable_features:
                    if col in force_numerical:
                        try:
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                            numerical_cols.append(col)
                        except:
                            continue
                    elif col in allow_categorical:
                        unique_count = data[col].nunique()
                        if unique_count <= 20:
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
                
                # Create raw feature matrix
                X_raw = data[numerical_cols].copy()
                
                # Basic cleaning for numerical features
                for col in numerical_cols:
                    missing_ratio = X_raw[col].isnull().sum() / len(X_raw[col])
                    if missing_ratio > 0.5:
                        X_raw = X_raw.drop(columns=[col])
                        continue
                    
                    if X_raw[col].isnull().sum() > 0:
                        X_raw[col] = X_raw[col].fillna(X_raw[col].median())
                
                # Simple encoding for categorical features
                label_encoders = {}
                if categorical_cols:
                    for col in categorical_cols:
                        if col in data.columns:
                            try:
                                data[col] = data[col].fillna('unknown')
                                unique_count = data[col].nunique()
                                if unique_count <= 20:
                                    le = LabelEncoder()
                                    encoded_values = le.fit_transform(data[col].astype(str))
                                    X_raw[f'{col}_encoded'] = encoded_values
                                    label_encoders[col] = le
                            except:
                                continue
                
                # Get target variable and align indices
                y = data['attack']
                common_indices = X_raw.index.intersection(y.index)
                X_raw = X_raw.loc[common_indices]
                y = y.loc[common_indices]
                
                # Check class distribution
                class_counts = y.value_counts()
                if len(class_counts) < 2:
                    st.error("❌ Only one class found!")
                    return
                
                # Create balanced dataset (same approach as ML)
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
                X_balanced = X_raw.loc[balanced_indices].copy()
                y_balanced = y.loc[balanced_indices].copy()
                
                balanced_counts = y_balanced.value_counts()
                st.success(f"✅ Balanced dataset: Normal: {balanced_counts.get(0, 0):,}, Attacks: {balanced_counts.get(1, 0):,}")
                
                # Train/test split
                test_split = st.session_state.dl_test_split / 100
                X_train, X_test, y_train, y_test = train_test_split(
                    X_balanced, y_balanced, test_size=test_split, random_state=42, stratify=y_balanced
                )
                
                progress_bar = st.progress(0)
                
                # Raw data normalization - simple MinMax scaling
                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                progress_bar.progress(33)
                
                # Build neural network architecture
                model = keras.Sequential()
                
                # Input layer
                model.add(layers.Input(shape=(X_train_scaled.shape[1],)))
                
                # Hidden layers with dropout
                for i in range(hidden_layers):
                    model.add(layers.Dense(neurons_per_layer, activation=activation))
                    model.add(layers.Dropout(dropout_rate))
                
                # Output layer for binary classification
                model.add(layers.Dense(1, activation='sigmoid'))
                
                # Compile model
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                progress_bar.progress(66)
                
                # Train the neural network
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        patience=10, 
                        restore_best_weights=True, 
                        verbose=0
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=5, 
                        factor=0.5, 
                        verbose=0
                    )
                ]
                
                start_time = time.time()
                
                history = model.fit(
                    X_train_scaled, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )
                
                training_time = time.time() - start_time
                
                progress_bar.progress(90)
                
                # Model evaluation
                y_pred_proba = model.predict(X_test_scaled, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Performance assessment
                if accuracy == 1.0:
                    st.warning("⚠️ Perfect accuracy detected - potential overfitting")
                elif accuracy > 0.98:
                    st.warning(f"⚠️ Very high accuracy ({accuracy:.1%}) - check for overfitting")
                elif accuracy > 0.85:
                    st.success(f"✅ Good performance ({accuracy:.1%}) - realistic and generalizable")
                else:
                    st.info(f"📊 Moderate performance ({accuracy:.1%})")
                
                progress_bar.progress(100)
                
                # Store results
                st.session_state.dl_results = {
                    "accuracy": accuracy * 100,
                    "precision": precision * 100,
                    "recall": recall * 100,
                    "f1_score": f1 * 100,
                    "training_time": training_time
                }
                
                # Store complete pipeline for export
                st.session_state.dl_pipeline = {
                    "model": model,
                    "scaler": scaler,
                    "feature_cols": list(X_balanced.columns),
                    "history": history,
                    "label_encoders": label_encoders,
                    "numerical_cols": numerical_cols,
                    "categorical_cols": categorical_cols
                }
                
                st.session_state.dl_trained = True
                st.success("✅ Deep Learning model trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Training error: {str(e)}")
    
    # Results Display
    if st.session_state.dl_trained:
        st.markdown("### 📊 Training Results")
        
        results_col1, results_col2, results_col3, results_col4 = st.columns(4)
        
        with results_col1:
            st.metric("Accuracy", f"{st.session_state.dl_results['accuracy']:.1f}%")
        with results_col2:
            st.metric("Precision", f"{st.session_state.dl_results['precision']:.1f}%")
        with results_col3:
            st.metric("Recall", f"{st.session_state.dl_results['recall']:.1f}%")
        with results_col4:
            st.metric("F1-Score", f"{st.session_state.dl_results['f1_score']:.1f}%")
        
        st.info(f"⏱️ Training Time: {st.session_state.dl_results['training_time']:.2f} seconds")
        
        # DL Approach Summary
        st.markdown("### 🧠 Deep Learning Approach Used")
        st.success("""
        ✅ **Raw data processing** - minimal feature engineering
        ✅ **Simple MinMax normalization** - no PCA or feature selection  
        ✅ **Neural network architecture** - learns features automatically
        ✅ **Focus on network hyperparameters** - layers, neurons, dropout
        """)
        
        # Training History Visualization
        if 'dl_pipeline' in st.session_state and 'history' in st.session_state.dl_pipeline:
            st.markdown("### 📈 Training History")
            
            history = st.session_state.dl_pipeline['history']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy plot
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    y=history.history['accuracy'],
                    mode='lines+markers',
                    name='Training Accuracy',
                    line=dict(color='blue')
                ))
                if 'val_accuracy' in history.history:
                    fig_acc.add_trace(go.Scatter(
                        y=history.history['val_accuracy'],
                        mode='lines+markers',
                        name='Validation Accuracy',
                        line=dict(color='red')
                    ))
                fig_acc.update_layout(
                    title="Model Accuracy",
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy",
                    height=300
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                # Loss plot
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=history.history['loss'],
                    mode='lines+markers',
                    name='Training Loss',
                    line=dict(color='blue')
                ))
                if 'val_loss' in history.history:
                    fig_loss.add_trace(go.Scatter(
                        y=history.history['val_loss'],
                        mode='lines+markers',
                        name='Validation Loss',
                        line=dict(color='red')
                    ))
                fig_loss.update_layout(
                    title="Model Loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=300
                )
                st.plotly_chart(fig_loss, use_container_width=True)
        
        # Model export information
        st.markdown("### 💾 Model Ready for Export")
        st.success("📦 DL model pipeline stored and ready for testing on unseen data")