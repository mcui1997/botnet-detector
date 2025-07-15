"""
IoT Botnet Detection - Data Handler Module
This module handles loading and preprocessing of IoT network traffic datasets for botnet detection.
Handles CSV files without headers by providing proper column names.
"""
import streamlit as st
import pandas as pd
import subprocess
import sys
import os

def show_data_tab():
    """
    Display the data loading tab interface.
    """
    st.markdown('<h2 class="section-header" data-testid="data-header">📊 Dataset Overview</h2>', 
                unsafe_allow_html=True)
    
    # Test dataset generator button (top)
    if st.button("🔧 Generate Test Datasets", key="generate_test_data", use_container_width=True):
        run_dataset_generator()
    
    st.markdown("---")  # Visual separator
    
    # Main dataset loading section
    dataset_path = "training_datasets/UNSW_2018_Iot_Botnet_Dataset_2.csv"
    
    if st.button("📂 Load IoT Botnet Dataset", key="load_dataset_btn", use_container_width=True):
        load_data(dataset_path)
    
    # Show current dataset status
    if st.session_state.get('data_loaded', False):
        data = st.session_state.data
        
        st.success(f"✅ Dataset Loaded - {len(data):,} records, {len(data.columns)} features")
        
        # Show class distribution
        if 'attack' in data.columns:
            attack_counts = data['attack'].value_counts()
            normal_count = attack_counts.get(0, 0)
            attack_count = attack_counts.get(1, 0)
            st.info(f"Class Distribution: Normal: {normal_count:,} | Attacks: {attack_count:,}")
        
        # Show attack types breakdown
        if 'category' in data.columns and attack_count > 0:
            attack_types = data[data['attack'] == 1]['category'].value_counts()
            top_attacks = dict(list(attack_types.items())[:5])
            st.info(f"Top Attack Types: {top_attacks}")
        
        # Show preview
        st.markdown("**Dataset Preview:**")
        st.dataframe(data.head(10), use_container_width=True, height=300)

def load_data(file_path):
    """
    Load and preprocess IoT botnet dataset from CSV file.
    """
    try:
        with st.spinner("Loading dataset..."):
            # Define column names for UNSW 2018 IoT Botnet Dataset
            column_names = [
                'pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport',
                'pkts', 'bytes', 'state', 'ltime', 'seq', 'dur', 'mean', 'stddev',
                'res_bps_payload', 'res_pps_payload', 'res_bps_ratio', 'res_pps_ratio',
                'ar_bps_payload', 'ar_pps_payload', 'ar_bps_ratio', 'ar_pps_ratio',
                'ar_bps_delta', 'trans_depth', 'response_body_len', 'ct_srv_src',
                'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
                'ct_dst_sport_ltm', 'category', 'subcategory'
            ]
            
            # Load data without headers
            data = pd.read_csv(file_path, sep=',', header=None, names=column_names)
            
            # Validate category column exists
            if 'category' not in data.columns:
                st.error("❌ Category column not found!")
                return
            
            # Create binary attack labels
            st.info("🔄 Creating binary labels from category data...")
            data['attack'] = (data['category'] != 'Normal').astype(int)
            
            # Show class distribution
            attack_counts = data['attack'].value_counts().sort_index()
            normal_count = attack_counts.get(0, 0)
            attack_count = attack_counts.get(1, 0)
            
            # Class balance analysis
            total_samples = len(data)
            normal_pct = (normal_count / total_samples) * 100
            attack_pct = (attack_count / total_samples) * 100
            
            if len(attack_counts) == 1:
                st.warning("⚠️ Dataset contains only one class - not suitable for binary classification training")
                return
            
            if min(normal_pct, attack_pct) < 1:
                st.warning(f"⚠️ Highly imbalanced: Normal: {normal_pct:.1f}%, Attacks: {attack_pct:.1f}%")
            
            # Identify feature columns
            exclude_cols = [
                'attack', 'category', 'subcategory', 'pkSeqID', 'stime', 'ltime',
                'seq', 'saddr', 'daddr', 'sport', 'dport', 'flgs', 'proto', 'state'
            ]
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            
            # Convert feature columns to numeric
            for col in feature_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Basic data quality checks
            missing_counts = data.isnull().sum()
            total_missing = missing_counts.sum()
            duplicate_count = data.duplicated().sum()
            
            if total_missing > 0:
                st.warning(f"⚠️ Found {total_missing:,} missing values")
            
            if duplicate_count > 0:
                st.warning(f"⚠️ Found {duplicate_count:,} duplicate rows")
            
            # Store in session state
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.session_state.feature_columns = feature_cols
            
        st.success("✅ Dataset loaded successfully!")
        st.rerun()
        
    except FileNotFoundError:
        st.error(f"❌ File not found: {file_path}")
    except pd.errors.EmptyDataError:
        st.error("❌ File is empty")
    except pd.errors.ParserError as e:
        st.error(f"❌ Error parsing CSV: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")

def run_dataset_generator():
    """
    Run the external generate_datasets.py script to create test datasets.
    """
    try:
        # Check if generate_datasets.py exists
        if not os.path.exists("generate_datasets.py"):
            st.error("❌ generate_datasets.py file not found! Please create it first.")
            st.info("💡 Create generate_datasets.py with the dataset generator code.")
            return
        
        with st.spinner("Running dataset generator..."):
            # Run the external Python script with UTF-8 encoding
            result = subprocess.run(
                [sys.executable, "generate_datasets.py"], 
                capture_output=True, 
                text=True,
                encoding='utf-8',
                timeout=60
            )
            
            if result.returncode == 0:
                st.success("✅ Test datasets generated successfully!")
                
                # Check what files were created
                test_folder = "testing_datasets"
                if os.path.exists(test_folder):
                    files = [f for f in os.listdir(test_folder) if f.endswith('.csv')]
                    if files:
                        st.info(f"📁 Created {len(files)} test datasets: {', '.join(files)}")
                        st.info("🎯 Ready for testing! Go to 'Compare & Test' tab to use them.")
                    else:
                        st.warning("⚠️ No CSV files found in testing_datasets folder")
                else:
                    st.warning("⚠️ testing_datasets folder not created")
                    
            else:
                st.error("❌ Error running dataset generator!")
                if result.stderr:
                    st.error(f"Error details: {result.stderr}")
                if result.stdout:
                    st.info(f"Output: {result.stdout}")
                    
    except subprocess.TimeoutExpired:
        st.error("❌ Dataset generation timed out (>60 seconds)")
    except Exception as e:
        st.error(f"❌ Error running generator: {str(e)}")
        st.info("💡 Make sure generate_datasets.py exists and is executable.")