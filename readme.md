# IoT Botnet Detection: Machine Learning vs Deep Learning Comparison

## Project Overview

This project implements and compares traditional Machine Learning (ML) and Deep Learning (DL) approaches for detecting IoT botnet behavior in network traffic. The system provides a complete pipeline for training both model types on the same dataset and evaluating their performance on unseen data to determine which approach better generalizes to new attack patterns.

## Research Objective

To empirically compare the effectiveness of feature engineering-based machine learning versus raw data processing deep learning for cybersecurity threat detection in IoT network environments.

## Dataset Requirements

The system is designed to work with network traffic datasets in CSV format containing:
- 35 columns of network flow features (packet counts, timing, protocol information)
- Binary classification labels (Normal vs Attack traffic)
- Support for various attack types (Reconnaissance, DoS, DDoS, etc.)

**Training Data Format:**
```
pkSeqID,stime,flgs,proto,saddr,sport,daddr,dport,pkts,bytes,state,ltime,seq,dur,mean,stddev,[...],category,subcategory
```

## Methodology

### Machine Learning Approach
**Philosophy:** Heavy feature engineering with traditional algorithms

**Processing Pipeline:**
1. Aggressive feature filtering to prevent data leakage
2. Constant and near-constant feature removal
3. Standard scaling and normalization
4. Feature selection using statistical methods
5. Principal Component Analysis (PCA) for dimensionality reduction
6. Traditional algorithms: Random Forest, Gradient Boosting, Logistic Regression

**Key Characteristics:**
- Manual feature engineering and selection
- Mathematical transforms and scaling
- Reduced feature space (typically 8-10 features → 5 PCA components)
- Focus on engineered feature relationships

### Deep Learning Approach
**Philosophy:** Minimal preprocessing with neural network feature learning

**Processing Pipeline:**
1. Basic feature filtering (same as ML for fair comparison)
2. Simple MinMax normalization (0-1 scaling)
3. Direct neural network input without feature engineering
4. Multi-layer feedforward architecture with dropout regularization
5. Automatic feature representation learning

**Key Characteristics:**
- Raw data processing approach
- Minimal manual feature engineering
- Neural network learns feature representations
- Focus on network architecture and hyperparameters

## Setup and Installation

### Prerequisites
```bash
python 3.8+
pip install streamlit pandas numpy scikit-learn tensorflow plotly
```

### Directory Structure
Create the following directory structure before running:
```
project/
├── app.py
├── data_handler.py
├── ml_models.py
├── dl_models.py
├── model_comparison.py
├── training_datasets/
│   └── UNSW_2018_Iot_Botnet_Dataset_1.csv
└── testing_datasets/
    ├── UNSW_2018_IoT_Botnet_Dataset_3.csv
    └── UNSW_2018_IoT_Botnet_Dataset_4.csv
```

### Data Preparation
1. Place training dataset in `training_datasets/` folder
2. Place test datasets in `testing_datasets/` folder
3. Ensure all CSV files follow the 35-column format without headers
4. Update the hardcoded path in `data_handler.py` if using different filenames

## Running the Application

### Launch Application
```bash
streamlit run app.py
```

### Complete Workflow

**Step 1: Data Loading**
1. Navigate to the "Dataset Overview" tab
2. Click "Load IoT Botnet Dataset" 
3. Verify dataset loading (expect ~1M samples with extreme class imbalance: 99.8% attacks, 0.2% normal)
4. The system automatically creates binary labels from category data

**Step 2: Machine Learning Training**
1. Go to "Machine Learning Model" tab
2. Configure feature engineering options:
   - Apply PCA Dimensionality Reduction (recommended: enabled)
   - Apply Standard Scaling (recommended: enabled)
   - Apply Feature Selection (recommended: enabled)
3. Select algorithm (Random Forest recommended)
4. Set test split percentage (20% recommended)
5. Click "Train ML Model"
6. Expected results: ~90-95% accuracy with balanced sampling

**Step 3: Deep Learning Training**
1. Go to "Deep Learning Model" tab
2. Configure network architecture:
   - Hidden Layers: 3-5 layers
   - Neurons per Layer: 128-256
   - Dropout Rate: 0.2-0.3
   - Activation: ReLU
3. Set training parameters:
   - Epochs: 50-100
   - Batch Size: 128
   - Learning Rate: 0.001
4. Click "Train DL Model"
5. Expected results: ~85-90% accuracy with same balanced sampling

**Step 4: Model Comparison and Testing**
1. Navigate to "Model Comparison & Testing" tab
2. Review training performance comparison
3. Select a test dataset from the dropdown
4. Click "Load Test Dataset"
5. Click "TEST BOTNET DETECTION"
6. Review botnet detection results and model performance comparison

## Expected Results

### Training Performance
- **ML Model**: Typically achieves 90-95% accuracy on balanced training data
- **DL Model**: Typically achieves 85-90% accuracy on balanced training data
- **Winner**: Usually ML due to effective feature engineering on this type of tabular data

### Testing Performance (Generalization)
- **Performance Drop**: Both models show 3-8% performance decrease on unseen data (normal and healthy)
- **Cross-Attack Generalization**: Models trained on Reconnaissance attacks tested on DoS attacks
- **Realistic Results**: 80-90% accuracy range indicating good but not perfect generalization

### Key Insights
1. **Feature Engineering Advantage**: ML benefits significantly from domain-specific feature engineering
2. **Data Leakage Prevention**: Aggressive feature filtering is crucial for realistic performance
3. **Class Imbalance Handling**: Balanced sampling during training improves generalization
4. **Model Robustness**: Both approaches can generalize across different attack types

## Technical Implementation Details

### Data Preprocessing
- **Label Hiding**: Test data labels are removed before prediction to simulate real deployment
- **Feature Consistency**: Same preprocessing pipeline applied to training and testing data
- **Balanced Sampling**: Equal numbers of normal and attack samples for training (typically 2k each)

### Model Architecture
- **ML Pipeline**: Feature filtering → Scaling → Selection → PCA → Classification
- **DL Pipeline**: Feature filtering → MinMax scaling → Neural network → Classification

### Performance Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Botnet Detection Focus**: Results presented in terms of threat detection capability
- **Generalization Testing**: Models tested on completely different attack types than training

## Troubleshooting

### Common Issues
1. **File Path Errors**: Ensure dataset files are in correct directories with exact filenames
2. **Column Mismatch**: Verify test datasets have same 35-column structure as training data
3. **Memory Issues**: Large datasets (1M+ samples) are automatically sampled to 200k for efficiency
4. **Perfect Accuracy**: Indicates data leakage; check feature filtering implementation

### Data Format Requirements
- CSV files without headers
- 35 columns exactly
- Last two columns: category, subcategory
- Comma-separated values
- Categories: "Normal" for legitimate traffic, specific names for attacks

## Model Export and Deployment

Trained models are automatically stored in session state and can be exported for deployment. The system maintains complete preprocessing pipelines ensuring consistent feature engineering between training and production environments.

## Conclusion

This implementation provides a comprehensive comparison of ML and DL approaches for IoT botnet detection, demonstrating that traditional machine learning with proper feature engineering often outperforms deep learning on structured network traffic data, particularly when data leakage is properly prevented and realistic evaluation conditions are maintained.