
# IoT Botnet Detection System

## 🎯 Project Overview

This project compares **Machine Learning (ML)** and **Deep Learning (DL)** approaches for detecting IoT botnet behavior in network traffic. The goal is to determine which method performs better for cybersecurity threat detection in IoT environments.

## 🔬 Research Question

**Does deep learning actually outperform traditional machine learning for IoT botnet detection, given their different architectural approaches?**

## 📊 Dataset

- **Primary Dataset**: [Bot IoT Dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
- **Data Type**: Network traffic features from IoT devices
- **Task**: Binary classification (Botnet vs Normal traffic)

## 🛠️ Methodology

### **Machine Learning Approach**
- **Focus**: Heavy feature engineering and traditional algorithms
- **Techniques**: 
  - PCA dimensionality reduction
  - Standard scaling and normalization
  - Feature selection methods
  - Mathematical transforms
- **Algorithms**: Random Forest, Gradient Boosting, SVM, Logistic Regression

### **Deep Learning Approach**
- **Focus**: Neural networks with raw data processing
- **Architecture**: Multi-layer feedforward neural network
- **Features**: 
  - Raw data input (minimal preprocessing)
  - Automatic feature learning
  - Dropout regularization
  - Optimized hyperparameters

## 🎯 Evaluation Strategy

1. **Training Performance**: Compare models on validation data
2. **Generalization**: Test both models on completely unseen data
3. **Metrics**: Accuracy, Precision, Recall, F1-Score
4. **Final Assessment**: 10% of grade based on best model's performance on unseen data

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Launch Application
```bash
streamlit run app.py
```

### Usage Flow
1. **📊 Data Tab**: Upload your CSV dataset
2. **🔧 ML Model Tab**: Configure and train traditional ML model
3. **🧠 DL Model Tab**: Configure and train neural network
4. **⚖️ Compare & Test Tab**: Compare performance and test on new data

## 📁 Project Structure

```
├── app.py              # Main Streamlit application
├── styles.css          # Custom CSS styling
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🎨 Features

- **Interactive Web Interface**: Clean, professional Streamlit app
- **Real-time Training**: Watch models train with progress indicators
- **Performance Comparison**: Side-by-side metric comparisons
- **Model Testing**: Upload new data to test generalization
- **Automatic Export**: Trained models saved as `.pkl` and `.h5` files

## 📈 Expected Outcomes

- **Performance Comparison**: Quantitative analysis of ML vs DL effectiveness
- **Insights**: Understanding which approach works better for IoT security
- **Model Artifacts**: Exportable trained models for deployment
- **Documentation**: Complete analysis workflow and results

## 🔧 Technical Implementation

- **Frontend**: Streamlit web application
- **ML Libraries**: Scikit-learn, Pandas, NumPy
- **DL Framework**: TensorFlow/Keras
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas for data manipulation

## 📋 Assignment Requirements Met

✅ **Analytic Development Process**: Structured ML and DL workflows  
✅ **Feature Engineering**: PCA, scaling, transforms for ML model  
✅ **Raw Data Processing**: Direct neural network input for DL model  
✅ **Model Optimization**: Hyperparameter tuning for both approaches  
✅ **Performance Comparison**: Comprehensive metric evaluation  
✅ **Unseen Data Testing**: Model generalization assessment  
✅ **Model Export**: Automatic saving of trained models  
✅ **Interactive Demo**: Video-ready Streamlit interface  

## 🎥 Demo

The application includes a complete workflow demonstration showing:
- Data upload and preprocessing
- Model training processes
- Performance comparisons
- Testing on unseen data
- Final results and model selection

---

*This project demonstrates the practical application of both traditional machine learning and modern deep learning techniques for cybersecurity in IoT environments.*