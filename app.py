"""
Streamlit web interface for malicious URL detection.

This application allows users to input URLs and get predictions using
pre-trained machine learning models. All analysis is performed offline
without opening or visiting URLs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from features import extract_features
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Malicious URL Detector",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .benign {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .malicious {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Load models function
@st.cache_resource
def load_models():
    """Load trained models and scalers."""
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        st.error(f"Models directory '{models_dir}' not found. Please train models first.")
        return None
    
    try:
        models = {
            'random_forest': joblib.load(os.path.join(models_dir, 'random_forest.pkl')),
            'logistic_regression': joblib.load(os.path.join(models_dir, 'logistic_regression.pkl')),
            'svm': joblib.load(os.path.join(models_dir, 'svm.pkl'))
        }
        
        scalers = {
            'lr_scaler': joblib.load(os.path.join(models_dir, 'lr_scaler.pkl')),
            'svm_scaler': joblib.load(os.path.join(models_dir, 'svm_scaler.pkl'))
        }
        
        feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
        
        return models, scalers, feature_names
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None


def predict_url(url, model, scaler=None, model_name='Model'):
    """
    Predict if a URL is malicious.
    
    Args:
        url (str): URL to classify
        model: Trained sklearn model
        scaler: Optional scaler for preprocessing
        model_name (str): Name of the model
        
    Returns:
        dict: Prediction results
    """
    # Extract features
    features_dict = extract_features(url)
    feature_values = np.array([list(features_dict.values())])
    
    # Preprocess if scaler provided
    if scaler is not None:
        feature_values = scaler.transform(feature_values)
    
    # Predict
    prediction = model.predict(feature_values)[0]
    
    # Get probabilities
    try:
        probabilities = model.predict_proba(feature_values)[0]
        confidence = probabilities[prediction]
    except:
        confidence = None
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': probabilities if 'probabilities' in locals() else None,
        'features': features_dict
    }


def get_feature_importance_explanation(features_dict, model, feature_names):
    """
    Get explanation based on feature importance.
    
    Args:
        features_dict (dict): Extracted features
        features_dict (dict): Extracted features
        model: Trained model (Random Forest for feature importance)
        feature_names (list): List of feature names
        
    Returns:
        list: List of important features contributing to prediction
    """
    explanations = []
    
    # Use Random Forest feature importance if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_pairs = list(zip(feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 5 most important features
        top_features = feature_importance_pairs[:5]
        
        for feat_name, importance in top_features:
            feat_value = features_dict.get(feat_name, 0)
            explanations.append({
                'feature': feat_name,
                'value': feat_value,
                'importance': importance
            })
    
    return explanations


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<div class="main-header">🔒 Malicious URL Detector</div>', 
                unsafe_allow_html=True)
    
    # Warning message
    st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Important:</strong> This tool analyzes URLs as plain text only. 
            URLs are <strong>never opened, visited, or executed</strong>. All analysis is 
            performed offline using lexical and structural features extracted from the URL string.
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    model_data = load_models()
    
    if model_data is None:
        st.stop()
    
    models, scalers, feature_names = model_data
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Choose a model:",
        ["Random Forest", "Logistic Regression", "SVM"],
        index=0
    )
    
    model_map = {
        "Random Forest": ("random_forest", None),
        "Logistic Regression": ("logistic_regression", "lr_scaler"),
        "SVM": ("svm", "svm_scaler")
    }
    
    model_key, scaler_key = model_map[selected_model_name]
    selected_model = models[model_key]
    selected_scaler = scalers[scaler_key] if scaler_key else None
    
    # Main input area
    st.header("URL Classification")
    
    url_input = st.text_input(
        "Enter a URL to analyze:",
        placeholder="https://example.com/path/to/page",
        help="Enter the full URL including protocol (http:// or https://)"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        classify_button = st.button("🔍 Classify URL", type="primary", use_container_width=True)
    
    # Process URL if button clicked
    if classify_button:
        if not url_input or url_input.strip() == "":
            st.warning("Please enter a URL to classify.")
        else:
            with st.spinner("Analyzing URL..."):
                # Get prediction
                result = predict_url(
                    url_input, 
                    selected_model, 
                    selected_scaler, 
                    selected_model_name
                )
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                prediction_label = "Malicious" if result['prediction'] == 1 else "Benign"
                confidence = result['confidence']
                
                # Prediction box
                box_class = "malicious" if result['prediction'] == 1 else "benign"
                st.markdown(
                    f"""
                    <div class="prediction-box {box_class}">
                        <h3>Classification: <strong>{prediction_label}</strong></h3>
                        {f'<p>Confidence: <strong>{confidence:.2%}</strong></p>' if confidence else ''}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Probability breakdown
                if result['probabilities'] is not None:
                    st.subheader("Probability Breakdown")
                    prob_df = pd.DataFrame({
                        'Class': ['Benign', 'Malicious'],
                        'Probability': [
                            result['probabilities'][0],
                            result['probabilities'][1]
                        ]
                    })
                    st.bar_chart(prob_df.set_index('Class'))
                
                # Feature explanation
                st.subheader("Key Features")
                
                # Get feature importance explanation
                rf_model = models['random_forest']  # Use RF for feature importance
                explanations = get_feature_importance_explanation(
                    result['features'], rf_model, feature_names
                )
                
                if explanations:
                    exp_df = pd.DataFrame(explanations)
                    exp_df['importance'] = exp_df['importance'].apply(lambda x: f"{x:.4f}")
                    exp_df.columns = ['Feature', 'Value', 'Importance']
                    st.dataframe(exp_df, use_container_width=True, hide_index=True)
                
                # Detailed features
                with st.expander("View All Extracted Features"):
                    features_df = pd.DataFrame([result['features']]).T
                    features_df.columns = ['Value']
                    features_df = features_df.sort_index()
                    st.dataframe(features_df, use_container_width=True)
    
    # Information section
    st.markdown("---")
    st.header("About")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔍 How It Works
        The system analyzes URLs using lexical and structural features extracted 
        from the URL string itself, including length, character patterns, 
        subdomain count, and more.
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Machine Learning
        Three models are available:
        - **Random Forest**: Ensemble method with feature importance
        - **Logistic Regression**: Linear classifier
        - **SVM**: Support Vector Machine with linear kernel
        """)
    
    with col3:
        st.markdown("""
        ### ⚡ Features Analyzed
        - URL length and structure
        - Character patterns (digits, special chars)
        - Subdomain count
        - IP address detection
        - Entropy calculation
        - Suspicious token detection
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "Malicious URL Detection System | Academic Research Tool"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
