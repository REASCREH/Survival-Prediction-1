import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import logging
from typing import Dict, List
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="HCT Survival Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .risk-moderate {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .feature-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def find_model_files():
    """Search for model files automatically"""
    search_patterns = [
        "*.pkl",
        "models/*.pkl",
        "XGB_model.pkl",
        "CatBoost_model.pkl",
        "./XGB_model.pkl", 
        "./CatBoost_model.pkl",
        "w2v_model.pkl",
        "word2vec_model.pkl"
    ]
    
    found_models = {}
    for pattern in search_patterns:
        files = glob.glob(pattern)
        for file in files:
            filename = file.lower()
            if "xgb" in filename and "xgb" not in found_models:
                found_models["xgb"] = file
            elif "catboost" in filename and "catboost" not in found_models:
                found_models["catboost"] = file
            elif "w2v" in filename and "w2v" not in found_models:
                found_models["w2v"] = file
            elif "word2vec" in filename and "w2v" not in found_models:
                found_models["w2v"] = file
            elif "xgb" not in found_models and ("xgb" in filename or "xgboost" in filename):
                found_models["xgb"] = file
            elif "catboost" not in found_models and "catboost" in filename:
                found_models["catboost"] = file
    
    return found_models

# Feature information - updated to match actual model features
FEATURE_INFO = {
    "dri_score": {
        "description": "Disease Risk Index - categorizes patient risk based on underlying disease",
        "type": "categorical",
        "options": ["Low", "Intermediate", "High", "N/A - non-malignant indication", "N/A - pediatric", "Unknown"],
        "importance": "High",
        "example": "High for aggressive cancers, Low for early-stage diseases"
    },
    "age_at_hct": {
        "description": "Patient age at transplantation in years",
        "type": "numerical", 
        "range": "0-80 years",
        "importance": "High",
        "example": "45.5 (middle-aged patient)"
    },
    "karnofsky_score": {
        "description": "Performance status score measuring patient's ability to perform ordinary tasks (0-100)",
        "type": "numerical",
        "range": "0-100 (higher is better)",
        "importance": "High",
        "example": "80 (patient can carry on normal activity with effort)"
    },
    "comorbidity_score": {
        "description": "Overall comorbidity burden - measures other health conditions",
        "type": "numerical", 
        "range": "0-10 (lower is better)",
        "importance": "High",
        "example": "2 (moderate comorbidity burden)"
    },
    "hla_match_total": {
        "description": "Total HLA match score between donor and recipient",
        "type": "numerical",
        "range": "0-20 (higher is better)", 
        "importance": "High",
        "example": "18 (good HLA match)"
    },
    "donor_age": {
        "description": "Age of the stem cell donor in years",
        "type": "numerical",
        "range": "0-80 years",
        "importance": "Medium",
        "example": "35 (young donor)"
    },
    "psych_disturb": {
        "description": "Presence of psychiatric disorders that may affect treatment adherence",
        "type": "categorical",
        "options": ["Yes", "No", "Unknown"],
        "importance": "Medium",
        "example": "No (no psychiatric issues)"
    },
    "diabetes": {
        "description": "Diabetes status - affects infection risk and healing", 
        "type": "categorical",
        "options": ["Yes", "No", "Unknown"],
        "importance": "Medium",
        "example": "No (no diabetes)"
    },
    "cardiac": {
        "description": "Cardiac disease presence - important for conditioning regimen tolerance",
        "type": "categorical",
        "options": ["Yes", "No", "Unknown"], 
        "importance": "Medium",
        "example": "No (no cardiac issues)"
    },
    "graft_type": {
        "description": "Type of stem cell graft used for transplantation",
        "type": "categorical",
        "options": ["Bone marrow", "Peripheral blood", "Cord blood", "Unknown"],
        "importance": "High",
        "example": "Peripheral blood (most common source)"
    },
    "tbi_status": {
        "description": "Total Body Irradiation status",
        "type": "categorical",
        "options": ["No TBI", "TBI +- Other, >cGy", "Unknown"],
        "importance": "Medium",
        "example": "No TBI (no radiation therapy)"
    }
}

@st.cache_resource
def load_models():
    """Load ML models with caching"""
    try:
        # Try to import required packages
        try:
            import catboost
            import xgboost
        except ImportError as e:
            st.error(f"❌ Missing required package: {e}")
            return None, None, None
        
        MODEL_PATHS = find_model_files()
        st.info(f"🔍 Found models: {MODEL_PATHS}")
        
        if not MODEL_PATHS:
            st.error("❌ No model files found. Please ensure XGB_model.pkl and CatBoost_model.pkl are in your project.")
            return None, None, None
        
        # Load models with error handling
        xgb_model = None
        catboost_model = None
        w2v_model = None
        
        if "xgb" in MODEL_PATHS:
            try:
                xgb_model = joblib.load(MODEL_PATHS["xgb"])
                st.success("✅ XGBoost model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading XGBoost model: {e}")
        
        if "catboost" in MODEL_PATHS:
            try:
                catboost_model = joblib.load(MODEL_PATHS["catboost"])
                st.success("✅ CatBoost model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading CatBoost model: {e}")
        
        if "w2v" in MODEL_PATHS:
            try:
                w2v_model = joblib.load(MODEL_PATHS["w2v"])
                st.success("✅ Word2Vec model loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Word2Vec model not found or error loading: {e}")
                st.info("Using default encoding for categorical variables")
        
        if xgb_model is None and catboost_model is None:
            st.error("❌ Failed to load both models. Please check your model files.")
            return None, None, None
            
        return xgb_model, catboost_model, w2v_model
        
    except Exception as e:
        st.error(f"❌ Unexpected error loading models: {e}")
        return None, None, None

def create_empty_feature_dataframe():
    """Create a DataFrame with all expected features set to 0"""
    # Basic numerical features
    basic_features = [
        'age_at_hct', 'karnofsky_score', 'comorbidity_score', 'hla_match_total', 
        'donor_age', 'year_hct', 'nan_value_each_row', 'age_group',
        'donor_ageage_at_hct', 'comorbidity_scorekarnofsky_score'
    ]
    
    # HLA features
    hla_features = [
        'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6', 'hla_high_res_6', 
        'hla_high_res_10', 'hla_match_dqb1_high', 'hla_nmdp_6', 'hla_match_c_low', 
        'hla_match_drb1_low', 'hla_match_dqb1_low', 'hla_match_a_high', 
        'hla_match_b_low', 'hla_match_a_low', 'hla_match_b_high', 'hla_low_res_8', 
        'hla_match_drb1_high', 'hla_low_res_10'
    ]
    
    # Derived HLA features
    derived_features = [
        'hla_high_res_sum', 'hla_high_res_avg', 'hla_high_low_diff', 'hla_high_low_ratio',
        'hla_match_count', 'hla_match_std', 'hla_high_res_log', 'hla_high_res_squared'
    ]
    
    # Word2Vec features for categorical variables (40 dimensions each)
    categorical_columns = ['dri_score', 'psych_disturb', 'diabetes', 'tbi_status',
                          'arrhythmia', 'graft_type', 'cardiac']
    
    w2v_features = []
    for col in categorical_columns:
        w2v_features.extend([f"{col}_w2v_{i}" for i in range(40)])
    
    # Combine all features
    expected_features = basic_features + hla_features + derived_features + w2v_features
    
    # Create DataFrame with all expected features set to 0
    feature_dict = {feature: 0.0 for feature in expected_features}
    return pd.DataFrame([feature_dict])

def get_w2v_embedding(word, model, vector_size=40):
    """Get Word2Vec embedding for a word"""
    if model is not None and word in model.wv:
        return model.wv[word]
    else:
        # Return zeros if model not available or word not in vocabulary
        return np.zeros(vector_size)

def preprocess_features(input_data: Dict, w2v_model=None) -> pd.DataFrame:
    """Preprocess and encode input features matching the original training pipeline"""
    
    # Start with empty feature DataFrame
    df = create_empty_feature_dataframe()
    
    # Fill in the basic numerical features
    df['age_at_hct'] = float(input_data['age_at_hct'])
    df['karnofsky_score'] = float(input_data['karnofsky_score'])
    df['comorbidity_score'] = float(input_data['comorbidity_score'])
    df['hla_match_total'] = float(input_data['hla_match_total'])
    df['donor_age'] = float(input_data['donor_age'])
    
    # Create derived features
    df['nan_value_each_row'] = 0.0
    df['age_group'] = float(int(input_data['age_at_hct'] // 10))
    df['donor_ageage_at_hct'] = float(input_data['donor_age'] - input_data['age_at_hct'])
    df['comorbidity_scorekarnofsky_score'] = float(input_data['comorbidity_score'] * input_data['karnofsky_score'])
    
    # Set default values for year
    df['year_hct'] = 2023.0
    
    # Set HLA features based on hla_match_total
    hla_total = float(input_data['hla_match_total'])
    
    # Basic HLA features
    df['hla_high_res_6'] = min(hla_total, 6.0)
    df['hla_high_res_8'] = min(hla_total, 8.0)
    df['hla_high_res_10'] = min(hla_total, 10.0)
    df['hla_low_res_6'] = min(hla_total, 6.0)
    df['hla_low_res_8'] = min(hla_total, 8.0)
    df['hla_low_res_10'] = min(hla_total, 10.0)
    df['hla_nmdp_6'] = min(hla_total, 6.0)
    
    # Individual HLA match features
    df['hla_match_a_high'] = 1.0 if hla_total >= 16 else 0.0
    df['hla_match_a_low'] = 1.0 if hla_total >= 14 else 0.0
    df['hla_match_b_high'] = 1.0 if hla_total >= 16 else 0.0
    df['hla_match_b_low'] = 1.0 if hla_total >= 14 else 0.0
    df['hla_match_c_high'] = 1.0 if hla_total >= 16 else 0.0
    df['hla_match_c_low'] = 1.0 if hla_total >= 14 else 0.0
    df['hla_match_drb1_high'] = 1.0 if hla_total >= 16 else 0.0
    df['hla_match_drb1_low'] = 1.0 if hla_total >= 14 else 0.0
    df['hla_match_dqb1_high'] = 1.0 if hla_total >= 16 else 0.0
    df['hla_match_dqb1_low'] = 1.0 if hla_total >= 14 else 0.0
    
    # Calculate derived HLA features
    df['hla_high_res_sum'] = df['hla_high_res_6'] + df['hla_high_res_8'] + df['hla_high_res_10']
    df['hla_high_res_avg'] = df['hla_high_res_sum'] / 3.0
    df['hla_high_low_diff'] = df['hla_high_res_sum'] - (df['hla_low_res_6'] + df['hla_low_res_8'] + df['hla_low_res_10'])
    
    low_res_avg = (df['hla_low_res_6'] + df['hla_low_res_8'] + df['hla_low_res_10']) / 3.0
    df['hla_high_low_ratio'] = df['hla_high_res_avg'] / (low_res_avg + 0.001)
    
    df['hla_match_count'] = hla_total
    df['hla_match_std'] = 1.0
    df['hla_high_res_log'] = np.log(df['hla_high_res_sum'] + 1.0)
    df['hla_high_res_squared'] = df['hla_high_res_sum'] ** 2
    
    # Handle categorical variables using Word2Vec encoding
    categorical_mappings = {
        'dri_score': input_data['dri_score'],
        'psych_disturb': input_data['psych_disturb'],
        'diabetes': input_data['diabetes'],
        'tbi_status': input_data['tbi_status'],
        'arrhythmia': input_data['arrhythmia'],
        'graft_type': input_data['graft_type'],
        'cardiac': input_data['cardiac']
    }
    
    for col_name, value in categorical_mappings.items():
        # Get Word2Vec embedding
        embedding = get_w2v_embedding(str(value), w2v_model, vector_size=40)
        
        # Set each dimension in the DataFrame
        for i in range(40):
            df[f'{col_name}_w2v_{i}'] = embedding[i]
    
    # Fill any remaining NaN values and ensure all are float
    df = df.fillna(0.0)
    
    # Convert all columns to float to avoid type issues
    for col in df.columns:
        df[col] = df[col].astype(float)
    
    logger.info(f"Processed data shape: {df.shape}")
    logger.info(f"Sample features: {dict(list(df.iloc[0].items())[:10])}")
    
    return df

def calculate_feature_contributions(patient_data: Dict, prediction: float) -> Dict[str, float]:
    """Calculate simplified feature contributions"""
    contributions = {}
    
    # Disease risk contribution
    dri_weights = {
        "Low": 0.1, 
        "Intermediate": 0.3, 
        "High": 0.6, 
        "N/A - non-malignant indication": 0.2, 
        "N/A - pediatric": 0.1,
        "Unknown": 0.3
    }
    contributions["disease_risk"] = dri_weights.get(patient_data["dri_score"], 0.3) * abs(prediction)
    
    # Age contribution (higher age = higher risk)
    age_contribution = min(patient_data["age_at_hct"] / 80 * 0.3, 0.3) * abs(prediction)
    contributions["age_factors"] = age_contribution
    
    # Performance status contribution (lower score = higher risk)
    karnofsky_contribution = ((100 - patient_data["karnofsky_score"]) / 100 * 0.2) * abs(prediction)
    contributions["performance_status"] = karnofsky_contribution
    
    # Comorbidity contribution
    comorbidity_contribution = min(patient_data["comorbidity_score"] / 10 * 0.2, 0.2) * abs(prediction)
    contributions["comorbidities"] = comorbidity_contribution
    
    # HLA match contribution (lower match = higher risk)
    hla_contribution = ((20 - patient_data["hla_match_total"]) / 20 * 0.15) * abs(prediction)
    contributions["hla_matching"] = hla_contribution
    
    # Other factors
    contributions["other_factors"] = max(0, abs(prediction) - sum(contributions.values()))
    
    # Normalize to 100%
    total = sum(contributions.values())
    if total > 0:
        contributions = {k: round(v/total * 100, 1) for k, v in contributions.items()}
    
    return contributions

def predict_survival_risk(patient_data: Dict, xgb_model, catboost_model, w2v_model) -> Dict:
    """Make prediction using ensemble model (XGBoost and CatBoost only)"""
    
    try:
        # Preprocess features
        processed_data = preprocess_features(patient_data, w2v_model)
        
        # Get individual model predictions
        predictions = []
        model_names = []
        
        if xgb_model is not None:
            xgb_pred = float(xgb_model.predict(processed_data)[0])
            predictions.append(xgb_pred)
            model_names.append("XGBoost")
        
        if catboost_model is not None:
            cat_pred = float(catboost_model.predict(processed_data)[0])
            predictions.append(cat_pred)
            model_names.append("CatBoost")
        
        if not predictions:
            st.error("No models available for prediction")
            return None
        
        # Ensemble prediction (average of available models)
        ensemble_pred = np.mean(predictions)
        
        # Create individual predictions dictionary
        individual_predictions = {}
        for i, name in enumerate(model_names):
            individual_predictions[name] = round(predictions[i], 4)
        
        # Interpret prediction based on Nelson-Aalen cumulative hazard
        if ensemble_pred < -0.8:
            risk_category = "Very Low Risk"
            confidence = "Very High"
            interpretation = "Excellent prognosis with very high survival probability. Patient has minimal cumulative hazard."
            risk_class = "risk-low"
            recommendations = [
                "Standard monitoring protocol",
                "Routine follow-up care",
                "Maintain current treatment plan"
            ]
        elif ensemble_pred < -0.3:
            risk_category = "Low Risk"
            confidence = "High"
            interpretation = "Good prognosis with favorable survival outcomes expected. Low cumulative hazard detected."
            risk_class = "risk-low"
            recommendations = [
                "Standard monitoring protocol",
                "Regular follow-up visits",
                "Maintain preventive care"
            ]
        elif ensemble_pred < 0:
            risk_category = "Moderate Risk"
            confidence = "Medium"
            interpretation = "Moderate prognosis requiring standard monitoring. Some cumulative hazard present."
            risk_class = "risk-moderate"
            recommendations = [
                "Enhanced monitoring",
                "Close follow-up schedule",
                "Consider supportive care interventions"
            ]
        elif ensemble_pred < 0.3:
            risk_category = "High Risk"
            confidence = "High"
            interpretation = "Higher risk profile with increased cumulative hazard. Requires close management."
            risk_class = "risk-high"
            recommendations = [
                "Intensive monitoring required",
                "Frequent follow-up visits",
                "Consider additional supportive therapies",
                "Multidisciplinary team review recommended"
            ]
        else:
            risk_category = "Very High Risk"
            confidence = "Very High"
            interpretation = "Very high cumulative hazard detected. Requires immediate and intensive intervention."
            risk_class = "risk-high"
            recommendations = [
                "Immediate specialist consultation",
                "Intensive monitoring protocol",
                "Aggressive supportive care",
                "Consider treatment plan adjustment",
                "Multidisciplinary team management essential"
            ]
        
        # Calculate feature contributions
        feature_contributions = calculate_feature_contributions(patient_data, ensemble_pred)
        
        # Calculate model agreement
        if len(predictions) > 1:
            model_agreement = "High" if abs(predictions[0] - predictions[1]) < 0.1 else "Medium"
        else:
            model_agreement = "Single Model"
        
        return {
            "prediction": round(ensemble_pred, 4),
            "risk_category": risk_category,
            "confidence": confidence,
            "interpretation": interpretation,
            "feature_contributions": feature_contributions,
            "individual_predictions": individual_predictions,
            "model_agreement": model_agreement,
            "recommendations": recommendations,
            "risk_class": risk_class,
            "explanation": f"The prediction score of {ensemble_pred:.4f} represents the Nelson-Aalen cumulative hazard estimate. " + 
                          ("Negative values indicate lower cumulative hazard and better survival probability." if ensemble_pred < 0 
                           else "Positive values indicate higher cumulative hazard and increased risk.")
        }
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        # Show debug information
        with st.expander("🔍 Debug Information"):
            st.write("Error details:", str(e))
            try:
                processed_data = preprocess_features(patient_data, w2v_model)
                st.write("Processed data shape:", processed_data.shape)
                st.write("Processed data columns sample:", list(processed_data.columns)[:20])
                st.write("Processed data sample values:", processed_data.iloc[0].head(10).to_dict())
            except Exception as debug_e:
                st.write("Debug error:", debug_e)
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 HCT Survival Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("Predict survival outcomes for Hematopoietic Cell Transplantation patients using machine learning")
    
    # Load models
    xgb_model, catboost_model, w2v_model = load_models()
    
    # Check if at least one model is loaded
    if xgb_model is None and catboost_model is None:
        st.error("""
        ❌ Unable to load models. Please ensure:
        1. Model files (XGB_model.pkl and CatBoost_model.pkl) are uploaded to your Streamlit project
        2. All required packages are installed (check requirements.txt)
        3. Model files are compatible with the current environment
        """)
        return
    
    # Show loaded models status
    loaded_models = []
    if xgb_model is not None:
        loaded_models.append("XGBoost")
    if catboost_model is not None:
        loaded_models.append("CatBoost")
    if w2v_model is not None:
        loaded_models.append("Word2Vec")
    
    st.success(f"✅ Ready for predictions using: {', '.join(loaded_models)}")
    
    # Sidebar for feature information
    with st.sidebar:
        st.header("📊 Feature Information")
        selected_feature = st.selectbox("Select feature to learn more:", list(FEATURE_INFO.keys()))
        
        if selected_feature:
            info = FEATURE_INFO[selected_feature]
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Type:** {info['type']}")
            st.markdown(f"**Importance:** {info['importance']}")
            st.markdown(f"**Example:** {info['example']}")
            if 'options' in info:
                st.markdown(f"**Options:** {', '.join(info['options'])}")
            if 'range' in info:
                st.markdown(f"**Range:** {info['range']}")
        
        st.markdown("---")
        st.markdown("### 🎯 About")
        st.markdown("""
        This system predicts survival outcomes using:
        - **XGBoost** and **CatBoost** ensemble
        - **Word2Vec** encoding for categorical variables
        - Nelson-Aalen cumulative hazard estimation
        - Clinical feature analysis
        """)
    
    # Main form
    with st.form("prediction_form"):
        st.header("📝 Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dri_score = st.selectbox(
                "Disease Risk Index (DRI Score)",
                options=FEATURE_INFO["dri_score"]["options"],
                help=FEATURE_INFO["dri_score"]["description"]
            )
            
            age_at_hct = st.slider(
                "Age at Transplantation (years)",
                min_value=0.0,
                max_value=80.0,
                value=45.0,
                step=1.0,
                help=FEATURE_INFO["age_at_hct"]["description"]
            )
            
            karnofsky_score = st.slider(
                "Karnofsky Performance Score",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=5.0,
                help=FEATURE_INFO["karnofsky_score"]["description"]
            )
            
            comorbidity_score = st.slider(
                "Comorbidity Score",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                help=FEATURE_INFO["comorbidity_score"]["description"]
            )
            
            hla_match_total = st.slider(
                "HLA Match Total",
                min_value=0.0,
                max_value=20.0,
                value=18.0,
                step=1.0,
                help=FEATURE_INFO["hla_match_total"]["description"]
            )
        
        with col2:
            donor_age = st.slider(
                "Donor Age (years)",
                min_value=0.0,
                max_value=80.0,
                value=35.0,
                step=1.0,
                help=FEATURE_INFO["donor_age"]["description"]
            )
            
            psych_disturb = st.radio(
                "Psychiatric Disturbance",
                options=["Yes", "No", "Unknown"],
                help=FEATURE_INFO["psych_disturb"]["description"]
            )
            
            diabetes = st.radio(
                "Diabetes",
                options=["Yes", "No", "Unknown"],
                help=FEATURE_INFO["diabetes"]["description"]
            )
            
            cardiac = st.radio(
                "Cardiac Disease",
                options=["Yes", "No", "Unknown"],
                help=FEATURE_INFO["cardiac"]["description"]
            )
            
            graft_type = st.selectbox(
                "Graft Type",
                options=FEATURE_INFO["graft_type"]["options"],
                help=FEATURE_INFO["graft_type"]["description"]
            )
            
            tbi_status = st.selectbox(
                "TBI Status",
                options=FEATURE_INFO["tbi_status"]["options"],
                help=FEATURE_INFO["tbi_status"]["description"]
            )
            
            arrhythmia = st.radio(
                "Arrhythmia",
                options=["Yes", "No", "Unknown"],
                help="Presence of heart rhythm disorders"
            )
        
        # Submit button
        submitted = st.form_submit_button("🎯 Predict Survival Risk", use_container_width=True)
    
    # Handle prediction
    if submitted:
        with st.spinner("🔄 Analyzing patient data and making prediction..."):
            # Prepare input data
            patient_data = {
                "dri_score": dri_score,
                "age_at_hct": age_at_hct,
                "karnofsky_score": karnofsky_score,
                "comorbidity_score": comorbidity_score,
                "hla_match_total": hla_match_total,
                "donor_age": donor_age,
                "psych_disturb": psych_disturb,
                "diabetes": diabetes,
                "cardiac": cardiac,
                "graft_type": graft_type,
                "tbi_status": tbi_status,
                "arrhythmia": arrhythmia
            }
            
            # Make prediction
            result = predict_survival_risk(patient_data, xgb_model, catboost_model, w2v_model)
            
            if result:
                # Display results
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                # Risk card
                risk_class = result["risk_class"]
                st.markdown(f"""
                <div class="prediction-card {risk_class}">
                    <h2>Risk Category: {result['risk_category']}</h2>
                    <h3>Prediction Score: {result['prediction']}</h3>
                    <p><strong>Confidence:</strong> {result['confidence']}</p>
                    <p><strong>Interpretation:</strong> {result['interpretation']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Model details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🤖 Model Predictions")
                    for model_name, prediction in result["individual_predictions"].items():
                        st.metric(model_name, prediction)
                    st.metric("Model Agreement", result["model_agreement"])
                
                with col2:
                    st.subheader("📈 Feature Contributions")
                    contributions = result["feature_contributions"]
                    for feature, contribution in contributions.items():
                        st.metric(feature.replace('_', ' ').title(), f"{contribution}%")
                
                # Recommendations
                st.subheader("💡 Clinical Recommendations")
                for i, recommendation in enumerate(result["recommendations"], 1):
                    st.markdown(f"{i}. {recommendation}")
                
                # Explanation
                with st.expander("🔍 Technical Explanation"):
                    st.info(result["explanation"])
                
                # Patient summary
                with st.expander("📋 Patient Summary"):
                    st.json(patient_data)
            else:
                st.error("❌ Prediction failed. Please check the input data and try again.")

if __name__ == "__main__":
    main()
