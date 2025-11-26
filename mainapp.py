import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import logging
from typing import Dict, List
import os

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
        "C:/Users/Qamar/Downloads/*.pkl",
        "C:/Users/Qamar/Desktop/*.pkl", 
        "./*.pkl",
        "./models/*.pkl"
    ]
    
    found_models = {}
    for pattern in search_patterns:
        files = glob.glob(pattern)
        for file in files:
            if "XGB" in file or "xgb" in file.lower():
                found_models["xgb"] = file
            elif "CatBoost" in file or "catboost" in file.lower():
                found_models["catboost"] = file
    
    return found_models

# Feature information
FEATURE_INFO = {
    "dri_score": {
        "description": "Disease Risk Index - categorizes patient risk based on underlying disease",
        "type": "categorical",
        "options": ["Low", "Intermediate", "High", "N/A - non-malignant indication", "N/A - pediatric"],
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
        "options": ["Yes", "No"],
        "importance": "Medium",
        "example": "No (no psychiatric issues)"
    },
    "diabetes": {
        "description": "Diabetes status - affects infection risk and healing", 
        "type": "categorical",
        "options": ["Yes", "No"],
        "importance": "Medium",
        "example": "No (no diabetes)"
    },
    "cardiac": {
        "description": "Cardiac disease presence - important for conditioning regimen tolerance",
        "type": "categorical",
        "options": ["Yes", "No"], 
        "importance": "Medium",
        "example": "No (no cardiac issues)"
    },
    "graft_type": {
        "description": "Type of stem cell graft used for transplantation",
        "type": "categorical",
        "options": ["Bone marrow", "Peripheral blood", "Cord blood"],
        "importance": "High",
        "example": "Peripheral blood (most common source)"
    }
}

@st.cache_resource
def load_models():
    """Load ML models with caching"""
    try:
        MODEL_PATHS = find_model_files()
        st.info(f"Found models: {MODEL_PATHS}")
        
        xgb_model = joblib.load(MODEL_PATHS["xgb"])
        catboost_model = joblib.load(MODEL_PATHS["catboost"])
        
        st.success("✅ Models loaded successfully!")
        return xgb_model, catboost_model
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None

def preprocess_features(input_data: Dict) -> pd.DataFrame:
    """Preprocess and encode input features matching training pipeline"""
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Feature engineering
    df['nan_value_each_row'] = 0
    df['age_group'] = (df['age_at_hct'] // 10).astype(int)
    df['donor_age-age_at_hct'] = df['donor_age'] - df['age_at_hct']
    df['comorbidity_score+karnofsky_score'] = df['comorbidity_score'] + df['karnofsky_score']
    df['comorbidity_score-karnofsky_score'] = df['comorbidity_score'] - df['karnofsky_score']
    df['comorbidity_score*karnofsky_score'] = df['comorbidity_score'] * df['karnofsky_score']
    df['comorbidity_score/karnofsky_score'] = df['comorbidity_score'] / (df['karnofsky_score'] + 0.001)
    
    # Handle categorical encoding
    categorical_mappings = {
        'dri_score': ['Low', 'Intermediate', 'High', 'N/A_-_non-malignant_indication', 'N/A_-_pediatric'],
        'psych_disturb': ['Yes', 'No'],
        'diabetes': ['Yes', 'No'],
        'cardiac': ['Yes', 'No'],
        'graft_type': ['Bone_marrow', 'Peripheral_blood', 'Cord_blood']
    }
    
    for col, options in categorical_mappings.items():
        for option in options:
            clean_option = option.replace(' ', '_').replace('/', '_').replace('-', '_')
            df[f'{col}_{clean_option}'] = (df[col] == option.replace('_', ' ')).astype(int)
    
    # Drop original categorical columns
    df = df.drop(columns=list(categorical_mappings.keys()), errors='ignore')
    
    # Add expected numerical features with default values
    expected_numerical = [
        'age_at_hct', 'karnofsky_score', 'comorbidity_score', 'hla_match_total', 
        'donor_age', 'nan_value_each_row', 'age_group', 'donor_age-age_at_hct',
        'comorbidity_score+karnofsky_score', 'comorbidity_score-karnofsky_score',
        'comorbidity_score*karnofsky_score', 'comorbidity_score/karnofsky_score'
    ]
    
    for col in expected_numerical:
        if col not in df.columns:
            df[col] = 0
    
    # Fill any remaining NaN values
    df = df.fillna(0)
    
    return df

def calculate_feature_contributions(patient_data: Dict, prediction: float) -> Dict[str, float]:
    """Calculate simplified feature contributions"""
    contributions = {}
    
    # Disease risk contribution
    dri_weights = {"Low": 0.1, "Intermediate": 0.3, "High": 0.6}
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

def predict_survival_risk(patient_data: Dict, xgb_model, catboost_model) -> Dict:
    """Make prediction using ensemble model (XGBoost and CatBoost only)"""
    
    try:
        # Preprocess features
        processed_data = preprocess_features(patient_data)
        
        # Get individual model predictions
        xgb_pred = float(xgb_model.predict(processed_data)[0])
        cat_pred = float(catboost_model.predict(processed_data)[0])
        
        # Ensemble prediction (average of two models)
        ensemble_pred = np.mean([xgb_pred, cat_pred])
        
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
        
        return {
            "prediction": round(ensemble_pred, 4),
            "risk_category": risk_category,
            "confidence": confidence,
            "interpretation": interpretation,
            "feature_contributions": feature_contributions,
            "individual_predictions": {
                "XGBoost": round(xgb_pred, 4),
                "CatBoost": round(cat_pred, 4)
            },
            "model_agreement": "High" if abs(xgb_pred - cat_pred) < 0.1 else "Medium",
            "recommendations": recommendations,
            "risk_class": risk_class,
            "explanation": f"The prediction score of {ensemble_pred:.4f} represents the Nelson-Aalen cumulative hazard estimate. " + 
                          ("Negative values indicate lower cumulative hazard and better survival probability." if ensemble_pred < 0 
                           else "Positive values indicate higher cumulative hazard and increased risk.")
        }
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 HCT Survival Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("Predict survival outcomes for Hematopoietic Cell Transplantation patients using machine learning")
    
    # Load models
    xgb_model, catboost_model = load_models()
    
    if xgb_model is None or catboost_model is None:
        st.error("❌ Unable to load models. Please check if model files are available.")
        return
    
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
                options=["Yes", "No"],
                help=FEATURE_INFO["psych_disturb"]["description"]
            )
            
            diabetes = st.radio(
                "Diabetes",
                options=["Yes", "No"],
                help=FEATURE_INFO["diabetes"]["description"]
            )
            
            cardiac = st.radio(
                "Cardiac Disease",
                options=["Yes", "No"],
                help=FEATURE_INFO["cardiac"]["description"]
            )
            
            graft_type = st.selectbox(
                "Graft Type",
                options=FEATURE_INFO["graft_type"]["options"],
                help=FEATURE_INFO["graft_type"]["description"]
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
                "graft_type": graft_type
            }
            
            # Make prediction
            result = predict_survival_risk(patient_data, xgb_model, catboost_model)
            
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
                    st.metric("XGBoost", result["individual_predictions"]["XGBoost"])
                    st.metric("CatBoost", result["individual_predictions"]["CatBoost"])
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

if __name__ == "__main__":
    main()
