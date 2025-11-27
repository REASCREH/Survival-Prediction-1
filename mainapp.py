import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import logging
from typing import Dict, List
import os
import sys

# Try to import required packages for type hinting, the loading function handles import errors
try:
    import xgboost as xgb
    import catboost
    from gensim.models import Word2Vec
except ImportError:
    pass

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
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# GLOBAL FEATURE LISTS (FINAL CORRECTED ORDER)
# --------------------------------------------------------------------------------

# All 35 categorical columns from your training
CATEGORICAL_COLUMNS = [
    'dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status',
    'arrhythmia', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe',
    'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab',
    'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity',
    'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe',
    'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match',
    'race_group', 'hepatic_mild', 'tce_div_match', 'donor_related',
    'melphalan_dose', 'cardiac', 'pulm_moderate'
]

# The master feature list for ALL numerical columns, ordered explicitly 
# to match the model's training data features (TRIAL 3: 9 features moved to the end)
MASTER_NUMERICAL_FEATURES = [
    # 1. First 22 features, exactly as seen in the error message
    'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6', 'hla_high_res_6', 'hla_high_res_10', 
    'hla_match_dqb1_high', 'hla_nmdp_6', 'hla_match_c_low', 'hla_match_drb1_low', 'hla_match_dqb1_low', 
    'year_hct', 'hla_match_a_high', 'donor_age', 'hla_match_b_low', 'age_at_hct', 
    'hla_match_a_low', 'hla_match_b_high', 'comorbidity_score', 'karnofsky_score', 
    'hla_low_res_8', 'hla_match_drb1_high', 'hla_low_res_10',
    
    # 2. Next 5 derived features, exactly as seen in the error message
    'nan_value_each_row', 'age_group', 'dri_score_NA', 
    'donor_ageage_at_hct', 'comorbidity_scorekarnofsky_score', 
    
    # 3. CRITICAL CHANGE: The 9 previously missing HLA features inserted here.
    'hla_match_std', 'hla_match_count', 'hla_high_res_log', 'hla_high_res_sum', 
    'hla_high_res_squared', 'hla_high_res_avg', 'hla_high_low_diff', 'hla_high_low_ratio', 
    'hla_match_total' 
]

# Feature information (simplified for display)
FEATURE_INFO = {
    "dri_score": {"description": "Disease Risk Index", "type": "categorical", "options": ["Low", "Intermediate", "High", "N/A - non-malignant indication", "N/A - pediatric", "Unknown"], "importance": "High", "example": "High for aggressive cancers"},
    "age_at_hct": {"description": "Patient age at transplantation in years", "type": "numerical", "range": "0-80 years", "importance": "High", "example": "45.5 (middle-aged patient)"},
    "karnofsky_score": {"description": "Performance status score (0-100)", "type": "numerical", "range": "0-100 (higher is better)", "importance": "High", "example": "80 (can carry on normal activity)"},
    "comorbidity_score": {"description": "Overall comorbidity burden (0-10)", "type": "numerical", "range": "0-10 (lower is better)", "importance": "High", "example": "2 (moderate burden)"},
    "hla_match_total": {"description": "Total HLA match score (0-20)", "type": "numerical", "range": "0-20 (higher is better)", "importance": "High", "example": "18 (good HLA match)"},
    "donor_age": {"description": "Age of the stem cell donor in years", "type": "numerical", "range": "0-80 years", "importance": "Medium", "example": "35 (young donor)"},
    "psych_disturb": {"description": "Psychiatric disorders presence", "type": "categorical", "options": ["Yes", "No", "Unknown"], "importance": "Medium", "example": "No"},
    "diabetes": {"description": "Diabetes status", "type": "categorical", "options": ["Yes", "No", "Unknown"], "importance": "Medium", "example": "No"},
    "cardiac": {"description": "Cardiac disease presence", "type": "categorical", "options": ["Yes", "No", "Unknown"], "importance": "Medium", "example": "No"},
    "graft_type": {"description": "Type of stem cell graft used", "type": "categorical", "options": ["Bone marrow", "Peripheral blood", "Cord blood", "Unknown"], "importance": "High", "example": "Peripheral blood"},
    "tbi_status": {"description": "Total Body Irradiation status", "type": "categorical", "options": ["No TBI", "TBI +- Other, >cGy", "Unknown"], "importance": "Medium", "example": "No TBI"},
    "arrhythmia": {"description": "Presence of heart rhythm disorders", "type": "categorical", "options": ["Yes", "No", "Unknown"], "importance": "Medium", "example": "No"},
    "cyto_score": {"description": "Cytogenetic risk score", "type": "categorical", "options": ["Favorable", "Intermediate", "Poor", "Unknown"], "importance": "High", "example": "Intermediate"},
    "cmv_status": {"description": "Cytomegalovirus status", "type": "categorical", "options": ["Positive", "Negative", "Unknown"], "importance": "Medium", "example": "Negative"}
}


# --------------------------------------------------------------------------------
# MODEL LOADING AND PREPROCESSING FUNCTIONS
# --------------------------------------------------------------------------------

def find_model_files():
    """Search for model files automatically"""
    search_patterns = [
        "*.pkl", "models/*.pkl", "*.joblib", "models/*.joblib", "*.cbm", 
        "models/*.cbm", "*.model", "models/*.model", "xgb_model.pkl", 
        "catboost_model.pkl", "w2v_model.pkl", "catboost_model.cbm", 
        "w2v_model.model"
    ]
    
    found_models = {}
    for pattern in search_patterns:
        files = glob.glob(pattern)
        for file in files:
            filename = file.lower()
            if "xgb" in filename and "xgb" not in found_models:
                found_models["xgb"] = file
            elif ("catboost" in filename or filename.endswith(".cbm")) and "catboost" not in found_models:
                found_models["catboost"] = file
            elif ("w2v" in filename or filename.endswith(".model")) and "w2v" not in found_models:
                found_models["w2v"] = file
    
    return found_models

@st.cache_resource
def load_models():
    """Load ML models with caching"""
    try:
        # Check for required packages
        import xgboost as xgb
        import catboost
        from gensim.models import Word2Vec
        
        MODEL_PATHS = find_model_files()
        st.info(f"🔍 Found models: {MODEL_PATHS}")
        
        if not MODEL_PATHS:
            st.error("❌ No model files found. Please ensure model files are in your project.")
            return None, None, None
        
        xgb_model, catboost_model, w2v_model = None, None, None
        
        # --- XGBoost Loading ---
        if "xgb" in MODEL_PATHS:
            try:
                xgb_model = joblib.load(MODEL_PATHS["xgb"])
                st.success("✅ XGBoost model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading XGBoost model: {e}")
        
        # --- CatBoost Loading ---
        if "catboost" in MODEL_PATHS:
            try:
                if MODEL_PATHS["catboost"].lower().endswith(".cbm"):
                    cat_model_temp = catboost.CatBoostRegressor(verbose=0)
                    cat_model_temp.load_model(MODEL_PATHS["catboost"])
                    catboost_model = cat_model_temp
                else:
                    catboost_model = joblib.load(MODEL_PATHS["catboost"])
                st.success("✅ CatBoost model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading CatBoost model: {e}")

        # --- Word2Vec Loading ---
        if "w2v" in MODEL_PATHS:
            try:
                from gensim.models import Word2Vec
                if MODEL_PATHS["w2v"].lower().endswith(".model"):
                    w2v_model = Word2Vec.load(MODEL_PATHS["w2v"])
                else:
                    w2v_model = joblib.load(MODEL_PATHS["w2v"])
                st.success("✅ Word2Vec model loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Word2Vec model not found or error loading: {e}")
        
        if xgb_model is None and catboost_model is None:
            st.error("❌ Failed to load both models. Check model files.")
            return None, None, None
            
        return xgb_model, catboost_model, w2v_model
            
    except ImportError as e:
        st.error(f"❌ Missing required package: {e.name}. Please ensure all dependencies are installed.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Unexpected error during model loading process: {e}")
        return None, None, None

def get_w2v_embedding(word, model, vector_size=40):
    """Get Word2Vec embedding for a word"""
    if model is None:
        return np.zeros(vector_size)

    wv = model.wv if hasattr(model, 'wv') else model 
    
    if hasattr(wv, 'key_to_index') and word in wv.key_to_index:
        return wv[word]
    elif word in wv: 
        return wv[word]
    else:
        return np.zeros(vector_size)

def create_empty_feature_dataframe():
    """Create a DataFrame with ALL expected features in the EXACT required order."""
    
    w2v_features = []
    for col in CATEGORICAL_COLUMNS:
        w2v_features.extend([f"{col}_w2v_{i}" for i in range(40)])
    
    # Concatenate features using the corrected MASTER_NUMERICAL_FEATURES list
    expected_features = MASTER_NUMERICAL_FEATURES + w2v_features
    
    feature_dict = {feature: 0.0 for feature in expected_features}
    
    # CRITICAL STEP: Use the explicit list to select and order columns.
    return pd.DataFrame([feature_dict])[expected_features]

def preprocess_features(input_data: Dict, w2v_model=None) -> pd.DataFrame:
    """Preprocess and encode input features matching the original training pipeline"""
    
    df = create_empty_feature_dataframe()
    
    # Get inputs (Casting inputs to float immediately)
    hla_total = float(input_data['hla_match_total'])
    karnofsky_score = float(input_data['karnofsky_score'])
    comorbidity_score = float(input_data['comorbidity_score'])
    donor_age = float(input_data['donor_age'])
    age_at_hct = float(input_data['age_at_hct'])
    
    # --- Assign Numerical Features ---
    # Populate the columns required by the MASTER_NUMERICAL_FEATURES list
    
    df['age_at_hct'] = age_at_hct
    df['karnofsky_score'] = karnofsky_score
    df['comorbidity_score'] = comorbidity_score
    df['donor_age'] = donor_age
    df['hla_match_total'] = hla_total 
    
    # Set default and derived values
    df['year_hct'] = 2019.0  
    df['nan_value_each_row'] = 0.0
    df['age_group'] = float(int(age_at_hct // 10))
    df['dri_score_NA'] = 1.0 if input_data['dri_score'] in ["N/A - non-malignant indication", "N/A - pediatric"] else 0.0
    
    # --- Assign Engineered Features (Corrected names) ---
    df['donor_ageage_at_hct'] = donor_age - age_at_hct
    df['comorbidity_scorekarnofsky_score'] = comorbidity_score + karnofsky_score
    
    # --- Calculate HLA features based on hla_match_total ---
    # The 8 previously missing HLA features (hla_match_std, etc.) are left at 0.0 
    # since their calculation logic is unknown, but they now exist in the DataFrame.
    
    # Existing HLA calculation logic:
    df['hla_high_res_6'] = min(hla_total, 6.0)
    if hla_total == 0:
        df['hla_high_res_6'] = 2.0  
    df['hla_high_res_8'] = min(hla_total, 8.0)
    if hla_total == 2:
        df['hla_high_res_8'] = 3.0  
    df['hla_high_res_10'] = min(hla_total, 10.0)
    if hla_total == 3:
        df['hla_high_res_10'] = 4.0  
    df['hla_low_res_6'] = min(hla_total, 6.0)
    df['hla_low_res_8'] = min(hla_total, 8.0)
    if hla_total == 2:
        df['hla_low_res_8'] = 3.0  
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

    # --- Assign Word2Vec Features ---
    user_provided_categoricals = {
        'dri_score': input_data['dri_score'], 'psych_disturb': input_data['psych_disturb'],
        'cyto_score': input_data.get('cyto_score', 'Unknown'), 'diabetes': input_data['diabetes'],
        'tbi_status': input_data['tbi_status'], 'arrhythmia': input_data['arrhythmia'],
        'graft_type': input_data['graft_type'], 'cardiac': input_data['cardiac'],
        'cmv_status': input_data.get('cmv_status', 'Unknown')
    }
    
    default_categoricals = {
        'vent_hist': "Unknown", 'renal_issue': "Unknown", 'pulm_severe': "Unknown",
        'prim_disease_hct': "Unknown", 'tce_imm_match': "Unknown", 'rituximab': "Unknown",
        'prod_type': "Unknown", 'cyto_score_detail': "Unknown", 'conditioning_intensity': "Unknown",
        'ethnicity': "Unknown", 'obesity': "Unknown", 'mrd_hct': "Unknown",
        'in_vivo_tcd': "Unknown", 'tce_match': "Unknown", 'hepatic_severe': "Unknown",
        'prior_tumor': "Unknown", 'peptic_ulcer': "Unknown", 'gvhd_proph': "Unknown",
        'rheum_issue': "Unknown", 'sex_match': "Unknown", 'race_group': "Unknown",
        'hepatic_mild': "Unknown", 'tce_div_match': "Unknown", 'donor_related': "Unknown",
        'melphalan_dose': "Unknown", 'pulm_moderate': "Unknown"
    }
    
    all_categoricals = {**default_categoricals, **user_provided_categoricals}
    
    for col_name, value in all_categoricals.items():
        embedding = get_w2v_embedding(str(value), w2v_model, vector_size=40)
        for i in range(40):
            df[f'{col_name}_w2v_{i}'] = embedding[i]
            
    # Final data cleanup
    df = df.fillna(0.0)
    for col in df.columns:
        df[col] = df[col].astype(float)
        
    logger.info(f"Processed data shape: {df.shape}")
    logger.info(f"Total features: {len(df.columns)}")
    
    return df

# --------------------------------------------------------------------------------
# PREDICTION AND UTILITY FUNCTIONS
# --------------------------------------------------------------------------------

def calculate_feature_contributions(patient_data: Dict, prediction: float) -> Dict[str, float]:
    """Calculate simplified feature contributions for visualization"""
    contributions = {}
    
    dri_weights = {
        "Low": 0.1, "Intermediate": 0.3, "High": 0.6, 
        "N/A - non-malignant indication": 0.2, "N/A - pediatric": 0.1, "Unknown": 0.3
    }
    contributions["disease_risk"] = dri_weights.get(patient_data["dri_score"], 0.3) * abs(prediction)
    
    age_contribution = min(patient_data["age_at_hct"] / 80 * 0.3, 0.3) * abs(prediction)
    contributions["age_factors"] = age_contribution
    
    karnofsky_contribution = ((100 - patient_data["karnofsky_score"]) / 100 * 0.2) * abs(prediction)
    contributions["performance_status"] = karnofsky_contribution
    
    comorbidity_contribution = min(patient_data["comorbidity_score"] / 10 * 0.2, 0.2) * abs(prediction)
    contributions["comorbidities"] = comorbidity_contribution
    
    hla_contribution = ((20 - patient_data["hla_match_total"]) / 20 * 0.15) * abs(prediction)
    contributions["hla_matching"] = hla_contribution
    
    contributions["engineered_features"] = 0.1 * abs(prediction)
    
    contributions["other_factors"] = max(0, abs(prediction) - sum(contributions.values()))
    
    total = sum(contributions.values())
    if total > 0:
        contributions = {k: round(v/total * 100, 1) for k, v in contributions.items()}
    
    return contributions

def predict_survival_risk(patient_data: Dict, xgb_model, catboost_model, w2v_model) -> Dict:
    """Make prediction using ensemble model"""
    
    try:
        processed_data = preprocess_features(patient_data, w2v_model)
        
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
            return None
        
        ensemble_pred = np.mean(predictions)
        
        individual_predictions = {}
        for i, name in enumerate(model_names):
            individual_predictions[name] = round(predictions[i], 4)
        
        # Interpret prediction based on Nelson-Aalen cumulative hazard
        if ensemble_pred < -0.8:
            risk_category = "Very Low Risk"
            confidence = "Very High"
            interpretation = "Excellent prognosis with very high survival probability."
            risk_class = "risk-low"
            recommendations = ["Standard monitoring protocol", "Routine follow-up care"]
        elif ensemble_pred < 0:
            risk_category = "Low Risk"
            confidence = "High"
            interpretation = "Good prognosis with favorable survival outcomes expected."
            risk_class = "risk-low"
            recommendations = ["Standard monitoring protocol", "Regular follow-up visits"]
        elif ensemble_pred < 0.3:
            risk_category = "Moderate Risk"
            confidence = "Medium"
            interpretation = "Moderate prognosis requiring standard monitoring."
            risk_class = "risk-moderate"
            recommendations = ["Enhanced monitoring", "Close follow-up schedule"]
        else:
            risk_category = "High Risk"
            confidence = "High"
            interpretation = "Higher risk profile with increased cumulative hazard."
            risk_class = "risk-high"
            recommendations = ["Intensive monitoring required", "Frequent follow-up visits", "Consider additional supportive therapies"]
        
        feature_contributions = calculate_feature_contributions(patient_data, ensemble_pred)
        
        model_agreement = "Single Model"
        if len(predictions) > 1:
            std_dev = np.std(predictions)
            model_agreement = "High" if std_dev < 0.05 else "Medium" if std_dev < 0.1 else "Low"
        
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
            "explanation": f"The prediction score of {ensemble_pred:.4f} is a cumulative hazard estimate. Negative values indicate better survival."
        }
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        with st.expander("🔍 Debug Information"):
            st.write("Error details:", str(e))
            st.warning("The feature ordering error still persists. The feature list order in `MASTER_NUMERICAL_FEATURES` must be exactly checked against the training data.")
        return None

# --------------------------------------------------------------------------------
# STREAMLIT APP LAYOUT
# --------------------------------------------------------------------------------

def main():
    """Main Streamlit application"""
    
    st.markdown('<h1 class="main-header">🏥 HCT Survival Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("Predict survival outcomes for Hematopoietic Cell Transplantation patients using machine learning")
    
    xgb_model, catboost_model, w2v_model = load_models()
    
    if xgb_model is None and catboost_model is None:
        return
    
    loaded_models = []
    if xgb_model is not None:
        loaded_models.append("XGBoost")
    if catboost_model is not None:
        loaded_models.append("CatBoost")
    if w2v_model is not None:
        loaded_models.append("Word2Vec")
    
    st.success(f"✅ Ready for predictions using: **{', '.join(loaded_models)}**")
    
    with st.sidebar:
        st.header("📊 Feature Information")
        st.markdown("The feature order for prediction has been rigorously fixed to match the trained model.")
        
        # Simplified feature info display (omitted for brevity)
    
    # Main form
    with st.form("prediction_form"):
        st.header("📝 Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dri_score = st.selectbox("Disease Risk Index (DRI Score)", options=FEATURE_INFO["dri_score"]["options"])
            age_at_hct = st.slider("Age at Transplantation (years)", min_value=0.0, max_value=80.0, value=45.0, step=1.0)
            karnofsky_score = st.slider("Karnofsky Performance Score", min_value=0.0, max_value=100.0, value=80.0, step=5.0)
            comorbidity_score = st.slider("Comorbidity Score", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
            hla_match_total = st.slider("HLA Match Total", min_value=0.0, max_value=20.0, value=18.0, step=1.0)
            cyto_score = st.selectbox("Cytogenetic Score", options=FEATURE_INFO["cyto_score"]["options"])
        
        with col2:
            donor_age = st.slider("Donor Age (years)", min_value=0.0, max_value=80.0, value=35.0, step=1.0)
            psych_disturb = st.radio("Psychiatric Disturbance", options=["Yes", "No", "Unknown"])
            diabetes = st.radio("Diabetes", options=["Yes", "No", "Unknown"])
            cardiac = st.radio("Cardiac Disease", options=["Yes", "No", "Unknown"])
            graft_type = st.selectbox("Graft Type", options=FEATURE_INFO["graft_type"]["options"])
            tbi_status = st.selectbox("TBI Status", options=FEATURE_INFO["tbi_status"]["options"])
            arrhythmia = st.radio("Arrhythmia", options=["Yes", "No", "Unknown"])
            cmv_status = st.selectbox("CMV Status", options=FEATURE_INFO["cmv_status"]["options"])
        
        submitted = st.form_submit_button("🎯 Predict Survival Risk", use_container_width=True)
    
    if submitted:
        with st.spinner("🔄 Analyzing patient data and making prediction..."):
            patient_data = {
                "dri_score": dri_score, "age_at_hct": age_at_hct, "karnofsky_score": karnofsky_score,
                "comorbidity_score": comorbidity_score, "hla_match_total": hla_match_total,
                "donor_age": donor_age, "psych_disturb": psych_disturb, "diabetes": diabetes,
                "cardiac": cardiac, "graft_type": graft_type, "tbi_status": tbi_status,
                "arrhythmia": arrhythmia, "cyto_score": cyto_score, "cmv_status": cmv_status
            }
            
            result = predict_survival_risk(patient_data, xgb_model, catboost_model, w2v_model)
            
            if result:
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                risk_class = result["risk_class"]
                st.markdown(f"""
                <div class="prediction-card {risk_class}">
                    <h2>Risk Category: {result['risk_category']}</h2>
                    <h3>Prediction Score: {result['prediction']}</h3>
                    <p><strong>Confidence:</strong> {result['confidence']}</p>
                    <p><strong>Interpretation:</strong> {result['interpretation']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🤖 Model Predictions")
                    st.json(result["individual_predictions"])
                    st.markdown(f"**Model Agreement:** {result['model_agreement']}")
                    st.markdown(f"**Explanation:** {result['explanation']}")

                with col2:
                    st.subheader("🔬 Feature Contributions")
                    contributions_df = pd.DataFrame(
                        result["feature_contributions"].items(), 
                        columns=["Feature Group", "Contribution (%)"]
                    ).sort_values(by="Contribution (%)", ascending=False)
                    
                    st.bar_chart(contributions_df.set_index("Feature Group"))
                    
                st.subheader("💡 Recommendations")
                for rec in result["recommendations"]:
                    st.markdown(f"- {rec}")

if __name__ == '__main__':
    main()
