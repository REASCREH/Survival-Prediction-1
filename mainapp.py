# app.py
import streamlit as st
import pandas as pd
import numpy as np
import glob
import joblib
import logging
import os
from typing import Tuple, List, Dict

# optional imports (we handle missing packages gracefully)
try:
    import xgboost as xgb  # optional
except Exception:
    xgb = None
try:
    import catboost
except Exception:
    catboost = None
try:
    from gensim.models import Word2Vec
except Exception:
    Word2Vec = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="HCT Survival Prediction", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1f77b4; text-align:center; margin-bottom:1rem; }
    .prediction-card { background-color:#f0f2f6; padding:1rem; border-radius:10px; border-left:5px solid #1f77b4; margin:1rem 0; }
    .risk-low { background-color:#d4edda; border-left:5px solid #28a745; }
    .risk-moderate { background-color:#fff3cd; border-left:5px solid #ffc107; }
    .risk-high { background-color:#f8d7da; border-left:5px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# --- Helper utilities ----------------------------------------------------------

def find_model_files() -> Dict[str, str]:
    """Search current directory for model files and return candidate paths."""
    patterns = ["*.pkl", "*.joblib", "*.cbm", "*.model", "models/*.pkl", "models/*.joblib", "models/*.cbm", "models/*.model"]
    found = {}
    for pat in patterns:
        for f in glob.glob(pat):
            name = f.lower()
            if "xgb" in name or (name.endswith(".model") and "word2vec" not in name and "w2v" not in name and xgb is not None):
                found.setdefault("xgb", f)
            if "catboost" in name or name.endswith(".cbm") or "cat" in name:
                found.setdefault("catboost", f)
            if "w2v" in name or "word2vec" in name:
                found.setdefault("w2v", f)
            # generic pick for sklearn-like
            if name.endswith(".pkl") or name.endswith(".joblib"):
                found.setdefault("sklearn_like", f)
    return found

@st.cache_resource
def load_model_from_path(path: str):
    """Load a model file using best-effort logic."""
    try:
        # CatBoost .cbm
        if catboost is not None and path.lower().endswith(".cbm"):
            model = catboost.CatBoost()
            model.load_model(path)
            return model
        # XGBoost native booster stored by joblib may be an sklearn wrapper or native
        if path.lower().endswith(".model") and xgb is not None:
            # Try xgboost.Booster / XGBClassifier saved by xgb native save_model
            try:
                booster = xgb.Booster()
                booster.load_model(path)
                return booster
            except Exception:
                pass
        # fallback joblib/pickle
        try:
            model = joblib.load(path)
            return model
        except Exception as e:
            st.warning(f"joblib.load failed for {path}: {e}")
            # as final fallback try CatBoost load_model on generic file
            if catboost is not None:
                try:
                    cb = catboost.CatBoost()
                    cb.load_model(path)
                    return cb
                except Exception:
                    pass
            raise
    except Exception as e:
        st.error(f"Failed to load model at {path}: {e}")
        return None

def get_model_feature_names(model) -> List[str]:
    """
    Try multiple strategies to detect the feature names the model expects, in order:
    - XGBoost Booster feature_names
    - XGB sklearn wrapper .feature_names_in_ or .get_booster().feature_names
    - CatBoost: model.feature_names_ or model.get_feature_names()
    - sklearn-like: .feature_names_in_ or .feature_names
    - If provided as a dict-like object (custom), attempt common keys.
    """
    if model is None:
        return []

    # XGBoost Booster (native)
    try:
        if xgb is not None and isinstance(model, xgb.core.Booster):
            fn = model.feature_names
            if fn is not None:
                return list(fn)
    except Exception:
        pass

    # XGBoost sklearn wrapper
    try:
        if xgb is not None and hasattr(model, "get_booster"):
            booster = model.get_booster()
            fn = getattr(booster, "feature_names", None)
            if fn:
                return list(fn)
    except Exception:
        pass

    # CatBoost
    try:
        # CatBoost sklearn wrapper
        if catboost is not None and hasattr(catboost, "CatBoost") and isinstance(model, catboost.CatBoost):
            # catboost model stores feature names in model.feature_names_ or model.get_feature_names()
            fn = getattr(model, "feature_names_", None)
            if fn:
                return list(fn)
            try:
                fn2 = model.get_feature_names()
                if fn2:
                    return list(fn2)
            except Exception:
                pass
    except Exception:
        pass

    # sklearn-like objects
    try:
        if hasattr(model, "feature_names_in_"):
            fn = list(model.feature_names_in_)
            return fn
    except Exception:
        pass

    try:
        if hasattr(model, "feature_names"):
            fn = getattr(model, "feature_names")
            if isinstance(fn, (list, tuple)):
                return list(fn)
    except Exception:
        pass

    # fallback: some joblib saved dict where keys are feature_names
    try:
        if isinstance(model, dict) and "feature_names" in model:
            return list(model["feature_names"])
    except Exception:
        pass

    # lastly, try to inspect model.booster or model.get_booster()
    try:
        booster = getattr(model, "booster_", None) or getattr(model, "booster", None)
        if booster is not None:
            fn = getattr(booster, "feature_names", None)
            if fn:
                return list(fn)
    except Exception:
        pass

    return []

def get_w2v_embedding(word: str, model, vector_size: int=40) -> np.ndarray:
    """Return W2V embedding or zeros if missing."""
    if model is None:
        return np.zeros(vector_size, dtype=float)
    try:
        # gensim 4.x: model.wv.key_to_index
        wv = model.wv if hasattr(model, "wv") else model
        if hasattr(wv, "key_to_index") and str(word) in wv.key_to_index:
            return np.array(wv[str(word)], dtype=float)
        if str(word) in wv:
            return np.array(wv[str(word)], dtype=float)
    except Exception:
        pass
    return np.zeros(vector_size, dtype=float)

# --------------------------------------------------------------------------------
# Preprocessing that aligns to model expected features
# --------------------------------------------------------------------------------

# Known categorical list (from your notebook)
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

def build_input_dataframe(patient_input: Dict, expected_feature_names: List[str], w2v_model=None) -> pd.DataFrame:
    """
    Create a single-row DataFrame matching expected_feature_names precisely.
    Fill computed features where possible, zero for missing ones.
    """
    # Start with zeros
    feature_dict = {f: 0.0 for f in expected_feature_names}

    # Basic numeric inputs provided by the app UI
    # be defensive: convert where present
    def safe_float(k, default=0.0):
        try:
            return float(patient_input.get(k, default))
        except Exception:
            return default

    hla_total = safe_float("hla_match_total", 0.0)
    age_at_hct = safe_float("age_at_hct", 0.0)
    karnofsky_score = safe_float("karnofsky_score", 0.0)
    comorbidity_score = safe_float("comorbidity_score", 0.0)
    donor_age = safe_float("donor_age", 0.0)

    # heuristic assignments for engineered features (only if model expects them)
    assignments = {
        "age_at_hct": age_at_hct,
        "karnofsky_score": karnofsky_score,
        "comorbidity_score": comorbidity_score,
        "donor_age": donor_age,
        "year_hct": 2019.0,
        "nan_value_each_row": 0.0,
        "age_group": float(int(age_at_hct // 10)) if age_at_hct is not None else 0.0,
        "dri_score_NA": 1.0 if patient_input.get("dri_score", "").startswith("N/A") else 0.0,
        "donor_ageage_at_hct": donor_age - age_at_hct,
        "comorbidity_scorekarnofsky_score": comorbidity_score + karnofsky_score
    }

    # HLA helper engineering (only set if those names are expected)
    hla_calc = {
        "hla_high_res_6": min(hla_total, 6.0),
        "hla_high_res_8": min(hla_total, 8.0),
        "hla_high_res_10": min(hla_total, 10.0),
        "hla_low_res_6": min(hla_total, 6.0),
        "hla_low_res_8": min(hla_total, 8.0),
        "hla_low_res_10": min(hla_total, 10.0),
        "hla_nmdp_6": min(hla_total, 6.0),
    }
    # Binary match features (threshold logic)
    hla_bin = {
        "hla_match_a_high": 1.0 if hla_total >= 16 else 0.0,
        "hla_match_a_low": 1.0 if hla_total >= 14 else 0.0,
        "hla_match_b_high": 1.0 if hla_total >= 16 else 0.0,
        "hla_match_b_low": 1.0 if hla_total >= 14 else 0.0,
        "hla_match_c_high": 1.0 if hla_total >= 16 else 0.0,
        "hla_match_c_low": 1.0 if hla_total >= 14 else 0.0,
        "hla_match_drb1_high": 1.0 if hla_total >= 16 else 0.0,
        "hla_match_drb1_low": 1.0 if hla_total >= 14 else 0.0,
        "hla_match_dqb1_high": 1.0 if hla_total >= 16 else 0.0,
        "hla_match_dqb1_low": 1.0 if hla_total >= 14 else 0.0,
    }

    # Apply assignments only where the feature name exists in model's expected list
    for k, v in {**assignments, **hla_calc, **hla_bin}.items():
        if k in feature_dict:
            feature_dict[k] = float(v)

    # Fill W2V features only if model expects them
    # Determine which categorical W2V columns are present in expected_feature_names
    w2v_columns_present = [f for f in expected_feature_names if "_w2v_" in f]
    # Map base categorical -> indices present
    categorical_w2v_bases = {}
    for col in w2v_columns_present:
        base = col.rsplit("_w2v_", 1)[0]
        categorical_w2v_bases.setdefault(base, []).append(col)

    # For each base categorical expected, compute embedding and assign to each expected column
    for base, cols in categorical_w2v_bases.items():
        # if the base is in our known CATEGORICAL_COLUMNS, pick value from patient_input; else try to infer
        base_name = base
        # patient provided value might be plain (like 'dri_score') or 'dri_score' but user input keys are simple names
        cat_value = patient_input.get(base_name, "Unknown")
        emb = get_w2v_embedding(str(cat_value), w2v_model, vector_size=40)
        # assign embedding to each expected column (ex: 'dri_score_w2v_0' gets emb[0])
        for col in cols:
            try:
                idx = int(col.rsplit("_w2v_", 1)[1])
            except Exception:
                idx = 0
            if idx < len(emb):
                feature_dict[col] = float(emb[idx])
            else:
                feature_dict[col] = 0.0

    # Final DataFrame
    df = pd.DataFrame([feature_dict], columns=expected_feature_names)
    # Ensure numeric types
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except Exception:
            df[c] = 0.0
    return df

# --------------------------------------------------------------------------------
# Prediction wrappers
# --------------------------------------------------------------------------------

def ensemble_predict(processed_df: pd.DataFrame, models: Dict[str, object]) -> Tuple[float, Dict]:
    """Perform predictions for each model present and return ensemble mean and individual preds"""
    preds = []
    individual = {}
    for name, model in models.items():
        if model is None:
            continue
        try:
            # XGBoost Booster vs sklearn wrapper
            if xgb is not None and isinstance(model, xgb.core.Booster):
                # booster expects DMatrix
                dmat = xgb.DMatrix(processed_df, feature_names=processed_df.columns.tolist())
                p = model.predict(dmat)
                val = float(p[0])
            elif xgb is not None and hasattr(model, "predict") and hasattr(model, "get_booster"):
                # sklearn wrapper
                val = float(model.predict(processed_df)[0])
            elif catboost is not None and isinstance(model, catboost.CatBoost):
                val = float(model.predict(processed_df)[0])
            else:
                # sklearn-like
                val = float(model.predict(processed_df)[0])
            preds.append(val)
            individual[name] = round(val, 6)
        except Exception as e:
            st.warning(f"Prediction failed for {name}: {e}")
    if not preds:
        return None, {}
    ens = float(np.mean(preds))
    return ens, individual

# --------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------

def main():
    st.markdown('<h1 class="main-header">🏥 HCT Survival Prediction (Robust Loader)</h1>', unsafe_allow_html=True)
    st.write("This app attempts to automatically align the input feature vector to the model's expected feature names/order.")

    model_files = find_model_files()
    st.info(f"Found model files: {model_files}")

    # Let user choose which model file to use (if many)
    selected = None
    if model_files:
        chosen_key = st.selectbox("Choose model to load", options=list(model_files.keys()))
        selected = model_files.get(chosen_key)
    else:
        st.warning("No model files found in working directory. Place model files (.pkl/.joblib/.cbm/.model) in the app folder.")
        selected = st.text_input("Or provide path to model file manually:")

    load_button = st.button("🔁 Load model")

    # store loaded models
    if "loaded_models" not in st.session_state:
        st.session_state.loaded_models = {}
    if "w2v_model" not in st.session_state:
        st.session_state.w2v_model = None
    if "expected_features" not in st.session_state:
        st.session_state.expected_features = []

    if load_button:
        if not selected:
            st.error("Provide a model file path.")
        else:
            model_obj = load_model_from_path(selected)
            if model_obj is None:
                st.error("Failed to load the model.")
            else:
                # Determine a friendly name
                friendly = os.path.splitext(os.path.basename(selected))[0]
                st.session_state.loaded_models[friendly] = model_obj
                st.success(f"Loaded model: {friendly}")

                # Attempt to also find a w2v model file
                w2v_candidates = [p for k,p in model_files.items() if k in ("w2v","word2vec") or "w2v" in p.lower()]
                if w2v_candidates:
                    try:
                        w2v_path = w2v_candidates[0]
                        from gensim.models import Word2Vec
                        st.session_state.w2v_model = Word2Vec.load(w2v_path) if w2v_path.lower().endswith(".model") else joblib.load(w2v_path)
                        st.info(f"Loaded Word2Vec model from: {w2v_path}")
                    except Exception as e:
                        st.warning(f"Could not load W2V model automatically: {e}")

                # Extract expected features from the loaded model (last loaded)
                expected = get_model_feature_names(model_obj)
                if not expected:
                    st.warning("Could not detect model feature names automatically. You can paste the feature names list manually (see debug section).")
                else:
                    st.session_state.expected_features = expected
                    st.success(f"Detected {len(expected)} expected features from the model.")

    # Sidebar help & manual override
    with st.sidebar:
        st.header("Deployment notes")
        st.markdown("- Ensure the model file(s) are in app directory.") 
        st.markdown("- If automatic feature extraction fails, paste the model's feature name list in the Debug panel.")
        st.markdown("If you trained with a saved `X_train.columns.tolist()` file, load it here.")

    # Input form
    with st.form("input_form"):
        st.subheader("📝 Patient Inputs (these map to model features where possible)")
        col1, col2 = st.columns(2)
        with col1:
            dri_score = st.selectbox("DRI Score", options=["Low","Intermediate","High","N/A - non-malignant indication","N/A - pediatric","Unknown"])
            age_at_hct = st.number_input("Age at transplant", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
            karnofsky_score = st.number_input("Karnofsky score", min_value=0.0, max_value=100.0, value=80.0)
            comorbidity_score = st.number_input("Comorbidity score", min_value=0.0, max_value=20.0, value=2.0)
            hla_match_total = st.number_input("HLA match total", min_value=0.0, max_value=20.0, value=18.0)
        with col2:
            donor_age = st.number_input("Donor age", min_value=0.0, max_value=120.0, value=35.0)
            psych_disturb = st.selectbox("Psych disturbance", options=["Yes","No","Unknown"])
            diabetes = st.selectbox("Diabetes", options=["Yes","No","Unknown"])
            cardiac = st.selectbox("Cardiac disease", options=["Yes","No","Unknown"])
            graft_type = st.selectbox("Graft type", options=["Bone marrow","Peripheral blood","Cord blood","Unknown"])
            tbi_status = st.selectbox("TBI status", options=["No TBI","TBI +- Other, >cGy","Unknown"])
            arrhythmia = st.selectbox("Arrhythmia", options=["Yes","No","Unknown"])
            cmv_status = st.selectbox("CMV status", options=["Positive","Negative","Unknown"])
        submit = st.form_submit_button("🔮 Predict")

    if submit:
        if not st.session_state.loaded_models:
            st.error("No model loaded. First load a model file (top controls).")
        else:
            # Use the most recently loaded model for feature extraction and prediction
            last_name = list(st.session_state.loaded_models.keys())[-1]
            model = st.session_state.loaded_models[last_name]
            expected = st.session_state.expected_features or get_model_feature_names(model)

            # If no expected features discovered, let user paste them manually
            if not expected:
                manual = st.text_area("Paste a Python list of feature names (e.g. ['f1','f2',...])")
                try:
                    expected = eval(manual) if manual.strip() else []
                except Exception as e:
                    st.error(f"Could not parse manual feature list: {e}")
                    expected = []

            patient_data = {
                "dri_score": dri_score, "age_at_hct": age_at_hct, "karnofsky_score": karnofsky_score,
                "comorbidity_score": comorbidity_score, "hla_match_total": hla_match_total,
                "donor_age": donor_age, "psych_disturb": psych_disturb, "diabetes": diabetes,
                "cardiac": cardiac, "graft_type": graft_type, "tbi_status": tbi_status,
                "arrhythmia": arrhythmia, "cmv_status": cmv_status
            }

            processed = build_input_dataframe(patient_data, expected, w2v_model=st.session_state.w2v_model)

            # Show debug summary: missing features we couldn't compute, etc.
            st.subheader("⚙️ Debug / Feature Alignment")
            st.markdown(f"- Model expected {len(expected)} features.")
            st.write("Preview of constructed input (first 40 cols):")
            st.dataframe(processed.iloc[:, :40])

            # Show which expected features were left zero (helpful)
            zero_cols = [c for c in processed.columns if processed[c].abs().sum() == 0]
            if zero_cols:
                st.warning(f"{len(zero_cols)} columns are zero-filled (model expects them but we didn't compute them from inputs). Example: {zero_cols[:10]}")
            else:
                st.success("All expected model columns have some non-zero value in the constructed input row (or no zero-only columns detected).")

            # Predict using ensemble if multiple loaded models
            ens, individual = ensemble_predict(processed, st.session_state.loaded_models)
            if ens is None:
                st.error("Prediction failed for all loaded models. See warnings above.")
            else:
                # Interpretation (same logic as your previous code)
                pred = float(ens)
                if pred < -0.8:
                    risk_category, confidence, interpretation, risk_class = ("Very Low Risk","Very High","Excellent prognosis","risk-low")
                elif pred < 0:
                    risk_category, confidence, interpretation, risk_class = ("Low Risk","High","Good prognosis","risk-low")
                elif pred < 0.3:
                    risk_category, confidence, interpretation, risk_class = ("Moderate Risk","Medium","Moderate prognosis","risk-moderate")
                else:
                    risk_category, confidence, interpretation, risk_class = ("High Risk","High","Higher risk profile","risk-high")

                st.markdown("---")
                st.markdown(f"""
                <div class="prediction-card {risk_class}">
                    <h2>Risk Category: {risk_category}</h2>
                    <h3>Prediction Score: {pred:.4f}</h3>
                    <p><strong>Confidence:</strong> {confidence}</p>
                    <p><strong>Interpretation:</strong> {interpretation}</p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("Model Predictions")
                st.json(individual)
                st.markdown(f"**Ensemble mean:** {pred:.6f}")

                st.subheader("Recommendations (automated)")
                if risk_category in ("Very Low Risk","Low Risk"):
                    st.write("- Standard monitoring protocol; routine follow-up.")
                elif risk_category == "Moderate Risk":
                    st.write("- Enhanced monitoring, closer follow-up.")
                else:
                    st.write("- Intensive monitoring, consider additional therapies.")

if __name__ == "__main__":
    main()
