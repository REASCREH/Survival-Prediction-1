# mainapp.py
import os
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import logging
from typing import Dict, List

# Optional imports (we'll handle missing packages gracefully)
try:
    import xgboost as xgb
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

# --- Known categorical columns (from your notebook) ---
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

# --- Numerical / engineered features observed in your notebook (from printed outputs) ---
# This list aggregates all the numeric/engineered names that appeared in your training pipeline.
BASE_NUMERICAL_FEATURES = [
    'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6', 'hla_high_res_6', 'hla_high_res_10',
    'hla_match_dqb1_high', 'hla_nmdp_6', 'hla_match_c_low', 'hla_match_drb1_low', 'hla_match_dqb1_low',
    'year_hct', 'hla_match_a_high', 'donor_age', 'hla_match_b_low', 'age_at_hct', 'hla_match_a_low',
    'hla_match_b_high', 'comorbidity_score', 'karnofsky_score', 'hla_low_res_8', 'hla_match_drb1_high',
    'hla_low_res_10', 'nan_value_each_row', 'age_group', 'dri_score_NA',
    # engineered interaction names (from your notebook prints)
    'donor_age-age_at_hct', 'comorbidity_score+karnofsky_score',
    'comorbidity_score-karnofsky_score', 'comorbidity_score*karnofsky_score',
    'comorbidity_score/karnofsky_score',
    # Additional names that were mentioned in the notebook -> include to be safe:
    'hla_high_res_sum', 'hla_high_res_avg', 'hla_high_low_diff', 'hla_high_low_ratio',
    'hla_match_total', 'hla_match_count', 'hla_match_std', 'hla_high_res_log', 'hla_high_res_squared'
]

# Derive the full expected feature list (numerical first, then W2V feature blocks for each categorical)
def make_expected_feature_list() -> List[str]:
    """
    Build the fallback expected features list to use if the loaded model does not expose feature names.
    This reproduces the same order used during training: numeric + (for each categorical) 40 W2V dims.
    """
    expected = []
    # Ensure deterministic ordering
    expected.extend(BASE_NUMERICAL_FEATURES)
    # Append W2V features in the exact order of CATEGORICAL_COLUMNS
    for cat in CATEGORICAL_COLUMNS:
        expected.extend([f"{cat}_w2v_{i}" for i in range(40)])
    return expected

FALLBACK_EXPECTED_FEATURES = make_expected_feature_list()

# ------------------------------
# Model loading utilities
# ------------------------------
def find_model_files() -> Dict[str, str]:
    """
    Look for the exact filenames you reported. Case-sensitive check to match files exactly.
    """
    candidates = {}
    if os.path.exists("XGB_model.pkl"):
        candidates["xgb"] = "XGB_model.pkl"
    if os.path.exists("CatBoost_model.pkl"):
        candidates["catboost"] = "CatBoost_model.pkl"
    if os.path.exists("w2v_model.pkl"):
        candidates["w2v"] = "w2v_model.pkl"
    # Also look for any other pickles if present (fallback)
    for f in glob.glob("*.pkl"):
        lname = f.lower()
        if "xgb" in lname and "xgb" not in candidates:
            candidates.setdefault("xgb", f)
        if "catboost" in lname and "catboost" not in candidates:
            candidates.setdefault("catboost", f)
        if "w2v" in lname and "w2v" not in candidates:
            candidates.setdefault("w2v", f)
    return candidates

@st.cache_resource
def load_models_from_disk():
    """
    Load models from disk. Return (xgb_model, catboost_model, w2v_model, model_feature_names)
    model_feature_names is a list if we could extract it from one of the loaded models (priority: XGB -> CatBoost -> sklearn wrapper)
    """
    found = find_model_files()
    st.info(f"Found model files: {found}")
    xgb_model = None
    cat_model = None
    w2v_model = None
    model_feature_names = None

    # Load Word2Vec first (used for preprocessing)
    if "w2v" in found:
        try:
            w2v_path = found["w2v"]
            # Try gensim load first
            if Word2Vec is not None and w2v_path.lower().endswith(".model"):
                w2v_model = Word2Vec.load(w2v_path)
            else:
                w2v_model = joblib.load(w2v_path)
            st.success("✅ Word2Vec loaded")
        except Exception as e:
            st.warning(f"Could not load w2v_model.pkl: {e}")

    # Load XGBoost
    if "xgb" in found:
        try:
            path = found["xgb"]
            mdl = joblib.load(path)
            xgb_model = mdl
            st.success("✅ XGBoost/saved model loaded")
        except Exception as e:
            st.warning(f"Failed to joblib.load XGB model: {e}")
            # Try to load as raw xgboost Booster if package available
            if xgb is not None:
                try:
                    booster = xgb.Booster()
                    booster.load_model(path)
                    xgb_model = booster
                    st.success("✅ XGBoost Booster loaded via xgboost.Booster()")
                except Exception as e2:
                    st.error(f"Could not load XGB model via Booster: {e2}")

    # Load CatBoost
    if "catboost" in found:
        try:
            path = found["catboost"]
            cat_model = joblib.load(path)
            st.success("✅ CatBoost (joblib) loaded")
        except Exception as e:
            # Try CatBoost native load if catboost package present
            try:
                if catboost is not None:
                    cb = catboost.CatBoost()
                    cb.load_model(path)
                    cat_model = cb
                    st.success("✅ CatBoost model loaded via CatBoost().load_model()")
            except Exception as e2:
                st.warning(f"Failed to load CatBoost model: {e2}")

    # Try to extract feature names from whichever model exposes them (priority order)
    def try_extract_feature_names(model_obj):
        if model_obj is None:
            return None
        # xgboost Booster
        try:
            if xgb is not None and isinstance(model_obj, xgb.core.Booster):
                fn = getattr(model_obj, "feature_names", None)
                if fn:
                    return list(fn)
        except Exception:
            pass
        # sklearn wrapper that has get_booster
        try:
            if xgb is not None and hasattr(model_obj, "get_booster"):
                booster = model_obj.get_booster()
                fn = getattr(booster, "feature_names", None)
                if fn:
                    return list(fn)
        except Exception:
            pass
        # CatBoost native
        try:
            if catboost is not None and isinstance(model_obj, catboost.CatBoost):
                fn = getattr(model_obj, "feature_names_", None)
                if fn:
                    return list(fn)
                try:
                    fn2 = model_obj.get_feature_names()
                    if fn2:
                        return list(fn2)
                except Exception:
                    pass
        except Exception:
            pass
        # sklearn-like
        try:
            if hasattr(model_obj, "feature_names_in_"):
                return list(getattr(model_obj, "feature_names_in_"))
        except Exception:
            pass
        try:
            if hasattr(model_obj, "feature_names"):
                fn = getattr(model_obj, "feature_names")
                if isinstance(fn, (list, tuple)):
                    return list(fn)
        except Exception:
            pass
        return None

    # Try extraction from xgb first, then catboost, then generic
    model_feature_names = try_extract_feature_names(xgb_model) or try_extract_feature_names(cat_model)

    return xgb_model, cat_model, w2v_model, model_feature_names

# ------------------------------
# Word2Vec helper
# ------------------------------
def get_w2v_embedding(value: str, w2v_model, vector_size: int = 40) -> np.ndarray:
    if w2v_model is None:
        return np.zeros(vector_size, dtype=float)
    try:
        wv = w2v_model.wv if hasattr(w2v_model, "wv") else w2v_model
        # gensim key check
        if hasattr(wv, "key_to_index"):
            if str(value) in wv.key_to_index:
                return np.array(wv[str(value)], dtype=float)
        # older gensim or fallback
        if str(value) in wv:
            return np.array(wv[str(value)], dtype=float)
    except Exception:
        pass
    return np.zeros(vector_size, dtype=float)

# ------------------------------
# Preprocessing that mirrors your training notebook
# ------------------------------
def feature_engineering_single_row(input_dict: Dict, w2v_model=None, expected_features: List[str] = None) -> pd.DataFrame:
    """
    Build a single-row DataFrame aligned to expected_features (if provided).
    If expected_features is None, build the fallback full vector (BASE_NUMERICAL_FEATURES + W2V for each categorical).
    """
    if expected_features is None or len(expected_features) == 0:
        expected_features = FALLBACK_EXPECTED_FEATURES

    # Start with zeros
    feature_values = {f: 0.0 for f in expected_features}

    # Safe getters
    def sf(k, default=0.0):
        v = input_dict.get(k, default)
        try:
            return float(v)
        except Exception:
            return default

    # Core numeric inputs commonly provided from UI
    age_at_hct = sf("age_at_hct", 0.0)
    donor_age = sf("donor_age", 0.0)
    comorbidity_score = sf("comorbidity_score", 0.0)
    karnofsky_score = sf("karnofsky_score", 0.0)
    hla_total = sf("hla_match_total", 0.0)

    # Basic assignments (only assign if model expects those names)
    assign_map = {
        'age_at_hct': age_at_hct,
        'donor_age': donor_age,
        'comorbidity_score': comorbidity_score,
        'karnofsky_score': karnofsky_score,
        'year_hct': 2019.0,
        'nan_value_each_row': 0.0,
        'age_group': float(int(age_at_hct // 10)) if age_at_hct is not None else 0.0,
        'dri_score_NA': 1.0 if str(input_dict.get('dri_score','')).startswith("N/A") else 0.0,
        # engineered interactions names (use exact symbols user used in notebook)
        'donor_age-age_at_hct': donor_age - age_at_hct,
        'comorbidity_score+karnofsky_score': comorbidity_score + karnofsky_score,
        'comorbidity_score-karnofsky_score': comorbidity_score - karnofsky_score,
        'comorbidity_score*karnofsky_score': comorbidity_score * (karnofsky_score if karnofsky_score != 0 else 0.0),
        'comorbidity_score/karnofsky_score': (comorbidity_score / karnofsky_score) if karnofsky_score not in (0, None) else 0.0
    }

    # HLA engineered features (as in your notebook)
    hla_assign = {
        "hla_high_res_6": min(hla_total, 6.0),
        "hla_high_res_8": min(hla_total, 8.0),
        "hla_high_res_10": min(hla_total, 10.0),
        "hla_low_res_6": min(hla_total, 6.0),
        "hla_low_res_8": min(hla_total, 8.0),
        "hla_low_res_10": min(hla_total, 10.0),
        "hla_nmdp_6": min(hla_total, 6.0),
        # binary thresholds
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
        "hla_high_res_sum": hla_total,  # approximate
        "hla_high_res_avg": (hla_total / 1.0) if hla_total else 0.0
    }

    # apply numeric assignments where expected
    for k, v in {**assign_map, **hla_assign}.items():
        if k in feature_values:
            try:
                feature_values[k] = float(v)
            except Exception:
                feature_values[k] = 0.0

    # Handle W2V features: for each categorical base expected by model, compute embedding from w2v_model
    # Determine which categorical_w2v bases are present in expected features
    w2v_cols_present = [c for c in expected_features if "_w2v_" in c]
    categorical_bases = {}
    for col in w2v_cols_present:
        base = col.rsplit("_w2v_", 1)[0]
        categorical_bases.setdefault(base, []).append(col)

    # For each base categorical expected, get value from input_dict or 'Unknown'
    for base, cols in categorical_bases.items():
        # base name should match one of the CATEGORICAL_COLUMNS; if not, use last part before any suffix
        base_name = base
        cat_value = input_dict.get(base_name, "Unknown")
        emb = get_w2v_embedding(str(cat_value), w2v_model, vector_size=40)
        for col in cols:
            try:
                idx = int(col.rsplit("_w2v_", 1)[1])
            except Exception:
                idx = 0
            feature_values[col] = float(emb[idx]) if idx < len(emb) else 0.0

    # Final dataframe in the exact expected_features order
    df = pd.DataFrame([feature_values], columns=expected_features)
    # Ensure floats
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except Exception:
            df[c] = 0.0
    return df

# ------------------------------
# Prediction helpers
# ------------------------------
def predict_with_models(processed_df: pd.DataFrame, models: Dict[str, object]) -> Dict:
    """
    Run predictions on processed_df for each provided model. Return dict with individual preds and ensemble mean.
    """
    preds = {}
    numerical_preds = []
    for name, model in models.items():
        if model is None:
            continue
        try:
            # XGBoost Booster
            if xgb is not None and isinstance(model, xgb.core.Booster):
                dmat = xgb.DMatrix(processed_df, feature_names=processed_df.columns.tolist())
                p = model.predict(dmat)
                val = float(p[0])
            else:
                # For sklearn-like and catboost wrappers
                val = float(model.predict(processed_df)[0])
            preds[name] = round(val, 6)
            numerical_preds.append(val)
        except Exception as e:
            st.warning(f"Prediction with {name} failed: {e}")
    if not numerical_preds:
        return None
    ensemble = float(np.mean(numerical_preds))
    return {"ensemble": round(ensemble, 6), "individual": preds}

# ------------------------------
# Streamlit UI and flow
# ------------------------------
def main():
    st.markdown("<h1 style='text-align:center;color:#1f77b4'>🏥 HCT Survival Prediction</h1>", unsafe_allow_html=True)
    st.markdown("This app reproduces your training preprocessing and loads your saved models for inference.")

    xgb_model, cat_model, w2v_model, model_feature_names = load_models_from_disk()

    loaded_models = {}
    if xgb_model is not None:
        loaded_models["XGB_model"] = xgb_model
    if cat_model is not None:
        loaded_models["CatBoost_model"] = cat_model

    if not loaded_models:
        st.error("No XGBoost or CatBoost models loaded. Ensure `XGB_model.pkl` and/or `CatBoost_model.pkl` are present in the app folder.")
        st.info("Found model files on disk? Filenames are case-sensitive. Place files in the same directory as mainapp.py.")
    else:
        st.success(f"Models ready: {', '.join(loaded_models.keys())}")

    if w2v_model is None:
        st.warning("Word2Vec model not loaded. W2V embeddings will be zero vectors. Place `w2v_model.pkl` or `w2v_model.model` in the app folder to enable embeddings.")

    # Show whether we could extract feature names from the model
    if model_feature_names:
        st.info(f"Detected {len(model_feature_names)} expected feature names from the model.")
    else:
        st.info(f"Model did not expose feature names. Using internal fallback list of {len(FALLBACK_EXPECTED_FEATURES)} features (numeric + W2V).")

    # Input form (only core fields — rest filled with defaults/Unknown)
    with st.form("patient_form"):
        st.subheader("Patient inputs (core values). Other fields are filled with defaults/Unknown to match training.")
        col1, col2 = st.columns(2)
        with col1:
            dri_score = st.selectbox("Disease Risk Index (DRI score)", options=["Low","Intermediate","High","N/A - non-malignant indication","N/A - pediatric","Unknown"])
            age_at_hct = st.number_input("Age at transplant", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
            karnofsky_score = st.number_input("Karnofsky score (0-100)", min_value=0.0, max_value=100.0, value=80.0)
            comorbidity_score = st.number_input("Comorbidity score (0-10)", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
            hla_match_total = st.number_input("HLA match total (0-20)", min_value=0.0, max_value=40.0, value=18.0, step=1.0)
        with col2:
            donor_age = st.number_input("Donor age", min_value=0.0, max_value=120.0, value=35.0)
            psych_disturb = st.selectbox("Psych disturbance", options=["Yes","No","Unknown"])
            diabetes = st.selectbox("Diabetes", options=["Yes","No","Unknown"])
            cardiac = st.selectbox("Cardiac disease", options=["Yes","No","Unknown"])
            graft_type = st.selectbox("Graft Type", options=["Bone marrow","Peripheral blood","Cord blood","Unknown"])
        submitted = st.form_submit_button("Predict survival risk")

    if submitted:
        # Build patient data dict (include all categorical keys expected by W2V)
        patient_inputs = {
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
        }
        # add all categorical keys with default 'Unknown' if not provided
        for cat in CATEGORICAL_COLUMNS:
            if cat not in patient_inputs:
                patient_inputs[cat] = patient_inputs.get(cat, "Unknown")

        # Decide which feature list to use: model_feature_names (if available) else fallback
        expected_features = model_feature_names if model_feature_names else FALLBACK_EXPECTED_FEATURES

        # Build processed dataframe row aligned to expected features
        processed_df = feature_engineering_single_row(patient_inputs, w2v_model=w2v_model, expected_features=expected_features)

        # Debug info: show preview and zero-filled columns
        st.subheader("⚙️ Input features preview (first 60 columns)")
        st.dataframe(processed_df.iloc[:, :60])

        zero_cols = [c for c in processed_df.columns if np.isclose(processed_df[c].abs().sum(), 0.0)]
        if zero_cols:
            st.warning(f"{len(zero_cols)} features are zero-filled (we could not compute them from inputs). Showing first 20 examples: {zero_cols[:20]}")
        else:
            st.success("No zero-only columns detected in constructed input.")

        # Run predictions
        preds = predict_with_models(processed_df, loaded_models)
        if preds is None:
            st.error("Prediction failed — no model produced output. Check the model logs/warnings above.")
        else:
            ens = preds["ensemble"]
            individual = preds["individual"]

            # Interpret ensemble score with same thresholds you used
            if ens < -0.8:
                risk_category = "Very Low Risk"; confidence = "Very High"; interpretation = "Excellent prognosis"
                css_class = "risk-low"
            elif ens < 0:
                risk_category = "Low Risk"; confidence = "High"; interpretation = "Good prognosis"
                css_class = "risk-low"
            elif ens < 0.3:
                risk_category = "Moderate Risk"; confidence = "Medium"; interpretation = "Moderate prognosis"
                css_class = "risk-moderate"
            else:
                risk_category = "High Risk"; confidence = "High"; interpretation = "Higher risk profile"
                css_class = "risk-high"

            st.markdown("---")
            st.markdown(f"""
            <div style="background:#f0f2f6;padding:1rem;border-radius:10px;border-left:5px solid #1f77b4">
                <h2>Risk Category: {risk_category}</h2>
                <h3>Prediction Score: {ens:.6f}</h3>
                <p><strong>Confidence:</strong> {confidence}</p>
                <p><strong>Interpretation:</strong> {interpretation}</p>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Model predictions (individual)")
            st.json(individual)
            st.markdown(f"**Ensemble mean**: {ens:.6f}")

            st.subheader("Recommendations (automated heuristic)")
            if risk_category in ("Very Low Risk", "Low Risk"):
                st.write("- Standard monitoring protocol; routine follow-up.")
            elif risk_category == "Moderate Risk":
                st.write("- Enhanced monitoring and closer follow-up.")
            else:
                st.write("- Intensive monitoring and consider additional supportive therapies.")

if __name__ == "__main__":
    main()
