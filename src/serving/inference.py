"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

Loads an MLflow pyfunc model and performs serving-time feature transforms.

Fixes included:
- Removes broken "/app/model" override (which caused \app\model errors on Windows)
- Loads ONE chosen MLflow model directory
- If feature_columns.txt is missing, automatically infers expected input columns
  from the loaded model and aligns to those.
"""

from pathlib import Path
import pandas as pd
import mlflow.pyfunc

# ============================================================
# MODEL LOADING
# ============================================================
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../src
MODEL_DIR = PROJECT_ROOT / "serving" / "model" / "3b1a41221fc44548aed629fa42b762e0" / "artifacts" / "model"
MODEL_DIR = MODEL_DIR.resolve()

#MODEL_DIR = Path(
  #  r"C:\Users\pranj\Desktop\telcochurn\Telco-Customer-Churn-ML\src\serving\model\3b1a41221fc44548aed629fa42b762e0\artifacts\model"
#).resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "src" / "serving" / "model" / "3b1a41221fc44548aed629fa42b762e0" / "artifacts" / "model"
MODEL_DIR = MODEL_DIR.resolve()

mlmodel_path = MODEL_DIR / "MLmodel"
if not mlmodel_path.exists():
    raise FileNotFoundError(f"MLflow model folder not valid (missing MLmodel): {MODEL_DIR}")

model = mlflow.pyfunc.load_model(str(MODEL_DIR))


# ============================================================
# FEATURE SCHEMA (feature_columns.txt optional)
# ============================================================

def _infer_feature_cols_from_model(m) -> list[str]:
    """
    Try to infer the model's expected input column names from MLflow metadata.
    Works for most sklearn/xgboost models logged with a signature.
    """
    cols = None

    # 1) If the model has a signature with input names
    try:
        sig = getattr(m, "metadata", None)
        if sig is not None and getattr(sig, "signature", None) is not None:
            inputs = sig.signature.inputs
            if inputs is not None and hasattr(inputs, "input_names") and inputs.input_names():
                cols = list(inputs.input_names())
    except Exception:
        pass

    # 2) If the underlying model exposes feature_names_in_ (sklearn pipelines)
    if not cols:
        try:
            impl = getattr(m, "_model_impl", None)
            sk_model = getattr(impl, "sk_model", None) if impl is not None else None
            if sk_model is not None and hasattr(sk_model, "feature_names_in_"):
                cols = list(sk_model.feature_names_in_)
        except Exception:
            pass

    if not cols:
        raise RuntimeError(
            "Could not infer feature columns from the MLflow model. "
            "You must provide feature_columns.txt inside the model directory."
        )

    return cols

"""
feature_file = MODEL_DIR / "feature_columns.txt"
if feature_file.exists():
    with open(feature_file, "r", encoding="utf-8") as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from {feature_file}")
else:
    FEATURE_COLS = _infer_feature_cols_from_model(model)
    print(f"⚠️ feature_columns.txt not found. Inferred {len(FEATURE_COLS)} columns from model signature/metadata.")
"""
# === FEATURE SCHEMA LOADING ===
# feature_columns.txt is stored in the artifacts folder (one level above MODEL_DIR)

feature_file = MODEL_DIR.parent / "feature_columns.txt"   # ✅ artifacts/feature_columns.txt

if not feature_file.exists():
    raise FileNotFoundError(f"feature_columns.txt not found at: {feature_file}")

with open(feature_file, "r", encoding="utf-8") as f:
    FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]

print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from {feature_file}")

# ============================================================
# TRANSFORM CONSTANTS (must match training)
# ============================================================

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # numeric coercion
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # binary encoding
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    # one-hot for remaining categoricals
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # bool -> int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # align to expected columns
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df


def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        result = preds[0] if isinstance(preds, (list, tuple)) and len(preds) == 1 else preds
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    return "Likely to churn" if result == 1 else "Not likely to churn"
