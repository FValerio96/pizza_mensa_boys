"""
train_risk_model.py

Addestra un modello a due stadi (Hurdle Model) sul dataset di Chicago:

  Stage 1 – Classificazione binaria:
      "Questo arco ha probabilità di avere crimini?" (si/no)
      → Random Forest Classifier con class_weight='balanced'

  Stage 2 – Regressione sulla severità:
      "Tra gli archi con crimini, quanti crimes_per_km ci aspettiamo?"
      → Gradient Boosting Regressor (su log1p per stabilità)

  Risk score finale per ogni arco:
      risk = P(crimine > 0) × E[crimes_per_km | crimine > 0]

Il modello salvato (models/risk_model.pkl) può essere applicato a qualsiasi
città di cui abbiamo le stesse feature OSM + lampioni Overpass.

Uso:
    python notebooks/train_risk_model.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
DATASET_PATH = Path("data/chicago_loop_edges_dataset.csv")
MODEL_DIR = Path("models")

# Feature disponibili ovunque su OSM / Overpass
FEATURE_COLS = [
    "highway",       # categorica → one-hot
    "length",        # numerica
    "lanes",         # numerica (parsata)
    "maxspeed",      # numerica (parsata)
    "lamp_count",    # numerica
    "lamp_density",  # numerica
    "lighting_score",# numerica
    "oneway",        # bool
]

# Tipi di highway da tenere come categorie; il resto → "other"
TOP_HIGHWAY_TYPES = [
    "footway", "service", "secondary", "tertiary", "residential",
    "steps", "trunk", "path", "corridor", "pedestrian",
    "unclassified", "secondary_link", "trunk_link", "elevator",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _parse_numeric(series: pd.Series) -> pd.Series:
    """Estrai il primo numero da una colonna mista (es. '30 mph', '["3","4"]').

    - Se è già un numero, lo ritorna.
    - Se è una stringa con numeri, prende il primo.
    - Se è una lista JSON, prende la media.
    - Altrimenti NaN.
    """
    def _parse_one(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        # Prova a interpretare come lista JSON ["3","4"]
        if s.startswith("["):
            try:
                vals = json.loads(s)
                nums = [float(re.sub(r"[^\d.]", "", str(v))) for v in vals if re.search(r"\d", str(v))]
                return np.mean(nums) if nums else np.nan
            except (json.JSONDecodeError, ValueError):
                pass
        # Trova il primo numero nella stringa
        m = re.search(r"[\d.]+", s)
        return float(m.group()) if m else np.nan

    return series.apply(_parse_one)


def _normalize_highway(series: pd.Series) -> pd.Series:
    """Normalizza highway: se è una lista, prende il primo elemento; se raro → 'other'."""
    def _norm(x):
        if pd.isna(x):
            return "other"
        s = str(x).strip()
        if s.startswith("["):
            try:
                vals = json.loads(s)
                s = str(vals[0]) if vals else "other"
            except (json.JSONDecodeError, ValueError):
                pass
        return s if s in TOP_HIGHWAY_TYPES else "other"
    return series.apply(_norm)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Partendo dal CSV grezzo, costruisce il DataFrame di feature pronte per il modello."""
    out = pd.DataFrame(index=df.index)

    # Numeriche
    out["length"] = df["length"].astype(float)
    out["lanes"] = _parse_numeric(df["lanes"])
    out["maxspeed"] = _parse_numeric(df["maxspeed"])
    out["lamp_count"] = df["lamp_count"].astype(float)
    out["lamp_density"] = df["lamp_density"].astype(float)
    out["lighting_score"] = df["lighting_score"].astype(float)
    out["oneway"] = df["oneway"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)

    # Imputa NaN numerici con 0 (lanes, maxspeed: assenza = sconosciuto → 0)
    out["lanes"] = out["lanes"].fillna(0)
    out["maxspeed"] = out["maxspeed"].fillna(0)

    # Feature derivate
    out["log_length"] = np.log1p(out["length"])
    out["has_lamps"] = (out["lamp_count"] > 0).astype(int)

    # Categorica: highway → one-hot
    hw = _normalize_highway(df["highway"])
    hw_dummies = pd.get_dummies(hw, prefix="hw", dtype=int)
    out = pd.concat([out, hw_dummies], axis=1)

    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_hurdle_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Addestra il modello a due stadi e restituisce un dizionario con tutto il necessario."""

    print("=" * 60)
    print("[train] Preparazione feature...")
    print("=" * 60)

    X = prepare_features(df)
    feature_names = list(X.columns)

    # Target
    y_bin = (df["num_crimes"] > 0).astype(int)
    y_reg = df.loc[y_bin == 1, "crimes_per_km"].astype(float)

    print(f"  Archi totali: {len(X)}")
    print(f"  Archi con crimini: {y_bin.sum()} ({y_bin.mean()*100:.1f}%)")
    print(f"  Feature: {len(feature_names)}")
    print(f"  Feature names: {feature_names}")

    # --- Split ---
    X_train, X_test, y_bin_train, y_bin_test = train_test_split(
        X, y_bin, test_size=test_size, random_state=random_state, stratify=y_bin
    )

    # =======================================================================
    # Stage 1: Classificatore binario
    # =======================================================================
    print("\n" + "=" * 60)
    print("[Stage 1] Classificazione binaria: has_crime sì/no")
    print("=" * 60)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_bin_train)

    # Evaluation
    y_bin_pred = clf.predict(X_test)
    y_bin_proba = clf.predict_proba(X_test)[:, 1]

    print("\nClassification Report (test set):")
    print(classification_report(y_bin_test, y_bin_pred, target_names=["safe", "crime"]))
    auc = roc_auc_score(y_bin_test, y_bin_proba)
    print(f"  ROC-AUC: {auc:.3f}")

    # Cross-validation AUC
    cv_auc = cross_val_score(clf, X, y_bin, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"  5-Fold CV AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    # Feature importance
    print("\n  Top 10 feature (classificatore):")
    importances_clf = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    for fname, imp in importances_clf.head(10).items():
        print(f"    {fname:25s} {imp:.4f}")

    # =======================================================================
    # Stage 2: Regressore sulla severità (solo archi con crimini)
    # =======================================================================
    print("\n" + "=" * 60)
    print("[Stage 2] Regressione: crimes_per_km (solo archi con crimini > 0)")
    print("=" * 60)

    # Filtra solo archi con crimini
    mask_train_pos = y_bin_train == 1
    mask_test_pos = y_bin_test == 1

    X_train_pos = X_train[mask_train_pos]
    y_train_pos = df.loc[X_train_pos.index, "crimes_per_km"].astype(float)

    X_test_pos = X_test[mask_test_pos]
    y_test_pos = df.loc[X_test_pos.index, "crimes_per_km"].astype(float)

    print(f"  Archi training con crimini: {len(X_train_pos)}")
    print(f"  Archi test con crimini: {len(X_test_pos)}")

    # Regressione su log1p(crimes_per_km) per ridurre skew
    y_train_log = np.log1p(y_train_pos)

    reg = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=random_state,
    )
    reg.fit(X_train_pos, y_train_log)

    # Evaluation
    y_test_pred_log = reg.predict(X_test_pos)
    y_test_pred = np.expm1(y_test_pred_log)  # torna in scala originale

    mae = mean_absolute_error(y_test_pos, y_test_pred)
    r2 = r2_score(y_test_pos, y_test_pred)
    print(f"\n  MAE (test): {mae:.2f} crimes/km")
    print(f"  R² (test):  {r2:.3f}")
    print(f"  Media reale (test): {y_test_pos.mean():.2f}, Media predetta: {y_test_pred.mean():.2f}")

    # Feature importance
    print("\n  Top 10 feature (regressore):")
    importances_reg = pd.Series(reg.feature_importances_, index=feature_names).sort_values(ascending=False)
    for fname, imp in importances_reg.head(10).items():
        print(f"    {fname:25s} {imp:.4f}")

    # =======================================================================
    # Salvataggio
    # =======================================================================
    model_bundle = {
        "classifier": clf,
        "regressor": reg,
        "feature_names": feature_names,
        "top_highway_types": TOP_HIGHWAY_TYPES,
        "training_stats": {
            "n_edges": len(df),
            "n_crime_edges": int(y_bin.sum()),
            "auc": float(auc),
            "cv_auc_mean": float(cv_auc.mean()),
            "reg_mae": float(mae),
            "reg_r2": float(r2),
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "risk_model.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"\n[train] Modello salvato in {model_path}")

    return model_bundle


# ---------------------------------------------------------------------------
# Compute risk score su tutto il dataset (per validazione)
# ---------------------------------------------------------------------------

def compute_risk_scores(model_bundle: dict, df: pd.DataFrame) -> pd.Series:
    """Calcola il risk score per ogni arco usando il modello a due stadi.

    risk = P(crimine > 0) × expm1(predicted_log_crimes_per_km)
    """
    clf = model_bundle["classifier"]
    reg = model_bundle["regressor"]

    X = prepare_features(df)
    # Assicura che ci siano le stesse colonne del training
    for col in model_bundle["feature_names"]:
        if col not in X.columns:
            X[col] = 0
    X = X[model_bundle["feature_names"]]

    p_crime = clf.predict_proba(X)[:, 1]
    severity_log = reg.predict(X)
    severity = np.expm1(severity_log).clip(0)

    risk = p_crime * severity
    return pd.Series(risk, index=df.index, name="risk_score")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[train_risk_model] Caricamento dataset Chicago...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  {len(df)} archi caricati da {DATASET_PATH}")

    model_bundle = train_hurdle_model(df)

    # Validazione: calcola risk score sull'intero dataset
    print("\n" + "=" * 60)
    print("[Validazione] Risk score sull'intero dataset Chicago")
    print("=" * 60)

    risk = compute_risk_scores(model_bundle, df)
    df["risk_score"] = risk

    print(f"\n  Risk score – statistiche:")
    print(f"    min:    {risk.min():.4f}")
    print(f"    median: {risk.median():.4f}")
    print(f"    mean:   {risk.mean():.4f}")
    print(f"    P95:    {np.percentile(risk, 95):.4f}")
    print(f"    max:    {risk.max():.4f}")

    # Confronto: archi con crimini reali vs risk score
    with_crimes = df[df["num_crimes"] > 0]
    without_crimes = df[df["num_crimes"] == 0]
    print(f"\n  Risk medio – archi CON crimini:   {with_crimes['risk_score'].mean():.4f}")
    print(f"  Risk medio – archi SENZA crimini: {without_crimes['risk_score'].mean():.4f}")
    ratio = with_crimes["risk_score"].mean() / max(without_crimes["risk_score"].mean(), 1e-9)
    print(f"  Rapporto: {ratio:.1f}x")

    # Top 10 archi più rischiosi
    print(f"\n  Top 10 archi per risk_score:")
    top = df.nlargest(10, "risk_score")[["name", "highway", "length", "num_crimes", "crimes_per_km", "risk_score"]]
    print(top.to_string(index=False))

    print("\n[train_risk_model] Completato.")


if __name__ == "__main__":
    main()
