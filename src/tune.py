# src/tune.py
import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def objective_xgb(trial, X, y, scale_pos_weight=1.0, cv_splits=5, random_state=42):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": random_state,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
    }

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, proba))

    return float(np.mean(aucs))

def tune_xgb(X, y, scale_pos_weight=1.0, n_trials=30, timeout=None, seed=42):
    study = optuna.create_study(direction="maximize", study_name="xgb_churn")
    study.optimize(lambda t: objective_xgb(t, X, y, scale_pos_weight), n_trials=n_trials, timeout=timeout)
    return study
