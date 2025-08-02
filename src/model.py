# src/model.py
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap

def compute_scale_pos_weight(y):
    """Compute class weight for XGBoost: (# negative / # positive)."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return float(neg / max(pos, 1))

def train_xgb(X_train, y_train, **kwargs) -> XGBClassifier:
    """
    Train an XGBoost classifier with imbalance-aware defaults.
    You can override any hyperparameter via kwargs (e.g., from Optuna).
    """
    # If caller doesn't pass scale_pos_weight, compute it from y_train
    spw = kwargs.pop("scale_pos_weight", compute_scale_pos_weight(y_train))

    params = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=spw,  # <<< key change: handle class imbalance
    )
    params.update(kwargs)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test) -> Dict[str, float]:
    """Return metrics and show a confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    return {
        "accuracy": report["accuracy"],
        "roc_auc": float(auc),
        "precision_no": report["0"]["precision"],
        "recall_no": report["0"]["recall"],
        "precision_yes": report["1"]["precision"],
        "recall_yes": report["1"]["recall"],
    }

def shap_summary(model, X_reference, out_path: str = None):
    """Create and optionally save a global SHAP summary plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_reference)
    shap.summary_plot(shap_values, X_reference, show=False)
    if out_path:
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
