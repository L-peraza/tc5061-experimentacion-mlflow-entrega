from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluar_modelo(modelo, X_valid, y_valid):
    predicciones = modelo.predict(X_valid)
    probabilidades = modelo.predict_proba(X_valid)[:, 1]
    metricas = {
        "val_accuracy": accuracy_score(y_valid, predicciones),
        "val_precision": precision_score(
            y_valid, predicciones, zero_division=0
        ),
        "val_recall": recall_score(y_valid, predicciones, zero_division=0),
        "val_f1": f1_score(y_valid, predicciones, zero_division=0),
        "val_roc_auc": roc_auc_score(y_valid, probabilidades),
    }
    return metricas, predicciones
