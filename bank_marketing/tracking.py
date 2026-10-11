import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import ConfusionMatrixDisplay

from bank_marketing.config import ROOT


def archivos_codigo():
    return sorted((ROOT / "bank_marketing").rglob("*.py")) + [
        ROOT / name
        for name in (
            "LICENSE",
            "train.py",
            "requirements.txt",
            "MLproject",
            "python_env.yaml",
            "pyproject.toml",
        )
    ]


def huella_codigo():
    huella = hashlib.sha256()
    for archivo in archivos_codigo():
        huella.update(str(archivo.relative_to(ROOT)).encode())
        huella.update(archivo.read_bytes())
    return huella.hexdigest()


def guardar_artefactos(metricas, y_valid, predicciones):
    with TemporaryDirectory() as carpeta:
        ruta = Path(carpeta)
        archivo = ruta / "metricas.json"
        archivo.write_text(json.dumps(metricas, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(archivo))
        ConfusionMatrixDisplay.from_predictions(
            y_valid, predicciones, display_labels=["No", "Sí"], cmap="Blues"
        )
        plt.title("Matriz de confusión en validación")
        plt.tight_layout()
        figura = ruta / "matriz_confusion.png"
        plt.savefig(figura, dpi=160)
        plt.close()
        mlflow.log_artifact(str(figura))
    for archivo in archivos_codigo():
        destino = Path("codigo") / archivo.relative_to(ROOT).parent
        mlflow.log_artifact(str(archivo), artifact_path=str(destino))
