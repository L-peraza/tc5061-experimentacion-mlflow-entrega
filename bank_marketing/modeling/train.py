import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np

from bank_marketing.config import DATA_FILE, EXPERIMENT, ROOT, TRACKING_URI
from bank_marketing.dataset import cargar_datos, dividir_datos
from bank_marketing.modeling.evaluate import evaluar_modelo
from bank_marketing.modeling.pipeline import crear_modelo
from bank_marketing.tracking import guardar_artefactos, huella_codigo


def leer_argumentos():
    parser = argparse.ArgumentParser(
        description="Bank Marketing: experimentos reproducibles con MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_FILE,
        help="Ruta al CSV de Bank Marketing, con separador punto y coma.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Regularización inversa; debe ser mayor que cero.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Ponderar las clases o darles el mismo peso.",
    )
    parser.add_argument(
        "--fit-intercept",
        choices=["true", "false"],
        default="true",
        help="Incluir o excluir el intercepto de la regresión.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla de las particiones y del modelo.",
    )
    parser.add_argument(
        "--modelo",
        choices=["logistica", "dummy"],
        default="logistica",
        help="Modelo aprendido o referencia de clase mayoritaria.",
    )
    parser.add_argument(
        "--run-name", default=None, help="Nombre opcional de la corrida."
    )
    args = parser.parse_args()
    if args.c <= 0:
        parser.error("--c debe ser mayor que cero")
    return args


def main():
    args = leer_argumentos()
    np.random.seed(args.seed)
    X, y, datos_sha256 = cargar_datos(args.data_path)
    X_train, X_valid, y_train, y_valid, n_prueba = dividir_datos(
        X, y, args.seed
    )
    modelo = crear_modelo(
        X_train,
        c=args.c,
        class_weight=args.class_weight,
        fit_intercept=args.fit_intercept == "true",
        seed=args.seed,
        modelo=args.modelo,
    )
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=args.run_name) as corrida:
        if args.run_name:
            mlflow.set_tag("mlflow.runName", args.run_name)
        parametros = {
            "modelo": args.modelo,
            "data_path_resuelto": str(args.data_path.expanduser().resolve()),
            "seed": args.seed,
            "train_size": len(X_train),
            "validation_size": len(X_valid),
            "test_reservado_size": n_prueba,
            "duration_excluida": True,
        }
        if args.modelo == "logistica":
            parametros.update(
                {
                    "C": args.c,
                    "class_weight": args.class_weight,
                    "fit_intercept": args.fit_intercept,
                    "solver": "liblinear",
                    "max_iter": 1000,
                }
            )
        else:
            parametros["strategy"] = "most_frequent"
        for nombre, valor in parametros.items():
            mlflow.log_param(nombre, valor)
        mlflow.set_tags(
            {
                "datos_sha256": datos_sha256,
                "codigo_sha256": huella_codigo(),
                "evaluacion": "validacion; prueba final sin consultar",
                "estructura": "Cookiecutter Data Science v2.3.0",
            }
        )
        modelo.fit(X_train, y_train)
        metricas, predicciones = evaluar_modelo(modelo, X_valid, y_valid)
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)
        guardar_artefactos(metricas, y_valid, predicciones)
        mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            serialization_format="cloudpickle",
            pip_requirements=str(ROOT / "requirements.txt"),
            code_paths=[str(ROOT / "bank_marketing")],
        )
        print("Run ID:", corrida.info.run_id)
        print(json.dumps(metricas, indent=2))


if __name__ == "__main__":
    main()
