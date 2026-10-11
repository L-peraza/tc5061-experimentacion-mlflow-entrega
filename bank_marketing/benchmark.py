import json
from statistics import median
from timeit import repeat

import numpy as np
import pandas as pd

from bank_marketing.config import DATA_FILE, ROOT
from bank_marketing.dataset import cargar_datos


def limpiar_con_listas(X):
    result = X.copy()
    for column in X.columns:
        result[column] = pd.Series(
            [np.nan if value == "unknown" else value for value in X[column]],
            index=X.index,
            dtype=X[column].dtype,
        )
    return result


def main():
    X, _, _ = cargar_datos()
    X = X[["job", "education", "contact"]]
    pd.testing.assert_frame_equal(
        limpiar_con_listas(X), X.replace("unknown", np.nan)
    )
    funciones = {
        "listas": lambda: limpiar_con_listas(X),
        "vectorizado": lambda: X.replace("unknown", np.nan),
    }
    tiempos = {
        nombre: median(repeat(funcion, repeat=5, number=3)) / 3
        for nombre, funcion in funciones.items()
    }
    with DATA_FILE.open() as archivo:
        filas = sum(1 for _ in archivo) - 1
    resultado = {
        "filas": filas,
        "columnas": 3,
        "repeticiones": 5,
        "ejecuciones_por_repeticion": 3,
        "segundos_mediana": tiempos,
        "razon_listas_sobre_vectorizado": (
            tiempos["listas"] / tiempos["vectorizado"]
        ),
        "salidas_equivalentes": True,
        "alcance": "Solo limpieza de tres columnas; depende del equipo.",
    }
    destino = ROOT / "reports" / "benchmark.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps(resultado, indent=2) + "\n")
    print(json.dumps(resultado, indent=2))


if __name__ == "__main__":
    main()
