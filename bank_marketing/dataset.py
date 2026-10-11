import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import train_test_split

from bank_marketing.config import DATA_FILE


def cargar_datos(archivo=DATA_FILE):
    archivo = Path(archivo).expanduser().resolve()
    if not archivo.exists():
        if archivo != DATA_FILE.resolve():
            raise FileNotFoundError(f"No existe el CSV: {archivo}")
        archivo.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
        )
        with urlopen(url, timeout=90) as respuesta:
            with ZipFile(BytesIO(respuesta.read())) as externo:
                with ZipFile(BytesIO(externo.read("bank.zip"))) as interno:
                    archivo.write_bytes(interno.read("bank-full.csv"))
    df = pd.read_csv(archivo, sep=";").drop_duplicates()
    X = df.drop(columns=["y", "duration"])
    y = df["y"].map({"no": 0, "yes": 1})
    huella = hashlib.sha256(archivo.read_bytes()).hexdigest()
    return X, y, huella


def dividir_datos(X, y, seed=42):
    X_desarrollo, X_prueba, y_desarrollo, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_desarrollo,
        y_desarrollo,
        test_size=0.25,
        stratify=y_desarrollo,
        random_state=seed,
    )
    return X_train, X_valid, y_train, y_valid, len(X_prueba)
