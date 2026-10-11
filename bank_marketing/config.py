import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "bank-full.csv"
EXPERIMENT = "TC5061_BankMarketing_Cookiecutter_PerazaLeonardo"
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"
)
