from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from bank_marketing.features import BankCleaner


def crear_modelo(
    X_train,
    c=1.0,
    class_weight="balanced",
    fit_intercept=True,
    seed=42,
    modelo="logistica",
):
    if modelo == "dummy":
        return DummyClassifier(strategy="most_frequent")
    numericas = X_train.select_dtypes(include="number").columns.tolist()
    categoricas = X_train.select_dtypes(exclude="number").columns.tolist()
    numeros = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", RobustScaler()),
        ]
    )
    categorias = Pipeline(
        [
            (
                "imputar",
                SimpleImputer(
                    strategy="constant", fill_value="sin_informacion"
                ),
            ),
            ("codificar", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocesar = ColumnTransformer(
        [
            ("numeros", numeros, numericas),
            ("categorias", categorias, categoricas),
        ]
    )
    clasificador = LogisticRegression(
        C=c,
        class_weight=None if class_weight == "none" else "balanced",
        fit_intercept=fit_intercept,
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
    )
    return Pipeline(
        [
            ("limpiar", BankCleaner()),
            ("preprocesar", preprocesar),
            ("clasificar", clasificador),
        ]
    )
