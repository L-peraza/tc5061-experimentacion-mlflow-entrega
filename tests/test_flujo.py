import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from bank_marketing.dataset import cargar_datos, dividir_datos
from bank_marketing.features import BankCleaner
from bank_marketing.modeling.evaluate import evaluar_modelo
from bank_marketing.modeling.pipeline import crear_modelo


class TestBankCleaner(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            {
                "job": ["unknown", "admin."],
                "balance": [-50, 100],
                "pdays": [-1, 20],
            }
        )

    def test_limpia_sin_modificar_la_entrada(self):
        original = self.X.copy()
        cleaner = BankCleaner()
        result = cleaner.fit_transform(self.X)
        pd.testing.assert_frame_equal(self.X, original)
        self.assertTrue(pd.isna(result.loc[0, "job"]))
        self.assertEqual(result.loc[0, "balance"], -50)
        self.assertEqual(result.loc[0, "pdays"], -1)
        self.assertEqual(cleaner.n_features_in_, 3)
        self.assertEqual(clone(cleaner).unknown_value, "unknown")

    def test_exige_fit_antes_de_transformar(self):
        with self.assertRaises(NotFittedError):
            BankCleaner().transform(self.X)

    def test_rechaza_esquema_distinto(self):
        cleaner = BankCleaner().fit(self.X)
        with self.assertRaises(ValueError):
            cleaner.transform(self.X.drop(columns="job"))


class TestDatosYModelo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, _ = cargar_datos()
        cls.particion = dividir_datos(cls.X, cls.y, seed=42)

    def test_particion_fija_y_separada(self):
        X_train, X_valid, _, _, n_prueba = self.particion
        repeticion = dividir_datos(self.X, self.y, seed=42)
        self.assertEqual(len(X_train), 27126)
        self.assertEqual(len(X_valid), 9042)
        self.assertEqual(n_prueba, 9043)
        self.assertTrue(X_train.index.intersection(X_valid.index).empty)
        pd.testing.assert_frame_equal(X_train, repeticion[0])
        pd.testing.assert_frame_equal(X_valid, repeticion[1])
        self.assertNotIn("duration", self.X)

    def test_imputacion_solo_con_entrenamiento(self):
        X = pd.DataFrame(
            {
                "balance": [1.0, 2.0, 3.0, np.nan],
                "job": ["admin.", "unknown", "admin.", "services"],
            }
        )
        modelo = crear_modelo(X).fit(X, [0, 1, 0, 1])
        numeros = modelo["preprocesar"].named_transformers_["numeros"]
        np.testing.assert_array_equal(numeros["imputar"].statistics_, [2.0])
        nuevos = pd.DataFrame({"balance": [999999.0], "job": ["nuevo"]})
        self.assertEqual(modelo.predict(nuevos).shape, (1,))
        np.testing.assert_array_equal(numeros["imputar"].statistics_, [2.0])

    def test_metricas_iguales_a_la_version_anterior(self):
        archivo = Path(__file__).parent / "fixtures" / "resultados_antes.csv"
        anteriores = pd.read_csv(archivo).iloc[:6]
        X_train, X_valid, y_train, y_valid, _ = self.particion
        for _, corrida in anteriores.iterrows():
            with self.subTest(corrida=corrida["corrida"]):
                modelo = crear_modelo(
                    X_train,
                    c=1.0 if pd.isna(corrida["C"]) else corrida["C"],
                    class_weight=corrida["class_weight"],
                    fit_intercept=str(corrida["fit_intercept"]).lower()
                    == "true",
                    seed=42,
                    modelo=corrida["modelo"],
                )
                modelo.fit(X_train, y_train)
                metricas, _ = evaluar_modelo(modelo, X_valid, y_valid)
                for nombre, valor in metricas.items():
                    self.assertAlmostEqual(valor, corrida[nombre], places=12)


if __name__ == "__main__":
    unittest.main()
