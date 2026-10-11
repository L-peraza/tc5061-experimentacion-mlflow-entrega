# Diseño y relación con la clase

## 1. Estructura Cookiecutter

Se generó la plantilla oficial Cookiecutter Data Science v2.3.0 con `ccds` y luego se adaptó el proyecto existente. Se conservan los directorios de datos, notebooks, código, modelos, informes, referencias y documentación. `docs/cookiecutter.json` registra versión y commit de la plantilla. El punto de entrada `train.py` delega en el módulo de entrenamiento. La raíz incluye README, Makefile y una licencia MIT en `LICENSE`.

## 2. Objetos, atributos y métodos

`BankCleaner` es una clase de preprocesamiento. Cada pipeline crea una instancia independiente. Su atributo `unknown_value` define el valor que se considera desconocido. Sus métodos `fit` y `transform` separan ajuste y aplicación.

## 3. self, __init__ y estado

`__init__` guarda solamente la configuración en `self.unknown_value`. `fit` guarda el esquema observado en entrenamiento: `feature_names_in_`, `n_features_in_` y `unknown_columns_`. El sufijo `_` identifica estado ajustado. `transform` comprueba que exista ese estado y rechaza columnas distintas. Las pruebas verifican ambos casos.

## 4. Herencia y polimorfismo

La clase hereda de `TransformerMixin` y `BaseEstimator`. Reutiliza `fit_transform`, `get_params` y `set_params`; define su comportamiento con `fit`, `transform` y `get_feature_names_out`. El pipeline puede usarla mediante la misma interfaz que los transformadores de scikit-learn. Eso demuestra polimorfismo sin añadir una jerarquía de clases innecesaria. Regresión logística y DummyClassifier también se evalúan mediante `predict` y `predict_proba`.

## 5. Aplicación en Machine Learning

El pipeline ejecuta limpieza, imputación, escalado, codificación y clasificación. `BankCleaner` convierte `unknown` en faltantes únicamente en ocupación, educación y canal. No aprende estadísticas de validación. Las medianas y el escalado se ajustan con entrenamiento. Excluir `duration` y eliminar duplicados son reglas deterministas de carga.

## 6. Código legible

Las funciones tienen nombres descriptivos y una responsabilidad concreta. Ruff comprueba imports, errores comunes, estilo y líneas de hasta 79 caracteres. Las explicaciones permanecen en este Markdown y en el README; no hay comentarios con `#` dentro del código Python.

## 7. Refactorización segura

Antes de reorganizar se conservaron las métricas de seis configuraciones. La prueba de regresión vuelve a entrenarlas y compara las cinco métricas a 12 decimales. Las demás pruebas verifican estado, ausencia de modificación de datos, esquema, particiones e imputación. La repetición registrada en MLflow verifica además la ejecución completa. Esta evidencia protege el comportamiento del flujo al cambiar su organización.

## 8. Cohesión y DRY

`dataset.py` descarga y divide; `features.py` limpia; `pipeline.py` construye el modelo; `evaluate.py` calcula métricas; `tracking.py` registra artefactos. `modeling/train.py` coordina la ejecución. El notebook importa la carga y división, sin copiar su lógica ni entrenar modelos. La configuración común vive en `config.py`.

## 9. Python eficiente

`benchmark.py` compara una comprensión de listas con el reemplazo vectorizado de pandas. Primero comprueba equivalencia y luego mide con `timeit`, cinco repeticiones de tres ejecuciones. Reporta la mediana por llamada. Usa una comprensión de diccionario para resumir tiempos y un generador para contar registros del CSV. El resultado no se generaliza al tiempo de entrenamiento ni a otros equipos.

## 10. MLflow Projects

`MLproject` declara un punto de entrada con tres hiperparámetros, semilla, tipo de modelo y nombre de corrida. La ubicación del CSV se recibe con `--data-path` en el script o `-P data_path=...` en MLflow Projects. La ruta estándar se conserva por defecto; una ubicación personalizada permite mover el mismo dataset sin editar el código. `python_env.yaml` declara Python 3.12.14 y las dependencias. Se verifica `mlflow run . --env-manager local`, que ejecuta ese punto de entrada en el entorno previamente preparado. No se afirma haber probado la creación automática de un entorno aislado. Los modelos se guardan con el código de `bank_marketing` para incluir la clase propia al recargarlos.

## Límites

La estructura y las pruebas facilitan reproducir el ejercicio, pero no validan un despliegue real. La comparación sigue usando datos históricos, una sola partición y umbral fijo. La prueba reservada permanece sin evaluar.
