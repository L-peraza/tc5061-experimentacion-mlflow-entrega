# TC5061 · Experimentación con MLflow

**Leonardo Peraza Alavez · A01840326 · Individual**

Regresión logística para predecir si un cliente suscribe un depósito bancario. Se compara con una referencia que siempre predice la clase mayoritaria.

## Estructura

Proyecto generado con **Cookiecutter Data Science v2.3.0** y adaptado al curso. La procedencia está en [docs/cookiecutter.json](docs/cookiecutter.json). No es necesario instalar el generador para entrenar.

```text
bank_marketing/
  config.py
  dataset.py
  features.py
  tracking.py
  benchmark.py
  modeling/
    pipeline.py
    train.py
    evaluate.py
data/
  raw/
  interim/
  processed/
  external/
notebooks/1.0-lpa-exploracion.ipynb
models/
reports/
  figures/
docs/
references/
tests/
train.py
MLproject
python_env.yaml
pyproject.toml
requirements.txt
requirements-dev.txt
Makefile
LICENSE
```

`data/raw` conserva el CSV sin modificar. Las demás carpetas de datos y `models` son puntos de organización de la plantilla; este ejercicio transforma en memoria y guarda los modelos en MLflow. `notebooks` contiene solo EDA. `reports` contiene resultados y evidencias; `bank_marketing` contiene el código reutilizable. Los datos descargados, la base SQLite y el almacén de MLflow se generan localmente y no se publican en GitHub.

## Preparar el entorno

Python **3.12**; las corridas se ejecutaron con **3.12.14**. Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

En Windows, activar con `.venv\Scripts\activate`.

## Ejecutar el script

```bash
python train.py
python train.py --help
python train.py --data-path "/ruta/a/bank-full.csv"
```

El CSV debe conservar el formato de Bank Marketing: separador `;` y las mismas columnas, incluidas `y` y `duration`. Las rutas relativas parten de la carpeta desde la que se ejecuta el comando. La ruta predeterminada sigue siendo `data/raw/bank-full.csv` dentro del proyecto y se descarga desde UCI si falta. Una ruta personalizada debe existir.

| Argumento | Significado | Predeterminado |
|---|---|---|
| `--data-path` | Ubicación del CSV de Bank Marketing. | `data/raw/bank-full.csv` dentro del proyecto |
| `--c` | Inverso de la regularización; menor C significa más regularización. | `1.0` |
| `--class-weight` | Ponderar las clases o darles el mismo peso. | `balanced` |
| `--fit-intercept` | Incluir o excluir el término constante. | `true` |
| `--seed` | Semilla de las particiones, NumPy y el clasificador. | `42` |
| `--modelo` | Regresión logística o referencia mayoritaria. | `logistica` |
| `--run-name` | Nombre visible de la corrida. | Generado por MLflow |

## Ejecutar con MLflow Projects

Se utilizan los parámetros y el comando declarados en `MLproject`. `python_env.yaml` declara Python y dependencias. Los comandos verificados usan `--env-manager local`: reutilizan el entorno preparado arriba; no crean otro entorno automáticamente.

En Linux/macOS:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow run . --env-manager local --experiment-name TC5061_BankMarketing_Cookiecutter_PerazaLeonardo
```

En PowerShell, la variable se define con `$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"`.

Para usar otra ubicación del mismo CSV, agrega el parámetro `data_path`:

```bash
mlflow run . --env-manager local --experiment-name TC5061_BankMarketing_Cookiecutter_PerazaLeonardo -P data_path="/ruta con espacios/bank-full.csv" -P run_name=Ruta_personalizada
```

La ejecución anterior usa los valores predeterminados de `MLproject` y corresponde a R1. Las otras configuraciones ejecutadas fueron:

```bash
mlflow run . --env-manager local --experiment-name TC5061_BankMarketing_Cookiecutter_PerazaLeonardo -P c=0.01 -P run_name=R2_C_001
mlflow run . --env-manager local --experiment-name TC5061_BankMarketing_Cookiecutter_PerazaLeonardo -P c=100 -P run_name=R3_C_100
mlflow run . --env-manager local --experiment-name TC5061_BankMarketing_Cookiecutter_PerazaLeonardo -P class_weight=none -P run_name=R4_sin_pesos
mlflow run . --env-manager local --experiment-name TC5061_BankMarketing_Cookiecutter_PerazaLeonardo -P fit_intercept=false -P run_name=R5_sin_intercepto
mlflow run . --env-manager local --experiment-name TC5061_BankMarketing_Cookiecutter_PerazaLeonardo -P modelo=dummy -P run_name=R6_mayoritaria
python train.py
```

R1-R5 son cinco configuraciones distintas del mismo modelo. R6 es la referencia. El último comando, sin argumentos, repite R1; después se le asignó el nombre R7_repeticion en MLflow. Los comandos completos se conservan en [reports/comandos_ejecutados.txt](reports/comandos_ejecutados.txt).

## Ver resultados

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abrir http://127.0.0.1:5000 y seleccionar **TC5061_BankMarketing_Cookiecutter_PerazaLeonardo**. Elegir R1-R5 y pulsar **Compare**. Las capturas están en [reports/figures](reports/figures).

Cada corrida registra parámetros, cinco métricas `val_*`, `metricas.json`, `matriz_confusion.png`, código, configuración del proyecto y el modelo. Las nuevas ejecuciones también registran la ruta del CSV y conservan `LICENSE` con los artefactos del código. El modelo incluye el paquete con el transformador propio para poder recargarlo. Los resultados completos de R1-R7 están en [reports/resultados.csv](reports/resultados.csv).

## Decisiones y análisis

Se usa **UCI Bank Marketing, bank-full.csv**: 45,211 filas, 16 variables de entrada y la etiqueta `y`. Se excluye `duration` por conocerse después de la llamada. `BankCleaner` convierte `unknown` de ocupación, educación y canal a faltantes dentro del pipeline. Se conservan saldos negativos, extremos plausibles y `pdays=-1`.

Partición estratificada con semilla 42: **60% entrenamiento, 20% validación y 20% prueba reservada**. Imputación, escalado y codificación se ajustan solo con entrenamiento. El criterio principal es **F1 de la clase positiva**, definido antes de comparar; se mantiene el umbral 0.5.

R3, con C=100, clases balanceadas e intercepto, obtuvo F1=0.374965. Su mejora frente a R1 es apenas 0.000176. El peso de clases fue el hiperparámetro más influyente: quitar `balanced` redujo F1 a 0.294379 y recall a 18.81%. La referencia tuvo exactitud de 88.30%, pero F1 y recall cero.

Son cinco configuraciones y una sola partición. No se demuestra una ventaja estadísticamente significativa de R3. La prueba reservada no se evalúa; el dataset ya se exploró en la actividad anterior, por lo que tampoco equivale a datos nuevos de producción. Faltan validación temporal y costos de los errores para decidir un uso real.

## Pruebas, formato y rendimiento

```bash
python -m unittest discover -s tests -v
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
python -m bank_marketing.benchmark
```

Las seis pruebas cubren limpieza sin modificar la entrada, estado del transformador, esquema, particiones, imputación con entrenamiento y equivalencia de las cinco métricas con las seis configuraciones anteriores. Las referencias numéricas están en `tests/fixtures/resultados_antes.csv`. La tolerancia es 12 decimales en el mismo entorno. La repetición de corridas también se verifica en `reports/reproducibilidad.json`.

La ruta personalizada se verificó desde el script y MLflow Projects, incluyendo espacios en el nombre del archivo. Las tres corridas de comprobación terminaron correctamente y conservaron las cinco métricas de R1. También se comprobó el rechazo de una ruta inexistente. La evidencia está en [reports/verificacion_data_path.json](reports/verificacion_data_path.json); estas corridas utilizan una base SQLite de verificación independiente.

El benchmark compara limpieza con comprensión de listas frente a reemplazo vectorizado, verifica que las salidas sean iguales y registra medianas de cinco repeticiones. Un generador cuenta las líneas del CSV sin construir una lista. El resultado está en `reports/benchmark.json`; mide solo esa operación y depende del equipo.

Las explicaciones de POO, herencia, polimorfismo y separación de responsabilidades están en [docs/diseno.md](docs/diseno.md). El código Python no contiene comentarios con `#`; las explicaciones están en Markdown. Para abrir el notebook se necesita Jupyter, adicional al entorno de entrenamiento.

## Evidencia y versión de entrega

Este repositorio reúne la versión de entrega en un único commit inicial. Las capturas y los registros de MLflow se generaron antes de esta consolidación; conservan sus identificadores y el commit de origen de aquellas ejecuciones. Los resultados no se modificaron al crear el nuevo repositorio. Las mejoras posteriores de licencia y ruta del CSV se comprobaron por separado en `reports/verificacion_data_path.json`.

## Licencia

El código del proyecto se distribuye bajo la [licencia MIT](LICENSE). La licencia del dataset se consulta en su fuente de UCI.

## Fuentes

- Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- DrivenData. (s. f.). *Cookiecutter Data Science*. https://cookiecutter-data-science.drivendata.org/
- MLflow. (s. f.). *MLflow Projects*. https://mlflow.org/docs/latest/ml/projects/
- scikit-learn developers. (s. f.). *Developing scikit-learn estimators*. https://scikit-learn.org/stable/developers/develop.html
- Python Software Foundation. (s. f.). *PEP 8: Style guide for Python code*. https://peps.python.org/pep-0008/
