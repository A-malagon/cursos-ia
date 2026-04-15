# Semana 3 — AKS + CI/CD + MLOps Stack completo

> "Kubernetes en Azure + automatización end-to-end + herramientas MLOps profesionales"

## Días

| Día | Tema | Estado |
|-----|------|--------|
| 15 | AKS: cluster creation, node pools, autoscaling, integración con ACR | ⬜ |
| 15b | Kubeflow conceptual + OpenShift AI conceptual (cuándo se usan vs Airflow+MLflow) | ⬜ |
| 16 | Azure DevOps Pipelines vs GitHub Actions (comparativa con Jenkins) | ⬜ |
| 17 | CI/CD para modelos ML + Great Expectations (validación de datos) | ⬜ |
| 17b | Airflow: DAGs, tasks, dependencias, scheduling — orquestar pipeline ML | ⬜ |
| 18 | Azure ML: workspaces, compute clusters, environments, model registry | ⬜ |
| 19 | MLflow en Azure: tracking, model registry, experimentos + DVC avanzado | ⬜ |
| 20 | Proyecto: Pipeline CI/CD completo con validación, tracking y despliegue en AKS | ⬜ |
| 21 | Test semana 3 + repaso | ⬜ |

**Checkpoint:** Pipeline CI/CD funcional — código → validación datos → entrenamiento → MLflow tracking → model registry → despliegue en AKS. Airflow orquestando el pipeline de datos.

---

## Día 15b — Kubeflow y OpenShift AI (conceptual)

### Kubeflow

**Qué es:** Plataforma MLOps completa que corre sobre Kubernetes. Incluye en un solo stack lo que nosotros construimos por separado: pipelines de entrenamiento, notebook servers, model serving, hyperparameter tuning.

**Cuándo usarlo vs Airflow + MLflow:**
| | Airflow + MLflow | Kubeflow |
|---|---|---|
| Complejidad | Baja-media | Alta |
| Flexibilidad | Alta | Media |
| Caso de uso | Pipelines de datos + tracking | Plataforma ML unificada enterprise |
| Curva de aprendizaje | Gradual | Pronunciada |
| Típico en | Startups, equipos ágiles | Grandes corporaciones con K8s propio |

**En Kyndryl:** probablemente verás Airflow + MLflow más que Kubeflow. Pero en clientes que ya tienen Kubeflow desplegado necesitas saber orientarte. El concepto es el mismo — es un orquestador de pipelines ML sobre K8s.

---

### OpenShift AI (Red Hat)

**Qué es:** Plataforma ML de Red Hat construida sobre OpenShift (la distribución enterprise de Kubernetes de Red Hat). Incluye Jupyter notebooks gestionados, pipelines, model serving y monitorización.

**Por qué importa para Kyndryl:**
Muchos clientes enterprise de banca y telco ya tienen OpenShift desplegado en producción (es el estándar en muchas grandes corporaciones europeas). Cuando Kyndryl lleva IA a esos clientes, lo despliega sobre OpenShift AI en lugar de AKS.

**Cómo encaja con lo que ya sabes:**
```
OpenShift = Kubernetes + capa enterprise de Red Hat
OpenShift AI = Kubeflow adaptado para OpenShift

Si sabes K8s + Helm + MLflow  →  OpenShift AI es familiar en 1-2 semanas
```

**No necesitas practicarlo** — no hay free tier accesible. Pero si en Kyndryl un cliente lo menciona, ya sabes situarte.

---

## Día 17 — CI/CD para modelos ML + Great Expectations

### ¿Qué es Great Expectations?
Librería Python para validar datos automáticamente antes de entrenar. Defines "expectativas" sobre tus datos (rangos de valores, tipos, distribuciones) y las verificas en cada ejecución del pipeline.

**Por qué importa en MLOps:**
Si los datos de entrada cambian silenciosamente (columna nueva, valores nulos inesperados, rango diferente), el modelo puede entrenarse sin errores técnicos pero producir predicciones incorrectas. Great Expectations detiene el pipeline antes de que eso ocurra.

```python
import great_expectations as ge

df = ge.read_csv("datos.csv")
df.expect_column_values_to_not_be_null("edad")
df.expect_column_values_to_be_between("edad", 0, 120)
df.expect_column_values_to_be_in_set("genero", ["M", "F", "otro"])

results = df.validate()
if not results["success"]:
    raise ValueError("Los datos no cumplen las expectativas — pipeline detenido")
```

**Dónde encaja en el pipeline:**
```
GitHub push
    ↓
GitHub Actions
    ↓
Paso 1: Great Expectations valida datos  ← si falla, para aquí
    ↓
Paso 2: Entrena modelo
    ↓
Paso 3: Evalúa métricas (si bajan del umbral, para aquí)
    ↓
Paso 4: Registra en MLflow model registry
    ↓
Paso 5: Despliega en AKS
```

---

## Día 17b — Airflow: orquestación de pipelines ML

### ¿Qué es Airflow?
Orquestador de workflows definidos como DAGs (Directed Acyclic Graphs). Cada nodo del grafo es una tarea, cada arista es una dependencia. Airflow gestiona el scheduling, los reintentos y las dependencias entre tareas.

**Analogía con tu experiencia:** Es como un pipeline Jenkins pero para datos — con UI visual, dependencias complejas y scheduling nativo.

**Diferencia Airflow vs CI/CD:**
- **GitHub Actions/Jenkins** → se dispara por eventos de código (push, PR)
- **Airflow** → se dispara por tiempo o eventos de datos (nuevos ficheros en S3/Blob, etc.)

**DAG típico de ML:**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG("pipeline_ml", schedule_interval="@daily", start_date=datetime(2026,1,1)) as dag:

    ingesta = PythonOperator(
        task_id="ingestar_datos",
        python_callable=ingestar_datos_nuevos
    )

    validacion = PythonOperator(
        task_id="validar_datos",
        python_callable=validar_con_great_expectations
    )

    entrenamiento = PythonOperator(
        task_id="entrenar_modelo",
        python_callable=entrenar_y_registrar_mlflow
    )

    ingesta >> validacion >> entrenamiento  # define dependencias
```

**Instalación local:**
```bash
pip install apache-airflow
airflow db init
airflow webserver --port 8080  # UI en http://localhost:8080
airflow scheduler
```

**Cómo encaja con el resto del stack:**
```
Airflow (orquesta cuándo y en qué orden)
    ↓
Great Expectations (valida los datos)
    ↓
Entrenamiento Python + scikit-learn
    ↓
MLflow (trackea el experimento)
    ↓
GitHub Actions (despliega si el modelo es mejor)
    ↓
AKS (corre el modelo en producción)
```

---

## Día 19 — MLflow + DVC avanzado

### DVC (Data Version Control)

**¿Qué es?** Git para datos y modelos. Git no puede versionar ficheros grandes (datasets de GBs, modelos de 100MB+). DVC los versiona en un storage externo (Azure Blob, S3, GCS) y guarda solo los metadatos en Git.

**Por qué es crítico para reproducibilidad:**
Sin DVC no puedes responder: "¿con qué datos exactamente se entrenó el modelo que está en producción ahora?". Con DVC sí — haces `dvc checkout` del commit correspondiente y tienes exactamente esos datos.

**Uso básico:**
```bash
# Inicializar DVC en el repo
dvc init

# Añadir dataset al tracking de DVC (no a Git)
dvc add data/train.csv           # crea data/train.csv.dvc
git add data/train.csv.dvc .gitignore
git commit -m "track dataset v1"

# Configurar remote storage (Azure Blob)
dvc remote add -d azure_storage azure://mi-container/dvc
dvc push                          # sube el dataset a Azure Blob

# Cuando alguien clona el repo
git clone ...
dvc pull                          # descarga los datos desde Azure Blob
```

**Integración con MLflow:**
```python
import mlflow
import dvc.api

# En el script de entrenamiento
with mlflow.start_run():
    # Log qué versión del dataset se usó
    mlflow.log_param("dataset_version", "v1.2")
    mlflow.log_param("dataset_hash", dvc.api.get_url("data/train.csv"))

    # Entrenar modelo...
    mlflow.log_metric("accuracy", 0.94)
    mlflow.sklearn.log_model(model, "model")
```

---

<!-- Más contenido se añade conforme se completan los días -->
