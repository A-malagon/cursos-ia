# Semana 4 — Observabilidad + Proyecto Final

> "Operar ML en producción con garantías"

## Días

| Día | Tema | Estado |
|-----|------|--------|
| 22 | Monitorización de modelos: data drift, model drift + Evidently | ⬜ |
| 23 | Prometheus + Grafana en AKS + dashboard Evidently integrado | ⬜ |
| 24 | Logging centralizado: Azure Monitor, Log Analytics | ⬜ |
| 25 | Gobierno ML: versionado, rollback, A/B testing, canary deploys | ⬜ |
| 26 | IaC completo: Terraform para todo el stack AKS + ACR + Azure ML | ⬜ |
| 27-28 | PROYECTO FINAL: Plataforma MLOps end-to-end en Azure | ⬜ |
| 29 | Test final global + simulacro entrevista técnica | ⬜ |
| 30 | Repaso de gaps detectados + preparación presentación del proyecto | ⬜ |

**Checkpoint final:** Plataforma MLOps completa — modelo en AKS, pipeline Airflow, MLflow tracking, Evidently monitoring, Prometheus+Grafana, todo provisionado con Terraform.

---

## Día 22 — Monitorización de modelos + Evidently

### Los dos tipos de drift

**Data drift** — los datos de entrada cambian estadísticamente respecto al entrenamiento.
- Ejemplo: tu modelo de scoring de crédito fue entrenado con datos de 2023. En 2025 los salarios medios han subido. La distribución de la feature "salario" es completamente diferente.
- El modelo no falla — devuelve predicciones — pero cada vez menos fiables.

**Model drift (concept drift)** — la relación entre features y target cambia.
- Ejemplo: un modelo de detección de fraude aprende los patrones de fraude de 2023. Los defraudadores cambian sus técnicas. El modelo sigue funcionando técnicamente pero el fraude ya no sigue los patrones que aprendió.

### Evidently — detección automática de drift

**¿Qué es?** Librería Python open source para monitorización de ML en producción. Genera reportes de drift, calidad de datos y performance del modelo.

**Instalación:**
```bash
pip install evidently
```

**Uso básico — detectar data drift:**
```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Datos de referencia (con los que se entrenó el modelo)
referencia = pd.read_csv("datos_entrenamiento.csv")

# Datos actuales en producción
produccion = pd.read_csv("datos_produccion_ultima_semana.csv")

# Generar reporte
reporte = Report(metrics=[DataDriftPreset()])
reporte.run(reference_data=referencia, current_data=produccion)
reporte.save_html("reporte_drift.html")  # reporte visual
```

**Integración con el pipeline completo:**
```python
from evidently.metric_preset import DataDriftPreset, ModelPerformancePreset
from evidently.report import Report

def monitorizar_modelo_semanal():
    reporte = Report(metrics=[
        DataDriftPreset(),           # ¿cambiaron los datos de entrada?
        ModelPerformancePreset()     # ¿bajó el accuracy/precision?
    ])
    reporte.run(reference_data=ref, current_data=prod)

    # Si hay drift significativo, disparar reentrenamiento
    resultados = reporte.as_dict()
    drift_detectado = resultados["metrics"][0]["result"]["dataset_drift"]

    if drift_detectado:
        disparar_pipeline_reentrenamiento()  # llama a Airflow DAG
```

**Cómo encaja con Prometheus + Grafana:**
```
Evidently genera métricas de drift
    ↓
Expone métricas en formato Prometheus
    ↓
Prometheus las recoge cada X minutos
    ↓
Grafana las muestra en dashboard
    ↓
Alerta si el drift supera umbral
```

---

## Proyecto Final — Plataforma MLOps end-to-end

```
GitHub Repo (código + DVC metadata)
    ↓
GitHub Actions CI/CD
    ↓
Great Expectations (valida datos)
    ↓
Airflow DAG (orquesta pipeline)
    ↓
Entrenar modelo scikit-learn
    ↓
MLflow tracking + model registry
    ↓
Build Docker imagen (FastAPI serving)
    ↓
Push a ACR (Azure Container Registry)
    ↓
Deploy en AKS (con Helm)
    ↓
Evidently (monitorización drift)
    ↓
Prometheus + Grafana (observabilidad)
    ↓
Terraform (toda la infra como código)
```

---

<!-- El contenido de cada día se añade aquí conforme se completa -->
