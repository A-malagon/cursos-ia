# Semana 1 — Docker + Fundamentos AWS para MLOps

> "Contenedores como unidad de despliegue"

## Días

| Día | Tema | Estado |
|-----|------|--------|
| 1 | Docker: conceptos, arquitectura, VM vs contenedor | ✅ Completado |
| 2 | Dockerfile, imágenes, capas, build y optimización | ✅ Completado |
| 3 | Docker Compose, redes, volúmenes, multi-container + FastAPI model serving | ⏳ En curso |
| 4 | Azure fundamentos: Resource Groups, AKS, ACR, Azure ML overview | ⬜ Pendiente |
| 5 | Azure CLI + Terraform provider Azure | ⬜ Pendiente |
| 5b | Python ML: scikit-learn + PyTorch básico — entrenar modelos para dockerizar | ⬜ Pendiente |
| 6 | Proyecto: modelo ML (sklearn) + FastAPI + Docker + MLflow local + push a ACR | ⬜ Pendiente |
| 7 | Test semana 1 + DVC básico (versionar dataset y modelo) | ⬜ Pendiente |

---

## Día 1 — Docker: conceptos, arquitectura, VM vs contenedor

### ¿Qué problema resuelve Docker?

El problema clásico: **"en mi máquina funciona"**.

Con VMs lo solucionabas virtualizando el hardware completo. Con Docker lo solucionas virtualizando solo el sistema de ficheros y procesos, **compartiendo el kernel del host**.

```
VM:                          Contenedor Docker:
┌─────────────────┐          ┌─────────────────┐
│   Tu App        │          │   Tu App        │
│   Librerías     │          │   Librerías     │
│   OS completo   │          │   (sin OS)      │
│   (2-10 GB)     │          │   (50-200 MB)   │
├─────────────────┤          ├─────────────────┤
│   Hypervisor    │          │   Docker Engine │
├─────────────────┤          ├─────────────────┤
│   Hardware      │          │   Kernel Host   │
└─────────────────┘          └─────────────────┘
```

---

### Conceptos clave

| Concepto | Qué es | Analogía AWS |
|----------|--------|--------------|
| **Image** | Plantilla inmutable (solo lectura) | AMI |
| **Container** | Instancia en ejecución de una imagen | EC2 instance |
| **Dockerfile** | Receta para construir una imagen | Script de provisioning |
| **Registry** | Almacén de imágenes | ECR / S3 |
| **Docker Engine** | Daemon que gestiona todo | El hypervisor |

---

### Arquitectura de Docker

```
┌─────────────────────────────────────────────┐
│              Docker CLI (tú)                │
│         docker build / run / push           │
└─────────────────┬───────────────────────────┘
                  │ REST API
┌─────────────────▼───────────────────────────┐
│           Docker Daemon (dockerd)           │
│   gestiona imágenes, contenedores, redes    │
└──────┬──────────────────────────┬───────────┘
       │                          │
┌──────▼──────┐           ┌───────▼──────┐
│  containerd │           │   Registry   │
│ (runtime)   │           │  (ECR/Hub)   │
└──────┬──────┘           └──────────────┘
       │
┌──────▼──────┐
│    runc     │
│ (OS-level)  │
└─────────────┘
```

---

### Ejercicio 1 — Verifica tu instalación

```bash
docker --version
docker run hello-world
```

**Output esperado de `hello-world`:**
```
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

---

### Ejercicio 2 — Explora tu primer contenedor

```bash
# Arranca un contenedor Ubuntu interactivo
docker run -it ubuntu bash

# Dentro del contenedor:
cat /etc/os-release   # ves el OS del contenedor
ls /                  # filesystem aislado
exit                  # sales, el contenedor se detiene
```

```bash
# Lista contenedores (activos e histórico)
docker ps        # solo los que están corriendo
docker ps -a     # todos, incluyendo los parados
```

---

### Comandos básicos — Cheatsheet Día 1

| Comando | Qué hace |
|---------|----------|
| `docker pull <imagen>` | Descarga imagen del registry |
| `docker run <imagen>` | Crea y arranca un contenedor |
| `docker run -it <imagen> bash` | Contenedor interactivo |
| `docker run -d <imagen>` | Contenedor en background (detached) |
| `docker ps` | Lista contenedores activos |
| `docker ps -a` | Lista todos los contenedores |
| `docker stop <id>` | Para un contenedor |
| `docker rm <id>` | Elimina un contenedor parado |
| `docker images` | Lista imágenes locales |
| `docker rmi <imagen>` | Elimina una imagen local |

---

### Mini-test Día 1 — Respuestas

1. **Imagen vs Contenedor:** la imagen es la plantilla estática (como una AMI), no ejecuta nada. El contenedor es la instancia en ejecución. De una imagen puedes crear N contenedores distintos.
2. **Equivalente al hypervisor:** el Docker Engine (dockerd).
3. **`docker run ubuntu` dos veces:** dos contenedores distintos, cada uno con su propio ID.
4. **Por qué pesa menos:** no incluye OS completo, comparte el kernel del host. Solo empaqueta app + librerías.
5. **Registry:** almacén de imágenes. Público = Docker Hub. En AWS = ECR (Elastic Container Registry).

---

### Notas del ejercicio `docker ps -a`

```
CONTAINER ID   IMAGE         COMMAND    STATUS
d825f199b6f7   ubuntu        "bash"     Exited (0)
a75b82bef534   hello-world   "/hello"   Exited (0)
```

- `COMMAND` muestra qué ejecutó cada contenedor: `bash` vs `/hello`
- `Exited (0)` = terminaron correctamente (código 0 = sin errores)
- Docker asigna nombres aleatorios si no usas `--name` → en producción siempre nombrar explícitamente

---

## Día 2 — Dockerfile, imágenes, capas, build y optimización

### El modelo mental clave: capas como commits de Git

Cada instrucción del Dockerfile crea una **capa** (layer) — un snapshot incremental de la imagen:

```
Instrucción                   Capa resultante
─────────────────             ──────────────────────────────
FROM python:3.11-slim    →    sha256:a1b2... (imagen base, ~130MB)
RUN pip install flask    →    sha256:c3d4... (+100MB)
COPY app.py /app/        →    sha256:e5f6... (+12KB)
CMD ["python", "app.py"] →    sha256:g7h8... (0B, solo metadata)
```

Las capas se **cachean**. Si no cambia `requirements.txt`, Docker reutiliza la capa del `pip install` → build en segundos en lugar de minutos.

---

### Instrucciones del Dockerfile

| Instrucción | Qué hace | Cuándo usarla |
|-------------|----------|---------------|
| `FROM` | Imagen base | Siempre, primera línea |
| `WORKDIR` | Establece directorio de trabajo | Antes de cualquier COPY o RUN |
| `COPY` | Copia archivos del host a la imagen | Tu código y configs |
| `RUN` | Ejecuta comando en build time | Instalar dependencias |
| `ENV` | Variable de entorno | Config de la app |
| `ARG` | Variable solo en build time | Versiones, secrets de build |
| `EXPOSE` | Documenta el puerto (no lo abre) | Solo documentación |
| `CMD` | Comando por defecto al arrancar | Arranque de la app |
| `ENTRYPOINT` | Como CMD pero no se sobreescribe | Ejecutables/scripts fijos |

---

### El orden de las capas importa — regla de oro

**MAL** — invalida caché en cada cambio de código:
```dockerfile
COPY . .                          # copia TODO incluyendo app.py
RUN pip install -r requirements.txt  # se re-ejecuta aunque solo cambió app.py
```

**BIEN** — dependencias cacheadas:
```dockerfile
COPY requirements.txt .           # solo el requirements
RUN pip install -r requirements.txt  # cacheado mientras no cambie
COPY app.py .                     # código al final, cambia más frecuentemente
```

**Regla:** lo que cambia menos frecuentemente → arriba. Lo que cambia más → abajo.

---

### Archivos del ejercicio

**`requirements.txt`**
```
flask==3.0.0
numpy==1.26.2
```
Versiones fijas (`==`) para garantizar reproducibilidad. Sin pinear versiones, un build en 6 meses puede traer versiones incompatibles y romper el modelo en producción.

**`.dockerignore`**
```
__pycache__/
*.pyc
.env
.git/
```
Excluye archivos del build context antes de mandarlo al daemon. Sin él Docker empaqueta todo el directorio, incluyendo `.git` y potenciales credenciales en `.env`.

**`app.py`** — API Flask con dos endpoints:
```python
from flask import Flask, jsonify

app = Flask(__name__)

def predict(x):
    return float(2 * x + 1)

@app.route("/predict/<float:x>")
def make_prediction(x):
    result = predict(x)
    return jsonify({"input": x, "prediction": result})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```
`host="0.0.0.0"` es obligatorio — sin él Flask solo escucha dentro del contenedor y no es accesible desde el host.

**`Dockerfile`**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
```

---

### Comandos del día

```bash
# Construir la imagen
docker build -t ml-api:v1 .

# Ver las capas y sus tamaños
docker history ml-api:v1

# Arrancar el contenedor en background con mapeo de puertos
docker run -d -p 5000:5000 --name ml-api ml-api:v1

# Probar los endpoints
curl http://localhost:5000/health
curl http://localhost:5000/predict/5.0   # → {"prediction": 11.0}

# Ver logs del contenedor
docker logs ml-api

# Parar y eliminar
docker stop ml-api && docker rm ml-api
```

---

### Análisis de capas — `docker history ml-api:v1`

```
SIZE      CAPA
0B        CMD ["python" "app.py"]          ← metadata, no pesa
0B        EXPOSE 5000                       ← metadata, no pesa
12.3kB    COPY app.py                       ← tu código, insignificante
99.9MB    RUN pip install flask + numpy     ← el peso real, por eso se cachea
12.3kB    COPY requirements.txt
8.19kB    WORKDIR /app
48.4MB    Python compilado (imagen base)
87.4MB    Debian base
```

---

### Demostración de caché

- **v1** (build limpio): **38.6 segundos**
- **v2** (solo cambió `app.py`): **2.8 segundos**

```
CACHED [2/5] WORKDIR /app
CACHED [3/5] COPY requirements.txt .
CACHED [4/5] RUN pip install ...      ← 100MB, 0 segundos gracias a caché
[5/5] COPY app.py .                   ← único step ejecutado realmente
```

En producción con `torch` o `tensorflow` (1-3GB de deps) esto es la diferencia entre un pipeline CI/CD de 3 minutos vs 30 minutos.

---

### Mini-test Día 2 — Respuestas

1. **`CMD` vs `ENTRYPOINT`:**
   - `CMD` define el comando por defecto pero se puede sobreescribir al hacer `docker run ml-api:v1 python otro.py`
   - `ENTRYPOINT` define el ejecutable fijo del contenedor, no se sobreescribe fácilmente. Se usa cuando el contenedor es un ejecutable en sí mismo (ej: un script de entrenamiento)
   - En la práctica: `ENTRYPOINT` para el binario, `CMD` para los argumentos por defecto

2. **Por qué `EXPOSE` pesa 0B:** es metadata pura. Solo documenta qué puerto usa la app, no abre ni mapea ningún puerto. El mapeo real ocurre en `docker run -p`. Es útil para que otros developers y herramientas sepan qué puerto expone el contenedor.

3. **Si añades `scipy` al `requirements.txt` y haces build:** NO se cachea el `pip install`. Docker detecta que el fichero `requirements.txt` cambió → invalida esa capa y todas las siguientes. Se reinstalan todas las dependencias desde cero.

4. **`docker run` sin `-p`:** el contenedor arranca y Flask escucha en el puerto 5000 **dentro** del contenedor, pero ese puerto no está expuesto al host. No puedes acceder desde `localhost:5000`. El contenedor está aislado en su propia red.

5. **`--no-cache-dir` en pip:** le dice a pip que no guarde la caché de paquetes descargados en disco. En un entorno normal esa caché acelera reinstalaciones, pero en Docker no tiene sentido — cada build parte de cero. Sin este flag la imagen ocupa más MB innecesariamente (la caché de pip queda dentro de la capa).

---

## Día 5b — Python ML: scikit-learn + PyTorch básico

> "No necesitas ser data scientist, pero sí saber entrenar un modelo para poder desplegarlo"

### Por qué aprenderlo

En Kyndryl trabajarás con modelos que otros construyeron. Para dockerizarlos, versionar con DVC, trackear con MLflow y desplegarlos en AKS necesitas entender mínimamente cómo se entrenan y qué producen. Sin eso es como desplegar una app sin saber qué hace.

---

### scikit-learn — ML clásico

El framework más usado para modelos de ML "tradicionales" (clasificación, regresión, clustering). La mayoría de modelos en producción en banca y seguros son scikit-learn, no deep learning.

**Flujo básico — siempre igual:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. Cargar datos
df = pd.read_csv("datos.csv")
X = df.drop("target", axis=1)   # features
y = df["target"]                 # variable objetivo

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Entrenar
modelo = RandomForestClassifier(n_estimators=100)
modelo.fit(X_train, y_train)

# 4. Evaluar
predicciones = modelo.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predicciones):.2f}")

# 5. Guardar modelo
import joblib
joblib.dump(modelo, "modelo.pkl")
```

**Cargar y servir el modelo guardado (en FastAPI):**
```python
import joblib
from fastapi import FastAPI

app = FastAPI()
modelo = joblib.load("modelo.pkl")

@app.post("/predict")
def predict(features: dict):
    import numpy as np
    X = np.array(list(features.values())).reshape(1, -1)
    prediccion = modelo.predict(X)
    return {"prediction": int(prediccion[0])}
```

**Modelos más usados en producción:**
| Modelo | Cuándo usarlo |
|--------|---------------|
| `RandomForestClassifier` | Clasificación robusta, muchas features |
| `GradientBoostingRegressor` / XGBoost | Regresión de alta precisión |
| `LogisticRegression` | Clasificación binaria interpretable |
| `KMeans` | Clustering, segmentación de clientes |

---

### PyTorch — Deep Learning básico

No necesitas ser experto, pero sí entender la estructura básica para no quedarte en blanco si en Kyndryl trabajan con modelos de redes neuronales.

**Analogía:** scikit-learn es como usar Terraform con módulos ya hechos. PyTorch es como escribir el Terraform desde cero — más control, más complejidad.

**Flujo básico:**
```python
import torch
import torch.nn as nn

# 1. Definir la arquitectura de la red
class ModeloSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Linear(10, 64),   # 10 features de entrada → 64 neuronas
            nn.ReLU(),
            nn.Linear(64, 1),    # 64 neuronas → 1 salida (regresión)
        )

    def forward(self, x):
        return self.capas(x)

# 2. Instanciar y entrenar
modelo = ModeloSimple()
optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
criterio = nn.MSELoss()

for epoch in range(100):
    prediccion = modelo(X_train)
    loss = criterio(prediccion, y_train)
    optimizador.zero_grad()
    loss.backward()
    optimizador.step()

# 3. Guardar
torch.save(modelo.state_dict(), "modelo.pt")
```

**Diferencia clave scikit-learn vs PyTorch:**
| | scikit-learn | PyTorch |
|---|---|---|
| Tipo de modelo | ML clásico (árboles, regresión, SVM) | Redes neuronales (deep learning) |
| Datos | Tabular (filas y columnas) | Imágenes, texto, series temporales |
| Complejidad | Baja | Media-alta |
| Uso en producción banca | Muy frecuente | Frecuente en NLP/visión |
| Guardar modelo | `joblib.dump()` | `torch.save()` |

---

### Integración con MLflow

Tanto scikit-learn como PyTorch se trackean igual con MLflow:

```python
import mlflow
import mlflow.sklearn    # o mlflow.pytorch

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    modelo.fit(X_train, y_train)
    acc = accuracy_score(y_test, modelo.predict(X_test))

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(modelo, "model")   # guarda en el registry
```

---

