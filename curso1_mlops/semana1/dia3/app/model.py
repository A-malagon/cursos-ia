"""
Entrena un LinearRegression sobre datos sintéticos y guarda modelo.pkl.
Se ejecuta automáticamente como paso del Dockerfile (RUN python model.py).
En MLOps real, este script lo lanzaría un pipeline separado (AzureML, Kubeflow...).
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

rng = np.random.default_rng(42)
X = rng.uniform(0, 100, size=(500, 1))
y = 3 * X.squeeze() + 7 + rng.normal(0, 2, size=500)

model = LinearRegression()
model.fit(X, y)

output = os.environ.get("MODEL_PATH", "modelo.pkl")
joblib.dump(model, output)
print(f"Modelo guardado → {output}")
print(f"  coef={model.coef_[0]:.4f}  intercept={model.intercept_:.4f}")
