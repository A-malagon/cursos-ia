import redis # type: ignore
from flask import Flask, jsonify # type: ignore

r = redis.Redis(host="redis", port=6379)
app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"status":"healthy"})

@app.route("/predict/<float:x>")
def predict(x):
    resultado = (2 * x) +1
    cached = r.get(str(x))       # busca en Redis
    if cached:
        return jsonify({"input": x, "prediction": float(cached), "cache": True})
    r.set(str(x), resultado)     # guarda en Redis
    return jsonify({"input": x, "prediction": resultado, "cache": False})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)