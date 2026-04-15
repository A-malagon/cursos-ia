from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"status":"ok"})

@app.route("/predict/<float:x>")
def predict(x):
    resultado = (2 * x) +1
    return jsonify({"input":x,"prediction": resultado})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)