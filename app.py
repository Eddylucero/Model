from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model_data = joblib.load("best_model1.pkl")
model = model_data["model"]
expected_features = model_data["features"]

print(f"Modelo cargado correctamente. Espera {len(expected_features)} features.")

@app.route("/")
def home():
    return jsonify({
        "message": "API de predicción de riesgo crediticio activa",
        "expected_features": expected_features
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = data.get("features")
        if features is None:
            return jsonify({"error": "Debe enviar un JSON con la clave 'features'"}), 400

        # Convertir a DataFrame
        df = pd.DataFrame(features if isinstance(features, list) else [features])

        # Verificar columnas
        missing_cols = [c for c in expected_features if c not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Faltan columnas: {missing_cols}"}), 400

        # Asegurar el mismo orden de columnas
        df = df[expected_features]

        # Realizar predicción (0,1)
        preds = model.predict(df)

        return jsonify({
            "predictions": preds.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
