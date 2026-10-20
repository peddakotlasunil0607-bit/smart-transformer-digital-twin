
import csv
from io import StringIO
from flask import Response, Flask, jsonify, request
import joblib
import pandas as pd
import json
import os
from datetime import datetime

app = Flask(__name__, static_folder="Static")

model = joblib.load("transformer_model.pkl")

HISTORY_FILE = "prediction_history.json"

FEATURES = [
    "location", "power", "self_protection", "avg_earth_ddt",
    "max_earth_ddt", "burning_rate", "criticality_ceramics",
    "removable_connectors", "num_users", "eens_kwh",
    "installation_type", "air_network", "circuit_queue",
    "network_km_lt", "year", "client_type_COMERCIAL",
    "client_type_HOUSEHOLD", "client_type_INDUSTRIAL",
    "client_type_OFICIAL",
]

THRESHOLD = 0.10


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


@app.route("/")
def home():
    return app.send_static_file("dashboard.html")


@app.route("/health")
def health():
    return jsonify({
        "project": "Smart Transformer Digital Twin",
        "status": "Running",
        "model_loaded": True
    })


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(load_history())


@app.route("/history/download", methods=["GET"])
def download_history():
    history = load_history()
    if not history:
        return jsonify({"error": "No prediction history to download."}), 404

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(history[0].keys()))
    writer.writeheader()
    writer.writerows(history)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=transformer_prediction_history.csv"
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a JSON object containing the features."
        }), 400

    missing = [feature for feature in FEATURES if feature not in data]
    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing
        }), 400

    try:
        input_data = pd.DataFrame(
            [[data[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        probability = float(model.predict_proba(input_data)[0][1])
        flagged = int(probability >= THRESHOLD)

        result = {
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "failure_probability_percent": round(probability * 100, 6),
            "predicted_failure": flagged,
            "threshold_percent": THRESHOLD * 100,
            "power": data["power"],
            "num_users": data["num_users"],
            "avg_earth_ddt": data["avg_earth_ddt"],
            "max_earth_ddt": data["max_earth_ddt"],
        }

        history = load_history()
        history.insert(0, result)
        save_history(history[:100])

        return jsonify({
            **result,
            "threshold": THRESHOLD,
            "note": (
                "Experimental model output, not a validated "
                "real-world diagnosis."
            )
        })

    except (TypeError, ValueError, KeyError, OSError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(debug=True)
