from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, json, numpy as np

app = Flask(__name__)
CORS(app)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_meta.json', 'r') as f:
    meta = json.load(f)

FEATURE_COLUMNS = meta['feature_columns']
LABEL_ENCODERS  = meta['label_encoders']

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "Employee Attrition API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        for col, mapping in LABEL_ENCODERS.items():
            if col in data:
                data[col] = mapping[str(data[col])]
        features = [float(data[col]) for col in FEATURE_COLUMNS]
        import numpy as np
        pred  = model.predict([features])[0]
        proba = model.predict_proba([features])[0][1]
        return jsonify({
            "prediction":    "Left" if pred == 1 else "Stayed",
            "probability":   round(float(proba), 4),
            "attrition_pct": round(float(proba) * 100, 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
