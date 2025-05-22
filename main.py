from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
from detector import detector
app = FastAPI()

# Load the trained autoencoder model
try:
    model = load_model("autoencoder.keras", custom_objects={"detector": detector})
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Set static anomaly detection threshold (determined from training data)
THRESHOLD = 0.044578623  # mean + std or determined via validation set
# Define request schema
class ECGInput(BaseModel):
    ecg: list[float]  # Should be exactly 140 values

@app.get("/")
def root():
    return {"message": "ECG Autoencoder API is running."}

@app.post("/predict/")
def predict_ecg(input_data: ECGInput):
    if len(input_data.ecg) != 140:
        raise HTTPException(status_code=400, detail="ECG input must contain exactly 140 values.")

    try:
        # Convert input list to numpy array
        ecg_array = np.array(input_data.ecg, dtype=np.float32).reshape(1, -1)

        # Normalize using min-max scaling (same as training)
        ecg_min = -6.2808752
        ecg_max = 7.4021031

        # Prevent division by zero in case of constant signal
        if ecg_max == ecg_min:
            norm_ecg = np.zeros_like(ecg_array)
        else:
            norm_ecg = (ecg_array - ecg_min) / (ecg_max - ecg_min)
        print(norm_ecg)
        # Predict and calculate reconstruction error (MAE)
        def prediction(model, data, threshold):
            rec = model(data)
            loss = tf.keras.losses.mae(rec, data)

            return tf.math.less(loss, threshold),loss
        
        pred,loss = prediction(model, norm_ecg,THRESHOLD)

        status = "Normal" if pred else "Anomaly"

        return {
            "status": status,
            "loss": float(loss),
            "threshold": THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error during prediction: {str(e)}")
