from tensorflow.keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ghDigitReco_10072025_1.h5")

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print("Model load error:", e)
    model = None
