import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown

# Load your trained model

MODEL_PATH = "ghDigitReco_10072025_1.h5"

def load_model_safely():
    try:
        if not os.path.exists(MODEL_PATH):
            print("Downloading model from Google Drive...")
            url = "https://drive.google.com/uc?id=1dkED1uKIq1iBjq2I4TkjzMxHGTsKU4F8"
            gdown.download(url, MODEL_PATH, quiet=False)

        if not os.path.exists(MODEL_PATH):
            print("Model file still not found after download.")
            return None

        print("Loading model...")
        return tf.keras.models.load_model(MODEL_PATH)

    except Exception as e:
        print("MODEL LOAD ERROR:", str(e))
        return None

model = load_model_safely()

# Preprocessing (same as tkinter version)
def preprocess_like_tkinter(img_data):
    try:
        if img_data is None:
            return None
        if isinstance(img_data, dict) and "composite" in img_data:
            arr = img_data["composite"]
        else:
            arr = img_data
        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[:, :, :3]  # Drop alpha channel
        pil_img = Image.fromarray(arr.astype("uint8")).convert("L").resize((128, 128))
        img_array = tf.keras.utils.img_to_array(pil_img)
        img_array = np.expand_dims(img_array, axis=0).astype("float32")
        return img_array
    except Exception:
        traceback.print_exc()
        return None

# Recognition function
def recognize(img_data):
    try:
        if model is None:
            return "### ⚠️ Model failed to load on server", None

        arr = preprocess_like_tkinter(img_data)

        if arr is None:
            return "### ⚠️ Preprocessing failed", None

        preds = model.predict(arr)
        digit = int(np.argmax(preds))
        confidence = float(preds[0][digit])

        result_text = f"# 🎯 {digit}\nConfidence: {confidence*100:.2f}%"
        return result_text, None

    except Exception as e:
        return f"### ⚠️ Error: {str(e)}", None
        

# Clear function → resets everything
def clear_all():
    return "", None

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## Gujarati Handwritten Digit Recognition\nDraw a digit (0–9), then click **Recognize**.")

    with gr.Row():
        sketch = gr.Sketchpad(
            canvas_size=(300, 300), 
            label="Draw Digit"
            #brush_radius=10
        )
        #sketch = gr.Sketchpad(canvas_size=(300, 300), label="Draw Digit")

        big_result = gr.Markdown("### 🎯 Predicted Digit: _waiting..._")

    with gr.Row():
        recognize_btn = gr.Button("Recognize")
        clear_btn = gr.Button("Clear")

    recognize_btn.click(fn=recognize, inputs=sketch, outputs=[big_result, sketch])
    clear_btn.click(fn=clear_all, inputs=None, outputs=[big_result, sketch])

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("Starting Gradio on port:", port)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
