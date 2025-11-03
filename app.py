from flask import Flask, request, jsonify
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import io
import os

app = Flask(__name__)

# ✅ Model ID
MODEL_ID = "Jayanth2002/phi-3-5-vision-decovqa"

# ✅ Add trust_remote_code=True (fixes Render crash)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

@app.route("/vision-ai", methods=["POST"])
def vision_ai():
    """
    POST /vision-ai
    Form-data:
        - question: text
        - image: file
    """
    try:
        question = request.form.get("question")
        if not question:
            return jsonify({"error": "Missing 'question' parameter"}), 400

        file = request.files.get("image")
        if not file:
            return jsonify({"error": "Missing 'image' file"}), 400

        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Preprocess + inference
        inputs = processor(text=question, images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100)
        answer = processor.batch_decode(output, skip_special_tokens=True)[0]

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "✅ Aegis Vision API is running successfully!"


if __name__ == "__main__":
    # ✅ Important for Render port binding
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
