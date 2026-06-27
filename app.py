import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification

app = Flask(__name__)
# Allows your website (a different domain) to call this server directly
# from the browser. In production you could restrict this to just your
# real domain instead of "*" — see the commented line below.
CORS(app)
# CORS(app, origins=["https://confessionwalls.com"])  # stricter alternative

MODEL_NAME = "prithivMLmods/Mature-Content-Detection"
print("Loading model... this happens once when the server starts.")
model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
print("Model loaded and ready.")

LABELS = {
    "0": "Anime Picture",
    "1": "Hentai",
    "2": "Neutral",
    "3": "Pornography",
    "4": "Enticing or Sensual",
}

# Labels that should cause a rejection. "Anime Picture" and "Enticing or
# Sensual" are deliberately left out — borderline categories, not
# necessarily explicit. Adjust anytime to be stricter or looser.
BLOCKED_LABELS = {"Hentai", "Pornography"}
CONFIDENCE_THRESHOLD = 0.6


@app.route("/", methods=["GET"])
def health_check():
    # Simple endpoint to confirm the server is alive — useful for Render's
    # own health checks, and for you to quickly test in a browser.
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/check-image", methods=["POST"])
def check_image():
    try:
        data = request.get_json(silent=True)
        if not data or "imageBase64" not in data:
            return jsonify({"approved": True}), 200

        image_base64 = data["imageBase64"]
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze()

        results = {LABELS[str(i)]: round(probs[i].item(), 4) for i in range(len(LABELS))}
        print(f"Classification result: {results}")

        flagged_label = None
        for label in BLOCKED_LABELS:
            if results.get(label, 0) >= CONFIDENCE_THRESHOLD:
                flagged_label = label
                break

        if flagged_label:
            print(f"Image rejected — flagged as {flagged_label}: {results}")
            return jsonify({
                "approved": False,
                "reason": "This image can't be posted — it may contain explicit or inappropriate content.",
            }), 200

        return jsonify({"approved": True}), 200

    except Exception as e:
        # Fail open — a moderation outage should never fully block posting.
        # The report button + manual admin review remain the backstop.
        print(f"Image check failed, approving by default: {e}")
        return jsonify({"approved": True}), 200


if __name__ == "__main__":
    # Render sets the PORT environment variable; default to 5000 for
    # local testing on your own computer.
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
