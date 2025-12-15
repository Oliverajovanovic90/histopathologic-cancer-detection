"""
predict.py

Purpose:
Load the trained CNN model and serve predictions
via a Flask web API.
"""

# =========================
# Imports
# =========================
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify


# =========================
# Configuration
# =========================
MODEL_PATH = "model_cnn_v1.pth"
IMAGE_SIZE = (96, 96)

# initialize Flask app
app = Flask(__name__)


# =========================
# CNN Model Definition
# (must match train.py exactly)
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# =========================
# Load model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# =========================
# Image preprocessing
# =========================
def preprocess_image(image_bytes):
    """
    Convert raw image bytes into a tensor suitable for the CNN.
    """
    # convert bytes to numpy array
    np_img = np.frombuffer(image_bytes, np.uint8)

    # decode image
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize image (safety check)
    img = cv2.resize(img, IMAGE_SIZE)

    # normalize pixels
    img = img / 255.0

    # convert to tensor (C, H, W)
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

    # add batch dimension
    img = img.unsqueeze(0)

    return img.to(device)


# =========================
# Prediction endpoint
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts an image file and returns cancer prediction.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    # preprocess image
    img_tensor = preprocess_image(image_bytes)

    # model inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    # map class to label
    label_map = {0: "non-cancer", 1: "cancer"}

    return jsonify({
        "prediction": label_map[pred_class],
        "confidence": round(confidence, 4)
    })


# =========================
# Run app
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
