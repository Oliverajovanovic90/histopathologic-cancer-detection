import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from flask import Flask, request, jsonify
from torchvision import transforms

# =========================
# App setup
# =========================
app = Flask(__name__)
DEVICE = torch.device("cpu")

# =========================
# Model definition (MUST MATCH TRAINING)
# =========================
class SimpleCNN(nn.Module):
    """
    Baseline CNN architecture (final selected model).
    """

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
model = SimpleCNN()
model.load_state_dict(torch.load("model_cnn_v1.pth", map_location=DEVICE))
model.eval()

# =========================
# Image preprocessing
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

# =========================
# Prediction endpoint
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    label = "cancer" if predicted.item() == 1 else "non-cancer"

    return jsonify({
        "prediction": label,
        "confidence": round(confidence.item(), 4)
    })

# =========================
# Cloud-safe run
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
