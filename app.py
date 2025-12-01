import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
from train import Net
from flask import Flask, request, jsonify
from flask_cors import CORS

#Loading the model from train.py
net = Net()
net.load_state_dict(torch.load('hotdog_model.pth')["model_state_dict"])
net.eval()

new_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5,0.5,0.5)
        )
])

class_names = ["hotdog", "nothotdog"]

app = Flask(__name__)
CORS(app)

def load_img(file):
        image = Image.open(io.BytesIO(file)).convert("RGB")
        image = new_transform(image)
        return image.unsqueeze(0)

@app.route("/predict", methods=["POST"])

def predict():
        if "file" not in request.files:
                return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        img = file.read()
        img_tensor = load_img(img)

        with torch.no_grad():       
                output = net(img_tensor)
                _, predicted = torch.max(output, 1)
                name = class_names[predicted.item()]

        return jsonify({"prediction": name})

if __name__ == "__main__":
        app.run(debug=True)

#NEEDS EDITING
#We need to find an API or something to connect the file uploaded from
#app.js to here so that it can be evaluated. Then, we return it to app.js