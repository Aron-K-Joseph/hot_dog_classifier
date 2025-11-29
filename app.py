import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
from train import Net

net = Net()
net.load_state_dict(torch.load('hotdog_model.pth')["model_state_dict"])
net.eval()
print(net)

class_names = ["hotdog", "nothotdog"]


new_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5,0.5,0.5)
        )
])


def load_image(image_path):
        image = Image.open(image_path).convert("RGB")
        image = new_transform(image)
        image = image.unsqueeze(0)
        return image

#NEEDS EDITING
#We need to find an API or something to connect the file uploaded from
#app.js to here so that it can be evaluated. Then, we return it to app.js
image_paths = ["dog.jpg", "hotdog.jpg"]
images = [load_image(img) for img in image_paths]

with torch.no_grad():
        for image in images:
                output = net(image)
                _, predicted = torch.max(output, 1)
                print(f'Prediction: {class_names[predicted.item()]}')