import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
       
        #3*32*32(one image)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        #6*28*28 -> 6*14*14(after pooling)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        #16*10*10 -> 16*5*5(after pooling)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        #16*5*5 = 400
        self.fc1 = nn.Linear(400,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_dataloaders(data_dir,batch_size=32):
    train_dir = os.path.join(data_dir,"train")
    test_dir = os.path.join(data_dir,"test")

    #this function ends up getting applied to all the images
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5,0.5,0.5)
        )
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = ["hotdog", "nothotdog"]
    return train_loader, test_loader, class_names


def train_model(model, train_loader, val_loader, num_epochs=16, lr=0.01, momentum=0.9):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=momentum)

    for epoch in range(num_epochs):
        #TRAINING
        model.train()
        running_loss = 0
        correct = 0
        total = 0

    # get some random training images

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            #compute loss
            loss = criterion(outputs, labels)
            #compute gradient
            loss.backward()
            #SGD step creates new weights
            optimizer.step()
            
            #loss.item() is average loss in batch and images.size(0) is num images in batch
            running_loss+=loss.item()*images.size(0)
            #torch.max returns max score and index
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct+=(predicted==labels).sum().item()
        train_loss = running_loss/total
        train_acc = correct/total
        

        #VALIDATION
        model.eval()
        val_loss=0
        val_correct=0
        val_total=0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs,labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = val_loss/val_total
        val_acc = val_correct/val_total

        print(f"Epoch {epoch+1}/{num_epochs}")
        print("Correct: ", correct, "Total: ", total)
        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
    return model

if __name__=="__main__":
    print("Script Starting") 
    train_loader,test_loader,class_names = get_dataloaders("data")
    print("Data Loaded")
    model = Net()
    print("Model Created")
    model = train_model(model,train_loader,test_loader)
    print("Model Trained")
    torch.save({
        "model_state_dict": model.state_dict()
    },"hotdog_model.pth")
    print("Model Saved")
