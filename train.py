import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DEVICE = "cpu"

def get_dataloaders(data_dir,batch_size=16):
    train_dir = os.path.join(data_dir,"train")
    test_dir = os.path.join(data_dir,"test")

    #this function ends up getting applied to all the images
    transform = transforms.Compose([
        transforms.Resize((224,224)),
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

if __name__=="__main__":
    data_dir = "data"
    train_loader,test_loader,class_names = get_dataloaders(data_dir)
    print(train_loader)
    print(class_names)
    iterator = iter(train_loader)
    images, labels = next(iterator)
    print(images)
    print(labels)
