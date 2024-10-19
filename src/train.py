import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MangaColorizationDataset, transform
from model import ColorizationNet

def train_model():
    train_dataset = MangaColorizationDataset('data/train/grayscale', 'data/train/color', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = ColorizationNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  # number of epochs
        for gray_images, color_images in train_loader:
            outputs = model(gray_images)
            loss = criterion(outputs, color_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train_model()