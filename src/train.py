import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MangaColorizationDataset, transform
from model import ColorizationNet  # Ensure this matches your model choice

def train_model():
    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training data
    train_dataset = MangaColorizationDataset('dataset/train/grayscale', 'dataset/train/color', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Load testing data
    test_dataset = MangaColorizationDataset('dataset/test/grayscale', 'dataset/test/color', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = ColorizationNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):  # Number of epochs
        model.train()
        for gray_images, color_images in train_loader:
            gray_images, color_images = gray_images.to(device), color_images.to(device)
            outputs = model(gray_images)
            loss = criterion(outputs, color_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Save the final trained model
    torch.save(model.state_dict(), 'models/colorization_model.pth')

    # Evaluation loop
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for gray_images, color_images in test_loader:
            gray_images, color_images = gray_images.to(device), color_images.to(device)
            outputs = model(gray_images)
            loss = criterion(outputs, color_images)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {avg_loss}')
    
if __name__ == "__main__":
    train_model()