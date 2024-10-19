import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Custom Dataset for loading manga panels
class MangaColorizationDataset(Dataset):
    def __init__(self, grayscale_dir, color_dir, transform=None):
        self.grayscale_dir = grayscale_dir
        self.color_dir = color_dir
        self.transform = transform
        self.image_names = os.listdir(grayscale_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        grayscale_image_path = os.path.join(self.grayscale_dir, self.image_names[idx])
        color_image_path = os.path.join(self.color_dir, self.image_names[idx])

        grayscale_image = Image.open(grayscale_image_path).convert('L')
        color_image = Image.open(color_image_path).convert('RGB')

        if self.transform:
            grayscale_image = self.transform(grayscale_image)
            color_image = self.transform(color_image)

        return grayscale_image, color_image

# Define a simple CNN model for colorization
class ColorizationCNN(nn.Module):
    def __init__(self):
        super(ColorizationCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # To ensure output is between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters and other settings
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# Transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load the dataset
grayscale_dir = 'path_to_grayscale_images'
color_dir = 'path_to_color_images'
dataset = MangaColorizationDataset(grayscale_dir, color_dir, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = ColorizationCNN().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (grayscale_images, color_images) in enumerate(dataloader):
        grayscale_images = grayscale_images.to(device)
        color_images = color_images.to(device)

        # Forward pass
        outputs = model(grayscale_images)
        loss = criterion(outputs, color_images)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")