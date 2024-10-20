import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MangaColorizationDataset(Dataset):
    def __init__(self, grayscale_dir, color_dir, transform=None):
        self.grayscale_dir = grayscale_dir
        self.color_dir = color_dir
        self.transform = transform
        self.grayscale_images = os.listdir(grayscale_dir)

    def __len__(self):
        return len(self.grayscale_images)

    def __getitem__(self, idx):
        grayscale_path = os.path.join(self.grayscale_dir, self.grayscale_images[idx])
        color_path = os.path.join(self.color_dir, self.grayscale_images[idx])

        grayscale_image = Image.open(grayscale_path).convert('L')
        color_image = Image.open(color_path).convert('RGB')

        if self.transform:
            grayscale_image = self.transform(grayscale_image)
            color_image = self.transform(color_image)

        return grayscale_image, color_image

transform = transforms.Compose([
    transforms.Resize((1200, 760)),  
    transforms.ToTensor(),
])