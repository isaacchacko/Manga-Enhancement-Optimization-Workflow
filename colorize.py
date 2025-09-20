import os
import torch
from PIL import Image
from torchvision import transforms
from model import FlexibleColorizationNet  # Ensure correct import path


# Recreate the model architecture
model = FlexibleColorizationNet()

# Load the saved state_dict
model.load_state_dict(torch.load('models/colorization_model.pth', map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((1200, 760)),  # Adjust based on your needs
    transforms.ToTensor(),
])

def colorize_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            grayscale_image = Image.open(os.path.join(input_dir, filename)).convert('L')
            input_tensor = transform(grayscale_image).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                output_tensor = model(input_tensor)

            output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
            output_image.save(os.path.join(output_dir, filename))

colorize_images('dataset/test/grayscale', 'output/colorized')
