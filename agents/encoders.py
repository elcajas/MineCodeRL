import torch
import torch.nn as nn
import mineclip.utils as U

MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 20 * 32, 512)  # Adjusted dimensions based on input size

    def forward(self, x):

        x = U.basic_image_tensor_preprocess(
            x, mean=MC_IMAGE_MEAN, std=MC_IMAGE_STD
        )
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 256 * 20 * 32)  # Flatten the feature map
        x = torch.relu(self.fc1(x))
        return x
    
if __name__ == "__main__":
    encoder = ImageEncoder()

    dummy_input = torch.randn(8, 3, 160, 256)
    output = encoder(dummy_input)

    # Check the output shape
    print("Output shape:", output.shape) 
