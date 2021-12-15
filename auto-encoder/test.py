import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from model import DeepAutoencoder
from data import SceneDataset


batch_size = 16

# Initializing the transform for the dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5), (0.5))
])

model = DeepAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

PATH = "model.pt"

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()

scan_root = "/data/siyich/am3s"
scan_set = SceneDataset(scan_root, transform = transform, img_name = "back.png")
scan_loader = torch.utils.data.DataLoader(scan_set, batch_size=batch_size)

am3s_root = "/data/siyich/am3s"
am3s_set = SceneDataset(am3s_root, transform = transform, img_name = "final.png")
am3s_loader = torch.utils.data.DataLoader(am3s_set, batch_size=batch_size)

rand_root = "/data/siyich/am3s_rand"
rand_set = SceneDataset(rand_root, transform = transform, img_name = "final.png")
rand_loader = torch.utils.data.DataLoader(rand_set, batch_size=batch_size)

test_loader = scan_loader

running_loss = 0

for batch in test_loader:
            
    # Loading image(s) and
    # reshaping it into a 1-d vector
    img = batch  
    img = img.reshape(-1, 640*480)
        
    # Generating output
    out = model(img)
        
    # Calculating loss
    loss = criterion(out, img)

    running_loss += loss.item()

running_loss /= batch_size
print(f"test_loss: {running_loss}")

