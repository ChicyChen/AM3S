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

test_loader = rand_loader

outputs = {}

img = list(test_loader)[-1]
img = img.reshape(-1, 480 * 640)
out = model(img)

outputs['img'] = img
outputs['out'] = out

# Plotting reconstructed images
# Initializing subplot counter
counter = 1
val = outputs['out'].detach().numpy()
  
# Plotting first 10 images of the batch
for idx in range(3, 5):
    plt.subplot(2, 2, counter)
    # plt.title("Reconstructed \n image")
    plt.imshow(val[idx].reshape(480, 640))
    plt.axis('off')
  
    # Incrementing subplot counter
    counter += 1
  
# Plotting original images
  
# Plotting first 10 images
for idx in range(3, 5):
    val = outputs['img']
    plt.subplot(2, 2, counter)
    plt.imshow(val[idx].reshape(480, 640))
    # plt.title("Original Image")
    plt.axis('off')
  
    # Incrementing subplot counter
    counter += 1
  
plt.tight_layout()
plt.savefig("rand_vis.png")
