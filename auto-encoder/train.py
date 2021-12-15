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


scan_root = "/data/siyich/am3s"
scan_set = SceneDataset(scan_root, transform = transform, img_name = "back.png")
scan_loader = torch.utils.data.DataLoader(scan_set, batch_size=batch_size)

# am3s_root = "/data/siyich/am3s"
# am3s_set = SceneDataset(am3s_root)

# rand_root = "/data/siyich/am3s_rand"
# rand_set = SceneDataset(rand_root)

# Instantiating the model and hyperparameters
model = DeepAutoencoder()
criterion = torch.nn.MSELoss()
num_epochs = 20
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# List that will store the training loss
train_loss = []
  
# Dictionary that will store the
# different images and outputs for 
# various epochs
outputs = {}

# Training loop starts
for epoch in range(num_epochs):
        
    # Initializing variable for storing 
    # loss
    running_loss = 0
      
    # Iterating over the training dataset
    for batch in scan_loader:
            
        # Loading image(s) and
        # reshaping it into a 1-d vector
        img = batch  
        img = img.reshape(-1, 640*480)
          
        # Generating output
        out = model(img)
          
        # Calculating loss
        loss = criterion(out, img)
          
        # Updating weights according
        # to the calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
          
        # Incrementing loss
        running_loss += loss.item()
    
    # Averaging out loss over entire batch
    running_loss /= batch_size
    train_loss.append(running_loss)
    print(f"epoch: {epoch}, running_loss: {running_loss}")
      
    # Storing useful images and
    # reconstructed outputs for the last batch
    outputs[epoch+1] = {'img': img, 'out': out}

PATH = "model50.pt"
torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, PATH)

# Plotting the training loss
plt.plot(range(1,num_epochs+1),train_loss)
plt.xlabel("Number of epochs")
plt.ylabel("Training Loss")
plt.savefig("loss.png")


