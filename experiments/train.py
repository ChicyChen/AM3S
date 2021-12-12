from model import Model, Loss
from SensorData import SensorData

import os, sys
import subprocess
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import cv2



class MyDataset(Dataset):
    # An object for representing the Dataset for Pytorch.
    def __init__(self, image_fns, dim=256):
        super().__init__()
        self.image_fns = image_fns
        self.dim = (dim, dim)

    def __len__(self):
        return len(self.image_fns)

    def getImage(self, index):
        images = [] 
        for image_fn in self.image_fns[index]:
            image = cv2.imread(image_fn, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, self.dim)
            images.append(image)

        images = np.array(images)
        images = torch.from_numpy(images)
        image = torch.tensor(images.float(), requires_grad=True)
        return image

    def __getitem__(self, index):
        image = self.getImage(index)
        return image

def convertSensToJPG(foldername):
    # Converts Sens to Jpeg files for ScanNet
    for folder in os.listdir(foldername):
        if folder.endswith(".csv"):
            continue
        for file in os.listdir(foldername + '/' + folder):
            if file.endswith(".sens"):
                sens_fn = foldername + '/' + folder + '/' + file
                output_path = foldername + '/' + folder + '/'
                if os.path.exists(output_path):
                    continue
                else:
                    os.makedirs(output_path)
                sys.stdout.write('loading %s...' % sens_fn)
                sd = SensorData(sens_fn)
                sd.export_color_images(os.path.join(output_path, 'color'))


def createDataset(images, batch_size, dim):
    # Making dataset and dataloaders for PyTorch from a set of image filenames and labels.
    dataset = MyDataset(images,dim)
    loader = DataLoader(dataset,batch_size=batch_size, shuffle=False)
    return dataset, loader


def readDataset(foldername, batch_size, split_ratio, dim=256, shuffle=False, max_depth=100):
    # Reading and splitting the images from trainval and returning the dataset objects and the dataloaders for training.
    images = []

    for folder in os.listdir(foldername):
        for file in os.listdir(foldername + '/' + folder):
            path = foldername + '/' + folder + "/" + file 
            if os.path.isdir(path):
                tmp_fns = []
                for img_fn in os.listdir(path):
                    tmp_path = path + '/' + img_fn
                    tmp_fns.append(tmp_path)
                if len(tmp_fns) > max_depth:
                    images.append(tmp_fns[0:max_depth])

    idx_shuffled = list(range(0, len(images)))

    if shuffle:
        random.shuffle(idx_shuffled)

    split_idx = int(split_ratio * len(images))
    train_images = [images[i] for i in idx_shuffled[:split_idx]]
    val_images = [images[i] for i in idx_shuffled[split_idx:]]


    print(str(len(images)) + " rgb-d scans read.")
    idx_shuffled = list(range(0, len(images)))

    if shuffle:
        random.shuffle(idx_shuffled)

    split_idx = int(split_ratio * len(images))
    train_images = [images[i] for i in idx_shuffled[:split_idx]]
    val_images = [images[i] for i in idx_shuffled[split_idx:]]

    train_dataset, train_loader = createDataset(train_images, batch_size, dim)
    val_dataset, val_loader = createDataset(val_images, batch_size, dim)
    print("Dataloaders created, train has " + str(len(train_dataset)) + " samples and val has " + str(len(val_dataset)) + " samples.")
    return train_loader, val_loader

def test_model(model, val_loader):
    model.eval()
    criterion = Loss()
    print("    ")
    print("**************************STARTED VALIDATION CHECK**************************")
    running_loss = 0
    for i, data in enumerate(val_loader):
        # get the inputs; data is a list of [batch, labels]
        batch = data
        batch = batch.float()
        batch_orig = batch.copy()
        # zero the parameter gradients
        outputs = model(batch)
        loss = criterion(outputs, batch_orig)
        running_loss += loss
    print("Val loss is " + str(loss) + "%")
    print("**************************ENDED VALIDATION CHECK**************************")
    return running_loss, criterion.ssim()

def train(foldername, batchSize, num_epochs, loadFn=None):
    model = Model(100)
    criterion = Loss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    best_loss = 100000000
    best_val_loss = 0
    best_model = Model(100)
    avg_ssim = 0
    start_epoch = 0

    train_loader, val_loader = readDataset(foldername, batchSize, 0.7)

    if loadFn:
        checkpoint = torch.load(loadFn)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        best_val_loss = checkpoint['vloss']
        avg_ssim = checkpoint['avg_ssim']
        print("Loaded Model")

    num_iters = len(train_loader)

    for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times  
        print("Training epoch: " + str(epoch) + " of " + str(num_epochs))
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [batch, labels]
            print("Started Iteration: " + str(i) + " of " + str(num_iters))
            batch = data
            batch = batch.float()
            print(batch.shape)
            batch_orig = batch.clone().detach().requires_grad_(True)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch_orig)
            loss.backward()
            running_loss += loss       
            print("Backward done on this iteration")
            optimizer.step()
            print("Model finished running iteration", i, "\n")
            print("Loss:", loss)

        # Validation Loss
        val_loss= test_model(model, val_loader)
        ssim = criterion.ssim()
        print("avg ssim", sum(ssim)/len(ssim))

        if running_loss < best_loss:
            best_loss = running_loss
            best_model = model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        model_fn = "./models/model_epoch" + str(epoch) + "_loss" + str(running_loss) + "_vloss" + str(val_loss) + ".pt"
        torch.save({
        'epoch': (epoch+1),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss,
        'vacc': val_loss,
        "avg_ssim" : sum(ssim)/len(ssim)
        }, model_fn)

        print("Finished Running Epoch ", epoch, " loss: ", loss, "\n")

    print('Finished Training')



if __name__ == '__main__':
    train(sys.argv[1], 4, 100, loadFn=None)