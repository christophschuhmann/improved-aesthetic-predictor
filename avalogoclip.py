import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tqdm
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer




x = np.load ("/mnt/spirit/ava_x.npy") #_openclip_l14.npy")

y = np.load ("/mnt/spirit/ava_y.npy") #_openclip_l14.npy")

x2 = np.load ("/mnt/spirit/x_logos_oai.npy") #_openclip_l14.npy")

y2 = np.load ("/mnt/spirit/y_logos_oai.npy") #_openclip_l14.npy")

x =np.concatenate((x,x2),axis = 0)
y =np.concatenate((y,y2),axis = 0)

from sklearn.utils import shuffle
x, y = shuffle(x, y)



train_tensor_x = torch.Tensor(x[:262000]) # transform to torch tensor
train_tensor_y = torch.Tensor(y[:262000])

train_dataset = TensorDataset(train_tensor_x,train_tensor_y) # create your datset
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,  num_workers=16) # create your dataloader


val_tensor_x = torch.Tensor(x[262000:]) # transform to torch tensor
val_tensor_y = torch.Tensor(y[262000:])

print(train_tensor_x.size())

print(val_tensor_x.size())

print( val_tensor_x.dtype)
print( val_tensor_x[0].dtype)


val_dataset = TensorDataset(val_tensor_x,val_tensor_y) # create your datset
val_loader = DataLoader(val_dataset, batch_size=512,  num_workers=16) # create your dataloader





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = MLP(768).to(device)

optimizer = torch.optim.Adam(model.parameters()) #, lr=1e-3
criterion = nn.MSELoss()
criterion2 = nn.L1Loss()

epochs = 30

model.train()
best_loss =999

for epoch in range(epochs):
    losses = []
    losses2 = []
    for batch_num, input_data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        losses.append(loss.item())


        optimizer.step()

        if batch_num % 1000 == 0:
            print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
            #print(y)

    print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))
    losses = []
    losses2 = []
    
    for batch_num, input_data in enumerate(val_loader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)
        lossMAE = criterion2(output, y)
        #loss.backward()
        losses.append(loss.item())
        losses2.append(lossMAE.item())
        #optimizer.step()

        if batch_num % 1000 == 0:
            print('\tValidation - Epoch %d | Batch %d | MSE Loss %6.2f' % (epoch, batch_num, loss.item()))
            print('\tValidation - Epoch %d | Batch %d | MAE Loss %6.2f' % (epoch, batch_num, lossMAE.item()))
            
            #print(y)

    print('Validation - Epoch %d | MSE Loss %6.2f' % (epoch, sum(losses)/len(losses)))
    print('Validation - Epoch %d | MAE Loss %6.2f' % (epoch, sum(losses2)/len(losses2)))
    if sum(losses)/len(losses) < best_loss:
        print("Best MAE Val loss so far. Saving model ava+logos-l14-reluMSE.pth")
        best_loss = sum(losses)/len(losses)
        print( best_loss ) 
        torch.save(model, "ava+logos-l14-reluMSE.pt")
        torch.save(model.state_dict(), "ava+logos-l14-reluMSE.pth")




#torch.save(model, "ava+logos-l14-reluMSE.pt")
#torch.save(model.state_dict(), "ava+logos-l14-reluMSE.pth")

print( best_loss ) 

model.eval()
output = model(x[:5].to(device))
print(output.size())
print(output)