import webdataset as wds
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm

from os.path import join
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip
import open_clip

from PIL import Image, ImageFile


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

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


model = MLP(768)

s = torch.load("/mnt/spirit/ava+logos-l14-reluMSE.pth")
model.load_state_dict(s)


model.to("cuda")
model.eval()





device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   ViT-L/14 


#model3, preprocess2 = clip.load("RN50x64", device=device)  #RN50x64   ViT-L/14


#device = "cuda" if torch.cuda.is_available() else "cpu"
#print(open_clip.list_pretrained())
#model3, _, preprocess2 = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e32') #open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e32')  

#model3.to(device)


#img= "/mnt/spirit/image.jpg"
c=0
urls= []
predictions=[]

for j in range(10):
   if j<10:
     dataset = wds.WebDataset("pipe:aws s3 cp s3://s-datasets/laion400m/laion400m-dat-release/0000"+str(j)+".tar -")  #"pipe:aws s3 cp s3://s-datasets/laion400m/laion400m-dat-release/00625.tar -")
   else:
     dataset = wds.WebDataset("pipe:aws s3 cp s3://s-datasets/laion400m/laion400m-dat-release/000"+str(j)+".tar -")  #"pipe:aws s3 cp s3://s-datasets/laion400m/laion400m-dat-release/00625.tar -")


   for i, d in enumerate(dataset):
      print(c)
      #print(d['json'])  
      metadata= json.loads(d['json'])       
      #print(type(metadata))
      #print(metadata["url"])                 


      pil_image = Image.open(io.BytesIO(d['jpg']))


      c=c+1
      try:
         image = preprocess(pil_image).unsqueeze(0).to(device)
         #image2 = preprocess2(pil_image).unsqueeze(0).to(device)
      except:
         continue

      with torch.no_grad():
         image_features = model2.encode_image(image)
         #image_features2 = model3.encode_image(image2)
         #text_features = model.encode_text(text)
    
       #print (type(image_features.cpu().detach().numpy() ))

      im_emb_arr = normalized(image_features.cpu().detach().numpy() )
      #im_emb_arr2 = normalized(image_features2.cpu().detach().numpy() )
      #print(im_emb_arr.shape)
      #print(torch.from_numpy(im_emb_arr).size())
      #output = model(torch.zeros([3, 1024]))
      #im_emb_arr = np.hstack( (im_emb_arr,im_emb_arr2) )
      prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
      #print(prediction)
      urls.append(metadata["url"])
      predictions.append(prediction)
      #im = pil_image.convert('RGB')
      #im = im.save("/opt/avaoutputlaion-oai-l14/"+str(prediction[0][0].item())+".jpg" ,quality=80)
      #print("/opt/avaoutputlaion-oai-l14/"+str(prediction[0][0].item())+".jpg")


df = pd.DataFrame(list(zip(urls, predictions)),
               columns =['filepath', 'prediction'])


buckets = [(i, i+1) for i in range(20)]


html= "<h1>Aesthetic subsets in LAION 100k samples</h1>"

i =0
for [a,b] in buckets:
    a = a/2
    b = b/2
    total_part = df[(  (df["prediction"] ) *1>= a) & (  (df["prediction"] ) *1 <= b)]
    print(a,b)
    print(len(total_part) )
    count_part = len(total_part) / len(df) * 100
    estimated =int ( len(total_part) )
    part = total_part[:50]
    #print("test1")
    html+=f"<h2>In bucket {a} - {b} there is {count_part:.2f}% samples:{estimated:.2f} </h2> <div>"
    for filepath in part["filepath"]:
        html+='<img src="'+filepath +'" height="200" />'
    #print("test2")

    html+="</div>"
    i+=1
    print(i)
with open("./aesthetic_viz_laion_ava+logos_L14_100k-reluMSE.html", "w") as f:
    f.write(html)
    

