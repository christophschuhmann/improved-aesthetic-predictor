


from datasets import load_dataset
import pandas as pd
import statistics
from torch.utils.data import Dataset, DataLoader
import clip
import torch
from PIL import Image, ImageFile
import numpy as np
import time

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)




device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


f = "/mnt/spirit/logos/00000.parquet"
df = pd.read_parquet(f)




x = []
y = []
c= 0

for idx, row in df.iterrows():
    start = time.time()

    mean_val = float(row.preference_average)
    print(mean_val)
    if mean_val <1:
       continue

    img= "/opt/AVA_dataset/AVA_dataset/images/images/" + str(idx)+".jpg"
    print(img)

    try:
       image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    except:
   	   continue

    with torch.no_grad():
       image_features = model.encode_image(image)

    im_emb_arr = image_features.cpu().detach().numpy() 
    x.append(normalized ( im_emb_arr) )
    y_ = np.zeros((1, 1))
    y_[0][0] = mean_val
    #y_[0][1] = stdev 

    y.append(y_)
    #print(im_emb_arr.shape )
    #print( y_ )
    
    #print(idx)
    #print(time.time()-start) 
    print(c)
    c+=1




x = np.vstack(x)
y = np.vstack(y)
print(x.shape)
print(y.shape)
np.save('ava_x_logos_oai.npy', x)
np.save('ava_y_logos_oai.npy', y)
