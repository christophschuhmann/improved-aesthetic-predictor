import webdataset as wds
import PIL
import io
import matplotlib.pyplot as plt
import os
import json
from warnings import filterwarnings


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm
from torch.utils.data import Dataset, DataLoader

from os.path import join
from datasets import load_dataset
import pandas as pd
import json

import clip


import logging
import sys
import time
import argparse

#if switching to mutlitprocess swithc both
import multiprocessing as mp
import multiprocessing as mq

import time
import queue
    
# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

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

def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        #args:[%r, %r] , args, kw
        logging.info('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result

    return timed

def load_model():
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

    @timeit
    def load_mdl(): 
        return torch.load("sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
    s=load_mdl()

    @timeit
    def load_state(s): 
        model.load_state_dict(s)
        model.to("cuda")
        model.eval()
    load_state(s)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @timeit
    def load_to_device():
        model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64
        logging.warning(f"Loaded model to clip using device : {device}")
        return model2, preprocess
    model2, preprocess = load_to_device()

    @timeit
    def resize(img_path, max_width):
        @timeit 
        def load_img():
            return PIL.Image.open(img_path)
        
        pil_image = load_img()


        @timeit
        def resize_img(img):
            width, height = img.size
            while width>max_width: 
                #we scale down by integral size so we can use bilinear - otherwise we need to do LANCZOS which is much slower
                width = width//2
                height = height//2
            if width==img.size[0]:
                return img
            else:
                return img.resize((width, height),resample=PIL.Image.NEAREST)
        
        pil_image=resize_img(pil_image)
        return pil_image

    @timeit
    def rank(img_path, pil_image):
        @timeit
        def preprocess_img():
            return preprocess(pil_image).unsqueeze(0).to(device)

        image = preprocess_img()


        @timeit
        def encode_model_img():
            with torch.no_grad():
                return model2.encode_image(image)

        image_features = encode_model_img()
        
        @timeit
        def run_prediction():
            im_emb_arr = normalized(image_features.cpu().detach().numpy() )
            return model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
        prediction = run_prediction()
        logging.debug(f"Aesthetic score predicted by the model for {img_path} : {prediction}")
        return prediction[0,0].item()
    return resize,rank

def do_job(tasks_to_accomplish, tasks_that_are_done, rank_fn, preprocess_fn, args, current_thread, status, kill):
    while True:
        status.value=0
        if kill.value==1: 
            break
        try:
            task = tasks_to_accomplish.get(block=True,timeout=1)
        except queue.Empty:
            status.value=1
            continue
        else:
            status.value=2
            img_path=task
            img = preprocess_fn(img_path,args.maxw)
            status.value=3
            tasks_that_are_done.put((img_path,img))
            status.value=4
            logging.info(f"resized: {img_path} by {current_thread}")
    return True


def run_mp(args, files, preprocess_fn, rank_fn):
    tasks_to_accomplish = mq.Queue()
    tasks_that_are_done = mq.Queue()
    processes = []
    number_of_processes = args.parallel
    file_cnt=0
    for f in files:
        tasks_to_accomplish.put(f)
        file_cnt+=1
    ct = 0
    states=[]
    kills=[]
    for w in range(number_of_processes):
        ct+=1
        states.append(mp.Manager().Value('i',0))
        kills.append(mp.Manager().Value('i',0))
        p = mp.Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done, rank_fn, preprocess_fn, args, ct, states[ct-1], kills[ct-1]))
        processes.append(p)
        p.start()
    done = 0
    while done<file_cnt:
        try:
            task = tasks_that_are_done.get(block=True,timeout=1)
        except queue.Empty:
            logging.debug("EMPTY!!")
            logging.info(f"Status: {[x.value for x in states]}")
            continue
        else:
            img_path,img = task
            rank = rank_fn(img_path,img)
            logging.warning(f"predicted: {img_path} {rank}   image_process: [{''.join([str(x.value) for x in states])}] ")
            yield img_path, rank
            done+=1
    for k in kills:
        k.value=1
    for p in processes:
        p.join()
    

def run_one(args, files, preprocess_fn, rank_fn):
    for f in files:
        img_path = f
        img = preprocess_fn(img_path,args.maxw)
        rank = rank_fn(img_path,img)
        logging.warning(f"predicted: {img_path} {rank}")
        yield img_path, rank


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", default=logging.INFO, type=lambda x: getattr(logging, x), help=f"Configure the logging level: {list(logging._nameToLevel.keys())}")
    parser.add_argument('--maxw', default=1920, type=int, help="Max width in pixels")
    parser.add_argument('--parallel', default=1, type=int, help="how many image decoding to run in parallel")
    parser.add_argument('--imagelist', action='store', help="file with one line per entry of images to process")
    parser.add_argument('rest', nargs=argparse.REMAINDER, help="parallel to process")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format='%(asctime)s %(message)s')

    preprocess_fn,rank_fn = load_model()
    def make_file_list():
        if args.imagelist:
            with open(args.imagelist) as file:
                for line in file:
                    yield(line.rstrip())
        for k in args.rest:
            yield k

    flist = make_file_list()
    if args.parallel==1:
        gen=run_one(args,flist,preprocess_fn,rank_fn)
    else:
        gen=run_mp(args,flist,preprocess_fn,rank_fn)
    for path,rank in gen:
        print (path,"%2.4f"%rank)


    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit


