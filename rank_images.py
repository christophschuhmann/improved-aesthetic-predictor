import webdataset as wds
import PIL
import io
import matplotlib.pyplot as plt
import os
import json
import shutil
import pyfastcopy
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

#if switching to mutlitprocess switch both
import threading as mp
import queue as mq
import threading

import time
import queue

import av
import traceback
from sortedcontainers import SortedDict

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

def load_model(device):
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

    @timeit
    def load_mdl(): 
        return torch.load("sac+logos+ava1-l14-linearMSE.pth",map_location=torch.device(device))   # load the model you trained previously or the model available in this repo
    s=load_mdl()


    @timeit
    def load_state(s): 
        model.load_state_dict(s)
        model.to(device)
        model.eval()
    load_state(s)


    
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
            return model(torch.from_numpy(im_emb_arr).to(device).type(torch.float32))
        prediction = run_prediction()
        logging.debug(f"Aesthetic score predicted by the model for {img_path} : {prediction}")
        return prediction[0,0].item()
    return resize,rank

def prepdir(f):
    dname=os.path.dirname(f)
    os.makedirs(dname, exist_ok=True)

def do_images_job(tasks_to_accomplish, tasks_that_are_done, rank_fn, preprocess_fn, args, current_thread, status, kill):
    while True:
        status[current_thread]='i'
        if kill[current_thread]==1: 
            break
        try:
            task = tasks_to_accomplish.get(block=True,timeout=1)
        except queue.Empty:
            status[current_thread]=' '
            continue
        else:
            status[current_thread]='p'
            img_path=task
            try:
                img = preprocess_fn(img_path,args.maxw)
                status[current_thread]='q'
                while tasks_that_are_done.qsize()>1000:
                    logging.info(f"Thread {current_thread} is waiting for tasks_that_are_done to be processed : qsize: {tasks_that_are_done.qsize()}")
                    status[current_thread]='w'
                    time.sleep(1)
                tasks_that_are_done.put(("PREPROCESSED",[img_path,img]))
                logging.info(f"resized: {img_path} by {current_thread}")
            except Exception:
                logging.error(f"Processing {img_path} on {current_thread} failed due to exception: {traceback.format_exc()}")
                tasks_that_are_done.put(("SKIPPED",(img_path,0.)))
    return True

def run_images_mp(args, files, preprocess_fn, rank_fn):
    tasks_to_accomplish = mq.Queue()
    tasks_that_are_done = mq.Queue()
    processes = []
    number_of_processes = args.parallel
    file_cnt=0
    states=[0 for x in range(number_of_processes)]
    kills=[0 for x in range(number_of_processes)]
    for ct in range(number_of_processes):
        p = mp.Thread(target=do_images_job, args=(tasks_to_accomplish, tasks_that_are_done, rank_fn, preprocess_fn, args, ct, states, kills))
        processes.append(p)
        p.start()
    for f in files:
        tasks_to_accomplish.put(f)
        file_cnt+=1
    done = 0
    while done<file_cnt:
        try:
            state,msg = tasks_that_are_done.get(block=True,timeout=1)
        except queue.Empty:
            logging.debug("EMPTY!!")
            logging.info(f"Status: {[x for x in states]}")
            continue
        else:
            if state=="PREPROCESSED":
                img_path,img = msg
                rank = rank_fn(img_path,img)
                try:
                    qsize = f" qsize: {tasks_that_are_done.qsize()}"
                except:
                    qsize = f""
                logging.warning(f"predicted: {img_path} {rank}   image_process: [{''.join([str(x) for x in states])}] files: {file_cnt} done: {done}{qsize}")
                yield img_path, rank
            done+=1
    
    for i in range(len(kills)):
        kills[i]=1
    logging.warning(f"Waiting on threads to finish")
    for p in processes:
        p.join()
    logging.warning(f"Finished main thread")
    

def do_video_job(tasks_to_accomplish, tasks_that_are_done, rank_fn, preprocess_fn, args, current_thread, status, kill):
    def prep_fname(f,expr,framenum):
        extless=os.path.splitext(f)[0]
        expr=expr.replace(r'%F',os.path.basename(extless))
        expr=expr.replace(r'%D',os.path.dirname(extless))
        expr=expr.replace(r'%f',extless)
        expr=expr.replace(r'%d','%06d'%framenum)
        return expr
    when = args.v_when
    if when.startswith("mod:"):
        every_frame = int(when.split(":")[1])
        logging.warning(f"configure to mod: {every_frame}")
        def should_process(frame, last_rank):
            return frame % every_frame == 0
    elif when.startswith("adaptive_mod:"):
        splt = when.split(":")
        every_frame = int(splt[1])
        adaptive_score = float(splt[2])
        logging.warning(f"configure to adaptive mod: {every_frame} cutoff : {adaptive_score}")
        def should_process(frame, last_rank):
            return frame % every_frame == 0 or last_rank>=adaptive_score
    class Bucket:
        def __init__(self,fn):
            self.container=SortedDict()
            self.window=args.v_window
            self.maxgap=args.v_maxgap
            self.windowframes=args.v_windowframes
            self.fn=fn
        def insert(self,newidx,rank,params):
            
            if self.window<=1:
                self.fn(True,params)
                return
            
            if len(self.container)==0:
                self.container[newidx]=(rank,params)
            else:
                lastidx,_ = self.container.peekitem(index=-1)
                if newidx-lastidx>self.maxgap:
                    self.flush()
                self.container[newidx]=(rank,params)
                if len(self.container)>=self.window or ((self.container.peekitem(index=-1)[0])-(self.container.peekitem(index=0)[0])>self.windowframes) :
                    self.flush()
        def flush(self):
            if len(self.container)==0: return

            maxrank = max([v[0] for (k,v) in self.container.items()])
            saved = 0
            for frame,(rank,args) in self.container.items():
                if maxrank==rank and saved==0:
                    logging.warning(f"window {current_thread} : picked frame {frame} to save from {len(self.container)} frames in buffer. first:{self.container.peekitem(index=0)[0]} last:{self.container.peekitem(index=-1)[0]}")
                    self.fn(True,*args)
                    saved+=1
                else:
                    self.fn(False,*args)
            self.container.clear()
    
    
    while True:
        status[current_thread]=0
        if kill[current_thread]==1: return True
        try:
            f = tasks_to_accomplish.get(block=True,timeout=1)
        except queue.Empty:
            status[current_thread]=-1
            continue
        else:
            status[current_thread]=-2
            if not f: 
                tasks_that_are_done.put(("DONE","Skipped"))
                continue
            left_retries = args.retry+1
            while (not kill[current_thread]==1) and (left_retries>0):
                try:
                    left_retries-=1
                    container = av.open(f)
                    logging.warning(f"opened video: thread: {current_thread} : {f}")
                    video = next(s for s in container.streams)
                    rank = 0.
                    next_file = False
                    def save_fn(should_save,rank,outfname,img_path,img):
                        if should_save:
                            prepdir(outfname)
                            logging.warning(f"saving {current_thread} : writing to: {outfname} from {img_path}")
                            img.save(outfname,quality=90)
                            tasks_that_are_done.put(("SAVED",(outfname,rank)))
                        else:
                            tasks_that_are_done.put(("SKIPPED",(img_path,rank)))

                    buckets = Bucket(save_fn)
                    for packet in container.demux(video):
                        if next_file: break
                        for frame in packet.decode():
                            status[current_thread]=frame.index
                            if kill[current_thread]==1: return True
                            if args.v_output:
                                outfname = prep_fname(f,args.v_output,frame.index)
                            if args.v_maxframes>0 and frame.index>args.v_maxframes:
                                next_file=True
                                break
                            if should_process(frame.index,rank):
                                img_path="%s:%d"%(f,frame.index)
                                if args.v_output and os.path.exists(outfname) and not args.force:
                                    rank = args.cutoff+0.1
                                    logging.warning(f"skipped: {current_thread} : {img_path} as {outfname} already exists!")
                                    continue

                                img=frame.to_image()
                                rank = rank_fn(img_path,img)
                                logging.warning(f"predicted: {current_thread} : {img_path} {rank}")
                                if rank>args.cutoff:
                                    if args.v_output:
                                        buckets.insert(frame.index,rank,(rank,outfname,img_path,img))
                                    else:
                                        tasks_that_are_done.put(("SKIPPED",(img_path,rank)))
                                    ##
                    buckets.flush()
                except Exception:
                    logging.error(f"Processing {f} on {current_thread} failed due to exception: {traceback.format_exc()}")
                    if left_retries>0:
                        time.sleep(args.retry_sleep)
                    else:
                        tasks_that_are_done.put(("DONE",f"Exception for {f}"))
                else:
                    tasks_that_are_done.put(("DONE",f"Succeeded for {f}"))
                    left_retries=0
    return True

def run_video_mp(args, files, preprocess_fn, rank_fn):
    tasks_to_accomplish = queue.Queue()
    tasks_that_are_done = queue.Queue()
    threads = []
    number_of_processes = args.parallel
    file_cnt=0
    for f in files:
        tasks_to_accomplish.put(f)
        file_cnt+=1
    states=[0 for x in range(number_of_processes)]
    kills=[0 for x in range(number_of_processes)]
    for ct in range(number_of_processes):
        t = threading.Thread(target=do_video_job, args=(tasks_to_accomplish, tasks_that_are_done, rank_fn, preprocess_fn, args, ct, states, kills))
        threads.append(t)
        t.start()
    done = 0
    while done<file_cnt:
        def process_string():
            return ",".join([str(x) for x in states])
        try:
            state,msg = tasks_that_are_done.get(block=True,timeout=1)
        except queue.Empty:
            logging.debug("EMPTY!!")
            logging.warning(f"video_process: [{process_string()}] files: {file_cnt} done: {done}")
            continue
        else:
            if state=="DONE":
                done+=1
                logging.warning(f"video_process: [{process_string()}] files: {file_cnt} done: {done} reason: {state}")
            else:
                logging.warning(f"video_process: [{process_string()}] files: {file_cnt} done: {done}")
                yield msg
    for i in range(len(kills)):
        kills[i]=1
    logging.warning(f"Waiting on threads to finish")
    for p in threads:
        p.join()
    logging.warning(f"Finished main thread")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", default=logging.INFO, type=lambda x: getattr(logging, x), help=f"Configure the logging level: {list(logging._nameToLevel.keys())}")
    parser.add_argument('--maxw', default=1920, type=int, help="Max width in pixels")
    parser.add_argument('--parallel', default=1, type=int, help="how many image decoding to run in parallel")
    parser.add_argument('--list', action='store', help="file with one line per entry of images to process")
    parser.add_argument('--cutoff', type=float, default=5.3, help="action of rank is above this")
    parser.add_argument('--force', action="store_true", default=False, help="whether to force overwrite of existing files")
    parser.add_argument('--i_output', default="", help="image: where to store the output. %%f - input filename. %%F - path-excluded filename")
    parser.add_argument('--video', action='store_true', default=False, help="whether to use video")
    parser.add_argument('--v_when', default="mod:1", help="video : rank not every frame. mod:1 - every frame, mod:5 - every 5th frame, adaptive_mod:30:5.0 - every 30th frame or if prev fame rank is >= 5.0. ")
    parser.add_argument('--retry', type=int, default=0, help="How many times to try ")
    parser.add_argument('--retry_sleep', type=int, default=5, help="How many seconds to sleep betwee retries")
    parser.add_argument('--v_maxframes', type=int, default=0, help="video : max number of frames to process - 0 is all")
    parser.add_argument('--v_window', type=int, default=100, help="video : how many images to rank before outputting best")
    parser.add_argument('--v_maxgap', type=int, default=100, help="video : how big index-gap should be when ranking image window")
    parser.add_argument('--v_windowframes', type=int, default=500, help="video : how big first-last index in window can be")
    parser.add_argument('--v_output', default="", help="video : where to store the output. %%f - extensionless input filename. %%F - path-excluded extensioneless filename. %%d - with 6-digit frame seq")
    parser.add_argument('--device', default=("cuda" if torch.cuda.is_available() else "cpu"), help="what to use to run ML. [cuda, cpu]")
    parser.add_argument('rest', nargs=argparse.REMAINDER, help="parallel to process")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format='%(asctime)s:%(lineno)d %(message)s')

    preprocess_fn,rank_fn = load_model(args.device)
    def prep_fname(f,expr):
        extless=f
        expr=expr.replace(r'%F',os.path.basename(extless))
        expr=expr.replace(r'%D',os.path.dirname(extless))
        expr=expr.replace(r'%f',extless)
        return expr
    def make_file_list():
        if args.list:
            with open(args.list) as file:
                for line in file:
                    yield(line.rstrip())
        for k in args.rest:
            yield k

    flist = make_file_list()
    if args.video:
        gen=run_video_mp(args,flist,preprocess_fn,rank_fn)
    else:
        gen=run_images_mp(args,flist,preprocess_fn,rank_fn)
    
    for path,rank in gen:
        print (path,"%2.4f"%rank)
        if args.i_output:
            outfname = prep_fname(path,args.i_output)
            if not os.path.exists(outfname) or args.force:
                if rank>args.cutoff:
                    prepdir(outfname)
                    if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
                        os.system(f'cp "{path}" "{outfname}"')
                    else:
                            try:
                                os.link(path,outfname)
                            except:
                                try:
                                    shutil.copy2(path,outfname)
                                except:
                                    pass
                    logging.warning(f"saving : writing to: {outfname} from {path}")
            else:
                logging.warning(f"skipping : already exists {outfname} from {path}")


    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit


