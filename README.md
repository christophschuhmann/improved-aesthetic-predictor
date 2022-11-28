# CLIP+MLP Aesthetic Score Predictor

Train, use and visualize an aesthetic score predictor ( how much people like on average an image ) based on a simple neural net that takes CLIP embeddings as inputs.


Link to the AVA training data ( already prepared) :
https://drive.google.com/drive/folders/186XiniJup5Rt9FXsHiAGWhgWz-nmCK_r?usp=sharing


Visualizations of all images from LAION 5B (english subset with 2.37B images) in 40 buckets with the model sac+logos+ava1-l14-linearMSE.pth:
http://captions.christoph-schuhmann.de/aesthetic_viz_laion_sac+logos+ava1-l14-linearMSE-en-2.37B.html


Now supports parallel image and video processing, and copy/extraction upon match with a score threshold.

Support for parallel computation:
    - for still images : parallel resizing of incoming images
    - for video files : a per-file decompression and prediction parallelism
    - `--parallel CORES`  - uses `#CORES` number for parallel computation

Example:
`
python rank_images.py --maxw 1000 --log_level=WARNING --v_when adaptive_mod:30:5.0 --video --v_output="%D/good_frames/%F_%d.jpg" --v_cutoff=5.3 --parallel=11 /mnt/u/photos/by_date/2022/2022-11-2*/C*.MP4
`