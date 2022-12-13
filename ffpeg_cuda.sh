#!/bin/bash
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
vi Makefile   # change the first line to PREFIX = ${CONDA_PREFIX}
make install
cd ..

git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
git checkout n5.1.2
conda install nasm
./configure --prefix=$CONDA_PREFIX --enable-cuda-sdk --enable-cuvid --enable-nvenc --enable-nonfree --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64
make -j 10
make install