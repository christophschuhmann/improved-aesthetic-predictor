#!/bin/bash
echo "************************  try building ffmpeg  *******************"
echo "Install nv-codec-headers"

git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git

cd nv-codec-headers && make -j4 && make install && cd ..

wget https://ffmpeg.org/releases/ffmpeg-5.1.2.tar.bz2
tar -xf ffmpeg-5.1.2.tar.bz2
cd ffmpeg-5.1.2
./configure --enable-cuda --enable-cuvid --enable-nvenc --enable-nonfree --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --enable-shared

make -j4
sudo checkinstall
cd ..

echo "************************  setting cuda path  *******************"
echo "export PATH=/usr/local/cuda/bin:/ffmpeg/libavdevice:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/targets/x86_64-linux/include:/usr/local/cuda/lib64:/usr/local/cuda/include:/usr/local/cuda/extras/CUPTI/lib64:/ffmpeg/libavdevice" >> ~/.bashrc

echo "************************  try building pyav  *******************"
git clone -b hwaccel https://github.com/rvillalba-novetta/PyAV.git

cd PyAV
source scripts/activate.sh
pip3 install -r tests/requirements.txt
make -j4

python3 setup.py install
cd ..

ldconfig /usr/local/cuda/lib64
echo "*********************   ffmpeg and pyav are built successfuly ***************"
