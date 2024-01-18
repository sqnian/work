#!/bin/bash
set -eoxu pipefail

cd ../data_set
if [ ! -d flower_data ];then
    mkdir flower_data 
fi
cd flower_data
if [ ! -d flower_photos ];then
        wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
        tar -zxvf flower_photos.tgz
fi

cd ..
python3 split_data.py
