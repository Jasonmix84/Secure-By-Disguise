#!/bin/bash

#prepare cv idx for segmentation
python3 prepare-cv-seg.py ../ModelData/CVC/cleaned-OG CVC-OG-cvidx.csv

#run segmentation on CVC
python3 segmentation.py --cvidx CVC-OG-cvidx.csv --nfolds 5 --model_name unet --trans 1 --image_size 512 >> CVC-RMT-Result 
