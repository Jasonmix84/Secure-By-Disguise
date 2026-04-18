#!/bin/bash

# Create Master CSV file first then run classifcation

# Example call pass dataset path and csv file name to create the master csv file with cvidx for 5 fold cross validation
python3 prepare-cv.py ../ModelData/Breast/resized-256 Breast_OG_cvidx.csv 

# Train model
python3 classify.py --model_name "vgg16" --cvidx Breast_OG_cvidx.csv --nfolds 5 --n_classes 2 >> B_og_result

