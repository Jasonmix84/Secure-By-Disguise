#!/bin/bash

# First get the weights needed for evaluating recovered images
# python3 classify.py --trans 4 --model_name resnet34 --cvidx Breast_resized-256_cvidx.csv --nfolds 5 --n_classes 2 >> Breast_Attack_Results

# Then after getting the weights use prepare_recovered_csv to prepare csv for the attacked images
# python3 prepare_recovered_csv.py --base_csv Breast_resized-256_cvidx.csv --encrypted_dir ../ModelData/Breast/resized-256 --recovered_dir ../Data/Breast/recovered/recovered-RMT-B2N0-kp1 --output_csv ./recovered/Breast_recovered-RMT-B2N0-kp1.csv

# Now that you have the csv for the attacked images, where all of the images are in the same split you can evaluate the model
# python3 evaluate_on_recovered.py --recovered_cvidx ./recovered/Breast_recovered-RMT-B2N0-kp1.csv --checkpoint_dir ./models_temp --model_name resnet34 --n_classes 2 --nfolds 5 --image_size 256 --trans 4 --base_cvidx Breast_resized-256_cvidx.csv >> Breast_Attack_Results


# create all the csvs for the attacked images
for settings in "B2N0" "B2N2" "B2N4" "B2N8" "B2N256" "B4N0" "B4N2" "B4N4" "B4N8" "B4N256" "B8N0" "B8N2" "B8N4" "B8N8" "B8N256"; do
    for kp in 1 2 3 10 25 50 100; do
        python3 prepare_recovered_csv.py --base_csv Breast_resized-256_cvidx.csv --encrypted_dir ../ModelData/Breast/resized-256 --recovered_dir ../Data/Breast/recovered/recovered-RMT-${settings}-kp${kp} --output_csv ./recovered/Breast_recovered-RMT-${settings}-kp${kp}.csv
        python3 evaluate_on_recovered.py --recovered_cvidx ./recovered/Breast_recovered-RMT-${settings}-kp${kp}.csv --checkpoint_dir ./models_temp --model_name resnet34 --n_classes 2 --nfolds 5 --image_size 256 --trans 4 --base_cvidx Breast_resized-256_cvidx.csv >> Breast_Attack_Results
    done
done