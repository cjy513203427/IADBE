#!/bin/bash

# Define the datasets array
datasets=('screw' 'pill' 'capsule' 'carpet' 'grid' 'tile' 'wood' 'zipper' 'cable' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' 'leather')

config_file="../configs/models/dfm.yaml"
# Path to the pre-trained model checkpoint
ckpt_path="/home/jinyao/PycharmProjects/IADBE/results/Dfm/MVTec/bottle/latest/weights/lightning/model.ckpt"

# Loop through each dataset and run the anomalib test command
for dataset in "${datasets[@]}"; do
    command="anomalib test --config $config_file --data anomalib.data.MVTec --data.category $dataset --ckpt_path $ckpt_path"
    echo "Running command: $command"
    # Execute command
    $command
done
