#!/bin/bash

datasets=('screw' 'pill' 'capsule' 'carpet' 'grid' 'tile' 'wood' 'zipper' 'cable' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' 'leather')
data_config_file="../configs/data/mvtec.yaml"
config_file="../configs/models/fre.yaml"

for dataset in "${datasets[@]}"
do
    command="anomalib train --data $data_config_file --data.category $dataset --config $config_file"
    echo "Running command: $command"
    # Excute command
    $command
done
