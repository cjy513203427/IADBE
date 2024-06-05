#!/bin/bash

datasets=("bagel" "cable_gland" "carrot" "cookie" "dowel" "foam" "peach" "potato" "rope" "tire")
config_file="./configs/models/cfa.yaml"

for dataset in "${datasets[@]}"
do
    command="anomalib train --data anomalib.data.MVTec3D --data.category $dataset --config $config_file"
    echo "Running command: $command"
    # Excute command
    $command
done



