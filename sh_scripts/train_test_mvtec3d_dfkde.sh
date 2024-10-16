#!/bin/bash

datasets=("bagel" "cable_gland" "carrot" "cookie" "dowel" "foam" "peach" "potato" "rope" "tire")
data_config_file="../configs/data/mvtec_3d.yaml"
config_file="../configs/models/dfkde.yaml"

for dataset in "${datasets[@]}"
do
    command="anomalib train --data $data_config_file --data.category $dataset --config $config_file"
    echo "Running command: $command"
    # Excute command
    $command
done



