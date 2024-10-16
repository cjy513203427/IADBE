#!/bin/bash

datasets=('carpet')
config_file="../configs/models/uflow.yaml"
data_config_file="../configs/data/mvtec.yaml"

for dataset in "${datasets[@]}"
do
    command="anomalib train --data $data_config_file --data.category $dataset --config $config_file"
    echo "Running command: $command"
    # Excute command
    $command
done
