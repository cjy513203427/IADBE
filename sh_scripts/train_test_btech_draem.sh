#!/bin/bash

datasets=("01" "02" "03")
data_config_file=".../configs/data/btech.yaml"
config_file=".../configs/models/draem.yaml"

for dataset in "${datasets[@]}"
do
    command="anomalib train --data $data_config_file --data.category $dataset --config $config_file"
    echo "Running command: $command"
    # Excute command
    $command
done
