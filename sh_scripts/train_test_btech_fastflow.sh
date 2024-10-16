#!/bin/bash

datasets=("01" "02" "03")
config_file="../configs/models/fastflow.yaml"

for dataset in "${datasets[@]}"
do
    command="anomalib train --data anomalib.data.BTech --data.category $dataset --config $config_file"
    echo "Running command: $command"
    # Excute command
    $command
done
