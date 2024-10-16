#!/bin/bash

datasets=("candle"
            "capsules"
            "cashew"
            "chewinggum"
            "fryum"
            "macaroni1"
            "macaroni2"
            "pcb1"
            "pcb2"
            "pcb3"
            "pcb4"
            "pipe_fryum")

data_config_file="../configs/data/visa.yaml"
config_file="../configs/models/draem.yaml"

for dataset in "${datasets[@]}"
do
    command="anomalib train --data $data_config_file --data.category $dataset --config $config_file"
    echo "Running command: $command"
    # Excute command
    $command
done
