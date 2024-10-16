#!/bin/bash

# shellcheck disable=SC2054
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
config_file="../configs/models/dsr.yaml"

for dataset in "${datasets[@]}"
do
    command="anomalib train --data anomalib.data.Visa --data.category $dataset --config $config_file"
    echo "Running command: $command"
    # Excute command
    $command
done
