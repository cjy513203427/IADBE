#!/bin/bash

data_config_file="./configs/data/custom_dataset_normal_abnormal_cardboard.yaml"

command="anomalib train --data $data_config_file --model anomalib.models.Csflow"
echo "Running command: $command"
# Excute command
$command
