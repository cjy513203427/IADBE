#!/bin/bash

data_config_file="./configs/data/avenue.yaml"
config_file="./configs/models/efficient_ad.yaml"

command="anomalib train --data $data_config_file --config $config_file"
echo "Running command: $command"
# Excute command
$command
