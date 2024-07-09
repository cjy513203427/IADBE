#!/bin/bash

data_config_file="./configs/data/btech.yaml"
config_file="./configs/models/draem.yaml"

command="anomalib train --data $data_config_file --config $config_file"
echo "Running command: $command"
# Excute command
$command
