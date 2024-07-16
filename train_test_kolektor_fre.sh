#!/bin/bash

data_config_file="./configs/data/kolektor.yaml"
config_file="./configs/models/fre.yaml"

command="anomalib train --data $data_config_file --config $config_file"
echo "Running command: $command"
# Excute command
$command
