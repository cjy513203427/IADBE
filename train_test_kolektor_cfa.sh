#!/bin/bash

config_file="./configs/models/cfa.yaml"

command="anomalib train --data anomalib.data.Kolektor --config $config_file"
echo "Running command: $command"
# Excute command
$command
