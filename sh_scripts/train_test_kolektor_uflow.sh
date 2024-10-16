#!/bin/bash

config_file="../configs/models/uflow.yaml"

command="anomalib train --data anomalib.data.Kolektor --config $config_file"
echo "Running command: $command"
# Excute command
$command
