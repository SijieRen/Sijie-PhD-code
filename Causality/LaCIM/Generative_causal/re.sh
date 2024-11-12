#!/bin/bash
for ((i=1;i<=$1;i++)); do
    echo "${@:3}"
    CUDA_VISIBLE_DEVICES=$2 python ${@:3}
done