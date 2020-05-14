#!/bin/bash

gpu=${gpu:-1,2,3}
conf_dir=${conf_dir:-/configs_folder/}
data_dir=${data_dir:-/data/}
results_dir=${results_dir:-/results/}

while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

IFS=', ' read -r -a gpu_ids <<< "$gpu"

nvidia-docker build --no-cache . -f Dockerfile.gpu -t mtl_exp

chmod -R 777 . $results_dir

i=0
for f in $conf_dir/*.yaml; do
    g=$(($i % 3))
    echo "Running configurations file: $f, gpu:" "${gpu_ids[$g]}"
    nvidia-docker run -d --runtime=nvidia  \
        -e NVIDIA_VISIBLE_DEVICES=${gpu_ids[$g]}  \
        -v $results_dir:/app/logs  \
        -v $data_dir:/app/data  \
        -v $f:/app/config/config.yaml  \
        mtl_exp

    ((i++))
done


echo "Experiments are running..."
