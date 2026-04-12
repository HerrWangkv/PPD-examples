#!/bin/bash

# Define paths
# HOST path where the data actually lives
HOST_DATA_ROOT="/mrtstorage/users/kwang/synthia_sim2real"

bash eval_synthia.sh data/synthia > synthia_eval.log

for folder_path in "$HOST_DATA_ROOT"/*; do
    if [ -d "$folder_path" ]; then
        folder_name=$(basename "$folder_path")
        
        # Construct the path as seen INSIDE the container
        # Since we mount HOST_DATA_ROOT to CONTAINER_DATA_ROOT
        container_path="data/synthia_sim2real/$folder_name"
        
        bash eval_synthia.sh "$container_path" > "${folder_name}_eval.log"
    fi
done

echo "All evaluations completed."