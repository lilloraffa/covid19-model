#!/bin/bash

clear
source $(conda info --base)/etc/profile.d/conda.sh
conda activate data
python python/calibration.py --model_name=Model-v6_err1abs --run_partial 0 10 \
    --param_exclude_window rg ra \
    --model_stepsforward=150 &
python python/calibration.py --model_name=Model-v6_err1abs --run_partial 10 -999 \
    --param_exclude_window rg ra \
    --model_stepsforward=150 &
