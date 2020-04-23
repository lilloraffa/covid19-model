#!/bin/bash

clear
source $(conda info --base)/etc/profile.d/conda.sh
conda activate data
python python/calibration.py --model_name=Model-v5_err1abs-fixedT --param_fixed --run_partial 0 10 \
    --param_exclude_finetuner t1 tgi2 tgn2 ta2 \
    --param_exclude_window rg ra t1 tgi2 tgn2 ta2 \
    --model_stepsforward=200 &

python python/calibration.py --model_name=Model-v5_err1abs-fixedT --param_fixed --run_partial 10 -999 \
    --param_exclude_finetuner t1 tgi2 tgn2 ta2 \
    --param_exclude_window rg ra t1 tgi2 tgn2 ta2 \
    --model_stepsforward=200 &