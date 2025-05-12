#!/bin/bash
# use bfloat16 for training!

CMD="""
accelerate launch \
    --gpu_ids 0,2,3,4,5,6,7 \
    `pwd`/scripts_ppo/generation_ppo_model_train.py \
"""

echo $CMD
eval "$CMD"