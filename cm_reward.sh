#!/bin/bash
# use bfloat16 for training!

CMD="""
accelerate launch \
    --gpu_ids 0,2,4,5,6,7 \
    `pwd`/scripts_reward_modeling/reward_model_train.py \
"""

echo $CMD
eval "$CMD"