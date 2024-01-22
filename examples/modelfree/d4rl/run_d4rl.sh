#!/bin/bash

python mujoco_td3.py --env=hopper-medium-v2 --eval_num=3 --learn_id=td3/hop-m-v2
python mujoco_td3.py --env=halfcheetah-medium-v2 --eval_num=3 --learn_id=td3/half-m-v2
