#!/bin/bash

python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=3 --offline_id=td3bc/hop-m-v2 --buffer_type=1 --learn_id=td3bcft/hop-m-v2-type1
python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=3 --offline_id=td3bc/hop-m-v2 --buffer_type=2 --learn_id=td3bcft/hop-m-v2-type2

python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=3 --offline_id=td3bc/half-m-v2 --buffer_type=1 --learn_id=td3bcft/half-m-v2-type1
python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=3 --offline_id=td3bc/half-m-v2 --buffer_type=2 --learn_id=td3bcft/half-m-v2-type2