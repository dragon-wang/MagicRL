#!/bin/bash

python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=3 --offline_id=td3bc/hop-m-v2 --buffer_type=1 --learn_id=td3bcft/hop-m-v2-type1/seed10 --seed=10
python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=3 --offline_id=td3bc/hop-m-v2 --buffer_type=1 --learn_id=td3bcft/hop-m-v2-type1/seed20 --seed=20
python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=3 --offline_id=td3bc/hop-m-v2 --buffer_type=1 --learn_id=td3bcft/hop-m-v2-type1/seed30 --seed=30
python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=3 --offline_id=td3bc/hop-m-v2 --buffer_type=2 --learn_id=td3bcft/hop-m-v2-type2/seed10 --seed=10
python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=3 --offline_id=td3bc/hop-m-v2 --buffer_type=2 --learn_id=td3bcft/hop-m-v2-type2/seed20 --seed=20
python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=3 --offline_id=td3bc/hop-m-v2 --buffer_type=2 --learn_id=td3bcft/hop-m-v2-type2/seed30 --seed=30


python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=3 --offline_id=td3bc/half-m-v2 --buffer_type=1 --learn_id=td3bcft/half-m-v2-type1/seed10 --seed=10
python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=3 --offline_id=td3bc/half-m-v2 --buffer_type=1 --learn_id=td3bcft/half-m-v2-type1/seed20 --seed=20
python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=3 --offline_id=td3bc/half-m-v2 --buffer_type=1 --learn_id=td3bcft/half-m-v2-type1/seed30 --seed=30
python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=3 --offline_id=td3bc/half-m-v2 --buffer_type=2 --learn_id=td3bcft/half-m-v2-type2/seed10 --seed=10
python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=3 --offline_id=td3bc/half-m-v2 --buffer_type=2 --learn_id=td3bcft/half-m-v2-type2/seed20 --seed=20
python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=3 --offline_id=td3bc/half-m-v2 --buffer_type=2 --learn_id=td3bcft/half-m-v2-type2/seed30 --seed=30