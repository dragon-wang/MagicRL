#!/bin/bash

python mujoco_td3bc.py --env=hopper-medium-v2 --learn_id=finetune/hop-m-v2/off/td3bc/sd10 --seed=10
python mujoco_td3bc.py --env=halfcheetah-medium-v2 --learn_id=finetune/half-m-v2/off/td3bc/sd10 --seed=10
python mujoco_td3bc.py --env=walker2d-medium-v2 --learn_id=finetune/walk-m-v2/off/td3bc/sd10 --seed=10

python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=5 --offline_id=finetune/hop-m-v2/off/td3bc/sd10 --buffer_type=1 --seed=10 --learn_id=finetune/hop-m-v2/ft/td3/bt1-sd10
python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=5 --offline_id=finetune/hop-m-v2/off/td3bc/sd10 --buffer_type=2 --seed=10 --learn_id=finetune/hop-m-v2/ft/td3/bt2-sd10
python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=5 --offline_id=finetune/hop-m-v2/off/td3bc/sd10 --buffer_type=3 --seed=10 --learn_id=finetune/hop-m-v2/ft/td3/bt2-sd10
python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=5 --offline_id=finetune/hop-m-v2/off/td3bc/sd10 --buffer_type=1 --seed=10 --no_optim --learn_id=finetune/hop-m-v2/ft/td3/bt1-sd10-nop
python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=5 --offline_id=finetune/hop-m-v2/off/td3bc/sd10 --buffer_type=2 --seed=10 --no_optim --learn_id=finetune/hop-m-v2/ft/td3/bt2-sd10-nop
python mujoco_td3_ft.py --env=hopper-medium-v2 --eval_num=5 --offline_id=finetune/hop-m-v2/off/td3bc/sd10 --buffer_type=3 --seed=10 --no_optim --learn_id=finetune/hop-m-v2/ft/td3/bt2-sd10-nop

python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=5 --offline_id=finetune/half-m-v2/off/td3bc/sd10 --buffer_type=1 --seed=10 --learn_id=finetune/half-m-v2/ft/td3/bt1-sd10
python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=5 --offline_id=finetune/half-m-v2/off/td3bc/sd10 --buffer_type=2 --seed=10 --learn_id=finetune/half-m-v2/ft/td3/bt2-sd10
python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=5 --offline_id=finetune/half-m-v2/off/td3bc/sd10 --buffer_type=3 --seed=10 --learn_id=finetune/half-m-v2/ft/td3/bt2-sd10
python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=5 --offline_id=finetune/half-m-v2/off/td3bc/sd10 --buffer_type=1 --seed=10 --no_optim --learn_id=finetune/half-m-v2/ft/td3/bt1-sd10-nop
python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=5 --offline_id=finetune/half-m-v2/off/td3bc/sd10 --buffer_type=2 --seed=10 --no_optim --learn_id=finetune/half-m-v2/ft/td3/bt2-sd10-nop
python mujoco_td3_ft.py --env=halfcheetah-medium-v2 --eval_num=5 --offline_id=finetune/half-m-v2/off/td3bc/sd10 --buffer_type=3 --seed=10 --no_optim --learn_id=finetune/half-m-v2/ft/td3/bt2-sd10-nop

python mujoco_td3_ft.py --env=walker2d-medium-v2 --eval_num=5 --offline_id=finetune/walk-m-v2/off/td3bc/sd10 --buffer_type=1 --seed=10 --learn_id=finetune/walk-m-v2/ft/td3/bt1-sd10
python mujoco_td3_ft.py --env=walker2d-medium-v2 --eval_num=5 --offline_id=finetune/walk-m-v2/off/td3bc/sd10 --buffer_type=2 --seed=10 --learn_id=finetune/walk-m-v2/ft/td3/bt2-sd10
python mujoco_td3_ft.py --env=walker2d-medium-v2 --eval_num=5 --offline_id=finetune/walk-m-v2/off/td3bc/sd10 --buffer_type=3 --seed=10 --learn_id=finetune/walk-m-v2/ft/td3/bt2-sd10
python mujoco_td3_ft.py --env=walker2d-medium-v2 --eval_num=5 --offline_id=finetune/walk-m-v2/off/td3bc/sd10 --buffer_type=1 --seed=10 --no_optim --learn_id=finetune/walk-m-v2/ft/td3/bt1-sd10-nop
python mujoco_td3_ft.py --env=walker2d-medium-v2 --eval_num=5 --offline_id=finetune/walk-m-v2/off/td3bc/sd10 --buffer_type=2 --seed=10 --no_optim --learn_id=finetune/walk-m-v2/ft/td3/bt2-sd10-nop
python mujoco_td3_ft.py --env=walker2d-medium-v2 --eval_num=5 --offline_id=finetune/walk-m-v2/off/td3bc/sd10 --buffer_type=3 --seed=10 --no_optim --learn_id=finetune/walk-m-v2/ft/td3/bt2-sd10-nop