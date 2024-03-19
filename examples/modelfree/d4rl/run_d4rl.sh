#!/bin/bash

# ========================= TD3 =========================
echo "---------Run TD3 in hopper---------"
python mujoco_td3.py --env=hopper-medium-v2 --learn_id=modelfree/d4rl/hop-m-v2/td3/seed10 --seed=10
python mujoco_td3.py --env=hopper-medium-v2 --learn_id=modelfree/d4rl/hop-m-v2/td3/seed20 --seed=20
python mujoco_td3.py --env=hopper-medium-v2 --learn_id=modelfree/d4rl/hop-m-v2/td3/seed30 --seed=30
echo "--------Run TD3 in halfcheetah---------"
python mujoco_td3.py --env=halfcheetah-medium-v2 --learn_id=modelfree/d4rl/half-m-v2/td3/seed10 --seed=10
python mujoco_td3.py --env=halfcheetah-medium-v2 --learn_id=modelfree/d4rl/half-m-v2/td3/seed20 --seed=20
python mujoco_td3.py --env=halfcheetah-medium-v2 --learn_id=modelfree/d4rl/half-m-v2/td3/seed30 --seed=30
echo "--------Run TD3 in walker2d---------"
python mujoco_td3.py --env=walker2d-medium-v2 --learn_id=modelfree/d4rl/walk-m-v2/td3/seed10 --seed=10
python mujoco_td3.py --env=walker2d-medium-v2 --learn_id=modelfree/d4rl/walk-m-v2/td3/seed20 --seed=20
python mujoco_td3.py --env=walker2d-medium-v2 --learn_id=modelfree/d4rl/walk-m-v2/td3/seed30 --seed=30