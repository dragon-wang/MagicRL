#!/bin/bash

echo "---------Run TD3BC in hopper---------"
python mujoco_td3bc.py --env=hopper-random-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/random/seed10 --seed=10
python mujoco_td3bc.py --env=hopper-random-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/random/seed20 --seed=20
python mujoco_td3bc.py --env=hopper-random-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/random/seed30 --seed=30

python mujoco_td3bc.py --env=hopper-medium-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/medium/seed10 --seed=10
python mujoco_td3bc.py --env=hopper-medium-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/medium/seed20 --seed=20
python mujoco_td3bc.py --env=hopper-medium-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/medium/seed30 --seed=30

python mujoco_td3bc.py --env=hopper-expert-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/expert/seed10 --seed=10
python mujoco_td3bc.py --env=hopper-expert-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/expert/seed20 --seed=20
python mujoco_td3bc.py --env=hopper-expert-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/expert/seed30 --seed=30

python mujoco_td3bc.py --env=hopper-medium-expert-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/medium-expert/seed10 --seed=10
python mujoco_td3bc.py --env=hopper-medium-expert-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/medium-expert/seed20 --seed=20
python mujoco_td3bc.py --env=hopper-medium-expert-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/medium-expert/seed30 --seed=30

python mujoco_td3bc.py --env=hopper-medium-replay-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/medium-replay/seed10 --seed=10
python mujoco_td3bc.py --env=hopper-medium-replay-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/medium-replay/seed20 --seed=20
python mujoco_td3bc.py --env=hopper-medium-replay-v2 --learn_id=offline/td3bc/mujoco/Hopper-v2/medium-replay/seed30 --seed=30

echo "--------Run TD3BC in halfcheetah---------"
python mujoco_td3bc.py --env=halfcheetah-random-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/random/seed10 --seed=10
python mujoco_td3bc.py --env=halfcheetah-random-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/random/seed20 --seed=20
python mujoco_td3bc.py --env=halfcheetah-random-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/random/seed30 --seed=30

python mujoco_td3bc.py --env=halfcheetah-medium-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/medium/seed10 --seed=10
python mujoco_td3bc.py --env=halfcheetah-medium-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/medium/seed20 --seed=20
python mujoco_td3bc.py --env=halfcheetah-medium-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/medium/seed30 --seed=30

python mujoco_td3bc.py --env=halfcheetah-expert-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/expert/seed10 --seed=10
python mujoco_td3bc.py --env=halfcheetah-expert-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/expert/seed20 --seed=20
python mujoco_td3bc.py --env=halfcheetah-expert-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/expert/seed30 --seed=30

python mujoco_td3bc.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/medium-expert/seed10 --seed=10
python mujoco_td3bc.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/medium-expert/seed20 --seed=20
python mujoco_td3bc.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/medium-expert/seed30 --seed=30

python mujoco_td3bc.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/medium-replay/seed10 --seed=10
python mujoco_td3bc.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/medium-replay/seed20 --seed=20
python mujoco_td3bc.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/td3bc/mujoco/HalfCheetah-v2/medium-replay/seed30 --seed=30

echo "--------Run TD3BC in walker2d---------"
python mujoco_td3bc.py --env=walker2d-random-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/random/seed10 --seed=10
python mujoco_td3bc.py --env=walker2d-random-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/random/seed20 --seed=20
python mujoco_td3bc.py --env=walker2d-random-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/random/seed30 --seed=30

python mujoco_td3bc.py --env=walker2d-medium-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/medium/seed10 --seed=10
python mujoco_td3bc.py --env=walker2d-medium-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/medium/seed20 --seed=20
python mujoco_td3bc.py --env=walker2d-medium-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/medium/seed30 --seed=30

python mujoco_td3bc.py --env=walker2d-expert-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/expert/seed10 --seed=10
python mujoco_td3bc.py --env=walker2d-expert-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/expert/seed20 --seed=20
python mujoco_td3bc.py --env=walker2d-expert-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/expert/seed30 --seed=30

python mujoco_td3bc.py --env=walker2d-medium-expert-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/medium-expert/seed10 --seed=10
python mujoco_td3bc.py --env=walker2d-medium-expert-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/medium-expert/seed20 --seed=20
python mujoco_td3bc.py --env=walker2d-medium-expert-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/medium-expert/seed30 --seed=30

python mujoco_td3bc.py --env=walker2d-medium-replay-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/medium-replay/seed10 --seed=10
python mujoco_td3bc.py --env=walker2d-medium-replay-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/medium-replay/seed20 --seed=20
python mujoco_td3bc.py --env=walker2d-medium-replay-v2 --learn_id=offline/td3bc/mujoco/Walker2d-v2/medium-replay/seed30 --seed=30