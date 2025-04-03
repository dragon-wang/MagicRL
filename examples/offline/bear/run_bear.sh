#!/bin/bash

echo "---------Run BEAR in hopper---------"
python mujoco_bear.py --env=hopper-random-v2 --learn_id=offline/bear/mujoco/Hopper-v2/random/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=hopper-random-v2 --learn_id=offline/bear/mujoco/Hopper-v2/random/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=hopper-random-v2 --learn_id=offline/bear/mujoco/Hopper-v2/random/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=hopper-medium-v2 --learn_id=offline/bear/mujoco/Hopper-v2/medium/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=hopper-medium-v2 --learn_id=offline/bear/mujoco/Hopper-v2/medium/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=hopper-medium-v2 --learn_id=offline/bear/mujoco/Hopper-v2/medium/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=hopper-expert-v2 --learn_id=offline/bear/mujoco/Hopper-v2/expert/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=hopper-expert-v2 --learn_id=offline/bear/mujoco/Hopper-v2/expert/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=hopper-expert-v2 --learn_id=offline/bear/mujoco/Hopper-v2/expert/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=hopper-medium-expert-v2 --learn_id=offline/bear/mujoco/Hopper-v2/medium-expert/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=hopper-medium-expert-v2 --learn_id=offline/bear/mujoco/Hopper-v2/medium-expert/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=hopper-medium-expert-v2 --learn_id=offline/bear/mujoco/Hopper-v2/medium-expert/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=hopper-medium-replay-v2 --learn_id=offline/bear/mujoco/Hopper-v2/medium-replay/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=hopper-medium-replay-v2 --learn_id=offline/bear/mujoco/Hopper-v2/medium-replay/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=hopper-medium-replay-v2 --learn_id=offline/bear/mujoco/Hopper-v2/medium-replay/seed30 --kernel_type=laplacian --seed=30

echo "--------Run BEAR in halfcheetah---------"
python mujoco_bear.py --env=halfcheetah-random-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/random/seed10 --kernel_type=gaussian --seed=10
python mujoco_bear.py --env=halfcheetah-random-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/random/seed20 --kernel_type=gaussian --seed=20
python mujoco_bear.py --env=halfcheetah-random-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/random/seed30 --kernel_type=gaussian --seed=30

python mujoco_bear.py --env=halfcheetah-medium-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/medium/seed10 --kernel_type=gaussian --seed=10
python mujoco_bear.py --env=halfcheetah-medium-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/medium/seed20 --kernel_type=gaussian --seed=20
python mujoco_bear.py --env=halfcheetah-medium-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/medium/seed30 --kernel_type=gaussian --seed=30

python mujoco_bear.py --env=halfcheetah-expert-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/expert/seed10 --kernel_type=gaussian --seed=10
python mujoco_bear.py --env=halfcheetah-expert-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/expert/seed20 --kernel_type=gaussian --seed=20
python mujoco_bear.py --env=halfcheetah-expert-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/expert/seed30 --kernel_type=gaussian --seed=30

python mujoco_bear.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/medium-expert/seed10 --kernel_type=gaussian --seed=10
python mujoco_bear.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/medium-expert/seed20 --kernel_type=gaussian --seed=20
python mujoco_bear.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/medium-expert/seed30 --kernel_type=gaussian --seed=30

python mujoco_bear.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/medium-replay/seed10 --kernel_type=gaussian --seed=10
python mujoco_bear.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/medium-replay/seed20 --kernel_type=gaussian --seed=20
python mujoco_bear.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/bear/mujoco/HalfCheetah-v2/medium-replay/seed30 --kernel_type=gaussian --seed=30

echo "--------Run BEAR in walker2d---------"
python mujoco_bear.py --env=walker2d-random-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/random/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=walker2d-random-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/random/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=walker2d-random-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/random/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=walker2d-medium-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/medium/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=walker2d-medium-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/medium/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=walker2d-medium-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/medium/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=walker2d-expert-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/expert/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=walker2d-expert-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/expert/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=walker2d-expert-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/expert/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=walker2d-medium-expert-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/medium-expert/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=walker2d-medium-expert-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/medium-expert/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=walker2d-medium-expert-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/medium-expert/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=walker2d-medium-replay-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/medium-replay/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=walker2d-medium-replay-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/medium-replay/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=walker2d-medium-replay-v2 --learn_id=offline/bear/mujoco/Walker2d-v2/medium-replay/seed30 --kernel_type=laplacian --seed=30