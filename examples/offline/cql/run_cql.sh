#!/bin/bash

echo "---------Run CQL in hopper---------"
python mujoco_cql.py --env=hopper-random-v2 --learn_id=offline/cql/mujoco/Hopper-v2/random/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=hopper-random-v2 --learn_id=offline/cql/mujoco/Hopper-v2/random/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=hopper-random-v2 --learn_id=offline/cql/mujoco/Hopper-v2/random/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=hopper-medium-v2 --learn_id=offline/cql/mujoco/Hopper-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=hopper-medium-v2 --learn_id=offline/cql/mujoco/Hopper-v2/medium/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=hopper-medium-v2 --learn_id=offline/cql/mujoco/Hopper-v2/medium/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=hopper-expert-v2 --learn_id=offline/cql/mujoco/Hopper-v2/expert/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=hopper-expert-v2 --learn_id=offline/cql/mujoco/Hopper-v2/expert/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=hopper-expert-v2 --learn_id=offline/cql/mujoco/Hopper-v2/expert/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=hopper-medium-expert-v2 --learn_id=offline/cql/mujoco/Hopper-v2/medium-expert/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=hopper-medium-expert-v2 --learn_id=offline/cql/mujoco/Hopper-v2/medium-expert/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=hopper-medium-expert-v2 --learn_id=offline/cql/mujoco/Hopper-v2/medium-expert/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=hopper-medium-replay-v2 --learn_id=offline/cql/mujoco/Hopper-v2/medium-replay/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=hopper-medium-replay-v2 --learn_id=offline/cql/mujoco/Hopper-v2/medium-replay/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=hopper-medium-replay-v2 --learn_id=offline/cql/mujoco/Hopper-v2/medium-replay/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

echo "--------Run CQL in halfcheetah---------"
python mujoco_cql.py --env=halfcheetah-random-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/random/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=halfcheetah-random-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/random/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=halfcheetah-random-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/random/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=halfcheetah-medium-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=halfcheetah-medium-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/medium/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=halfcheetah-medium-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/medium/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=halfcheetah-expert-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/expert/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=halfcheetah-expert-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/expert/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=halfcheetah-expert-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/expert/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/medium-expert/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/medium-expert/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/medium-expert/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/medium-replay/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/medium-replay/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/cql/mujoco/HalfCheetah-v2/medium-replay/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

echo "--------Run CQL in walker2d---------"
python mujoco_cql.py --env=walker2d-random-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/random/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=walker2d-random-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/random/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=walker2d-random-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/random/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=walker2d-medium-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=walker2d-medium-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/medium/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=walker2d-medium-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/medium/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=walker2d-expert-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/expert/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=walker2d-expert-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/expert/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=walker2d-expert-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/expert/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=walker2d-medium-expert-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/medium-expert/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=walker2d-medium-expert-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/medium-expert/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=walker2d-medium-expert-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/medium-expert/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=walker2d-medium-replay-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/medium-replay/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=walker2d-medium-replay-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/medium-replay/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=walker2d-medium-replay-v2 --learn_id=offline/cql/mujoco/Walker2d-v2/medium-replay/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30