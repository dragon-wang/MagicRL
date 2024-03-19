#!/bin/bash

# ========================= BCQ =========================
echo "---------Run BCQ in hopper---------"
python mujoco_bcq.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/bcq/seed10 --seed=10
python mujoco_bcq.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/bcq/seed20 --seed=20
python mujoco_bcq.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/bcq/seed30 --seed=30

python mujoco_bcq.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/bcq/seed10 --seed=10
python mujoco_bcq.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/bcq/seed20 --seed=20
python mujoco_bcq.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/bcq/seed30 --seed=30

python mujoco_bcq.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/bcq/seed10 --seed=10
python mujoco_bcq.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/bcq/seed20 --seed=20
python mujoco_bcq.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/bcq/seed30 --seed=30

python mujoco_bcq.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/bcq/seed10 --seed=10
python mujoco_bcq.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/bcq/seed20 --seed=20
python mujoco_bcq.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/bcq/seed30 --seed=30

python mujoco_bcq.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/bcq/seed10 --seed=10
python mujoco_bcq.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/bcq/seed20 --seed=20
python mujoco_bcq.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/bcq/seed30 --seed=30

echo "---------Run BCQ in halfcheetah---------"
python mujoco_bcq.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/bcq/seed10 --seed=10
python mujoco_bcq.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/bcq/seed20 --seed=20
python mujoco_bcq.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/bcq/seed30 --seed=30

python mujoco_bcq.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/bcq/seed10 --seed=10
python mujoco_bcq.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/bcq/seed20 --seed=20
python mujoco_bcq.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/bcq/seed30 --seed=30

python mujoco_bcq.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/bcq/seed10 --seed=10
python mujoco_bcq.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/bcq/seed20 --seed=20
python mujoco_bcq.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/bcq/seed30 --seed=30

python mujoco_bcq.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/bcq/seed10 --seed=10
python mujoco_bcq.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/bcq/seed20 --seed=20
python mujoco_bcq.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/bcq/seed30 --seed=30

python mujoco_bcq.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/bcq/seed10 --seed=10
python mujoco_bcq.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/bcq/seed20 --seed=20
python mujoco_bcq.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/bcq/seed30 --seed=30

echo "---------Run BCQ in walker2d---------"
python mujoco_bcq.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/bcq/seed10 --seed=10
python mujoco_bcq.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/bcq/seed20 --seed=20
python mujoco_bcq.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/bcq/seed30 --seed=30

python mujoco_bcq.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/bcq/seed10 --seed=10
python mujoco_bcq.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/bcq/seed20 --seed=20
python mujoco_bcq.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/bcq/seed30 --seed=30

python mujoco_bcq.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/bcq/seed10 --seed=10
python mujoco_bcq.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/bcq/seed20 --seed=20
python mujoco_bcq.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/bcq/seed30 --seed=30

python mujoco_bcq.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/bcq/seed10 --seed=10
python mujoco_bcq.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/bcq/seed20 --seed=20
python mujoco_bcq.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/bcq/seed30 --seed=30

python mujoco_bcq.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/bcq/seed10 --seed=10
python mujoco_bcq.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/bcq/seed20 --seed=20
python mujoco_bcq.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/bcq/seed30 --seed=30

# ========================= BEAR =========================
echo "---------Run BEAR in hopper---------"
python mujoco_bear.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/bear/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/bear/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/bear/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/bear/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/bear/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/bear/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/bear/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/bear/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/bear/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/bear/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/bear/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/bear/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/bear/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/bear/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/bear/seed30 --kernel_type=laplacian --seed=30

echo "--------Run BEAR in halfcheetah---------"
python mujoco_bear.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/bear/seed10 --kernel_type=gaussian --seed=10
python mujoco_bear.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/bear/seed20 --kernel_type=gaussian --seed=20
python mujoco_bear.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/bear/seed30 --kernel_type=gaussian --seed=30

python mujoco_bear.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/bear/seed10 --kernel_type=gaussian --seed=10
python mujoco_bear.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/bear/seed20 --kernel_type=gaussian --seed=20
python mujoco_bear.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/bear/seed30 --kernel_type=gaussian --seed=30

python mujoco_bear.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/bear/seed10 --kernel_type=gaussian --seed=10
python mujoco_bear.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/bear/seed20 --kernel_type=gaussian --seed=20
python mujoco_bear.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/bear/seed30 --kernel_type=gaussian --seed=30

python mujoco_bear.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/bear/seed10 --kernel_type=gaussian --seed=10
python mujoco_bear.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/bear/seed20 --kernel_type=gaussian --seed=20
python mujoco_bear.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/bear/seed30 --kernel_type=gaussian --seed=30

python mujoco_bear.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/bear/seed10 --kernel_type=gaussian --seed=10
python mujoco_bear.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/bear/seed20 --kernel_type=gaussian --seed=20
python mujoco_bear.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/bear/seed30 --kernel_type=gaussian --seed=30

echo "--------Run BEAR in walker2d---------"
python mujoco_bear.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/bear/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/bear/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/bear/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/bear/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/bear/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/bear/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/bear/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/bear/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/bear/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/bear/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/bear/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/bear/seed30 --kernel_type=laplacian --seed=30

python mujoco_bear.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/bear/seed10 --kernel_type=laplacian --seed=10
python mujoco_bear.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/bear/seed20 --kernel_type=laplacian --seed=20
python mujoco_bear.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/bear/seed30 --kernel_type=laplacian --seed=30

# ========================= CQL =========================
echo "---------Run CQL in hopper---------"
python mujoco_cql.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

echo "--------Run CQL in halfcheetah---------"
python mujoco_cql.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

echo "--------Run CQL in walker2d---------"
python mujoco_cql.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

python mujoco_cql.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/cql/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10
python mujoco_cql.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/cql/seed20 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=20
python mujoco_cql.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/cql/seed30 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=30

# ========================= PLAS =========================
echo "---------Run PLAS in hopper---------"
python mujoco_plas.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/plas/seed10 --seed=10
python mujoco_plas.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/plas/seed20 --seed=20
python mujoco_plas.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/plas/seed30 --seed=30

python mujoco_plas.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/plas/seed10 --seed=10
python mujoco_plas.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/plas/seed20 --seed=20
python mujoco_plas.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/plas/seed30 --seed=30

python mujoco_plas.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/plas/seed10 --seed=10
python mujoco_plas.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/plas/seed20 --seed=20
python mujoco_plas.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/plas/seed30 --seed=30

python mujoco_plas.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/plas/seed10 --seed=10
python mujoco_plas.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/plas/seed20 --seed=20
python mujoco_plas.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/plas/seed30 --seed=30

# The network structure should be changed in small dataset.
# python mujoco_plas.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/plas/seed10 --seed=10
# python mujoco_plas.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/plas/seed20 --seed=20
# python mujoco_plas.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/plas/seed30 --seed=30

echo "--------Run PLAS in halfcheetah---------"
python mujoco_plas.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/plas/seed10 --seed=10
python mujoco_plas.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/plas/seed20 --seed=20
python mujoco_plas.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/plas/seed30 --seed=30

python mujoco_plas.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/plas/seed10 --seed=10
python mujoco_plas.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/plas/seed20 --seed=20
python mujoco_plas.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/plas/seed30 --seed=30

python mujoco_plas.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/plas/seed10 --seed=10
python mujoco_plas.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/plas/seed20 --seed=20
python mujoco_plas.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/plas/seed30 --seed=30

python mujoco_plas.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/plas/seed10 --seed=10
python mujoco_plas.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/plas/seed20 --seed=20
python mujoco_plas.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/plas/seed30 --seed=30

# python mujoco_plas.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/plas/seed10 --seed=10
# python mujoco_plas.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/plas/seed20 --seed=20
# python mujoco_plas.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/plas/seed30 --seed=30echo "--------Run PLAS in walker2d---------"
python mujoco_plas.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/plas/seed10 --seed=10
python mujoco_plas.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/plas/seed20 --seed=20
python mujoco_plas.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/plas/seed30 --seed=30

python mujoco_plas.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/plas/seed10 --seed=10
python mujoco_plas.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/plas/seed20 --seed=20
python mujoco_plas.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/plas/seed30 --seed=30

python mujoco_plas.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/plas/seed10 --seed=10
python mujoco_plas.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/plas/seed20 --seed=20
python mujoco_plas.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/plas/seed30 --seed=30

python mujoco_plas.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/plas/seed10 --seed=10
python mujoco_plas.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/plas/seed20 --seed=20
python mujoco_plas.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/plas/seed30 --seed=30

# python mujoco_plas.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/plas/seed10 --seed=10
# python mujoco_plas.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/plas/seed20 --seed=20
# python mujoco_plas.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/plas/seed30 --seed=30

# ========================= TD3-BC =========================
echo "---------Run TD3BC in hopper---------"
python mujoco_td3bc.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/td3bc/seed30 --seed=30

python mujoco_td3bc.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/td3bc/seed30 --seed=30

python mujoco_td3bc.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/td3bc/seed30 --seed=30

python mujoco_td3bc.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/td3bc/seed30 --seed=30

python mujoco_td3bc.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/td3bc/seed30 --seed=30

echo "--------Run TD3BC in halfcheetah---------"
python mujoco_td3bc.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/td3bc/seed30 --seed=30

python mujoco_td3bc.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/td3bc/seed30 --seed=30

python mujoco_td3bc.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/td3bc/seed30 --seed=30

python mujoco_td3bc.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/td3bc/seed30 --seed=30

python mujoco_td3bc.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/td3bc/seed30 --seed=30

echo "--------Run TD3BC in walker2d---------"
python mujoco_td3bc.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/td3bc/seed30 --seed=30

python mujoco_td3bc.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/td3bc/seed30 --seed=30

python mujoco_td3bc.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/td3bc/seed30 --seed=30

python mujoco_td3bc.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/td3bc/seed30 --seed=30

python mujoco_td3bc.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/td3bc/seed10 --seed=10
python mujoco_td3bc.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/td3bc/seed20 --seed=20
python mujoco_td3bc.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/td3bc/seed30 --seed=30

# ========================= IQL =========================
echo "---------IQL in hopper---------"
python mujoco_iql.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/iql/seed10 --seed=10
python mujoco_iql.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/iql/seed20 --seed=20
python mujoco_iql.py --env=hopper-random-v2 --learn_id=offline/mujoco/Hopper-v2/random/iql/seed30 --seed=30

python mujoco_iql.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/iql/seed10 --seed=10
python mujoco_iql.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/iql/seed20 --seed=20
python mujoco_iql.py --env=hopper-medium-v2 --learn_id=offline/mujoco/Hopper-v2/medium/iql/seed30 --seed=30

python mujoco_iql.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/iql/seed10 --seed=10
python mujoco_iql.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/iql/seed20 --seed=20
python mujoco_iql.py --env=hopper-expert-v2 --learn_id=offline/mujoco/Hopper-v2/expert/iql/seed30 --seed=30

python mujoco_iql.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/iql/seed10 --seed=10
python mujoco_iql.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/iql/seed20 --seed=20
python mujoco_iql.py --env=hopper-medium-expert-v2 --learn_id=offline/mujoco/Hopper-v2/medium-expert/iql/seed30 --seed=30

python mujoco_iql.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/iql/seed10 --seed=10
python mujoco_iql.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/iql/seed20 --seed=20
python mujoco_iql.py --env=hopper-medium-replay-v2 --learn_id=offline/mujoco/Hopper-v2/medium-replay/iql/seed30 --seed=30

echo "--------Run IQL in halfcheetah---------"
python mujoco_iql.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/iql/seed10 --seed=10
python mujoco_iql.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/iql/seed20 --seed=20
python mujoco_iql.py --env=halfcheetah-random-v2 --learn_id=offline/mujoco/HalfCheetah-v2/random/iql/seed30 --seed=30

python mujoco_iql.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/iql/seed10 --seed=10
python mujoco_iql.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/iql/seed20 --seed=20
python mujoco_iql.py --env=halfcheetah-medium-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium/iql/seed30 --seed=30

python mujoco_iql.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/iql/seed10 --seed=10
python mujoco_iql.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/iql/seed20 --seed=20
python mujoco_iql.py --env=halfcheetah-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/expert/iql/seed30 --seed=30

python mujoco_iql.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/iql/seed10 --seed=10
python mujoco_iql.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/iql/seed20 --seed=20
python mujoco_iql.py --env=halfcheetah-medium-expert-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-expert/iql/seed30 --seed=30

python mujoco_iql.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/iql/seed10 --seed=10
python mujoco_iql.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/iql/seed20 --seed=20
python mujoco_iql.py --env=halfcheetah-medium-replay-v2 --learn_id=offline/mujoco/HalfCheetah-v2/medium-replay/iql/seed30 --seed=30

echo "--------Run IQL in walker2d---------"
python mujoco_iql.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/iql/seed10 --seed=10
python mujoco_iql.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/iql/seed20 --seed=20
python mujoco_iql.py --env=walker2d-random-v2 --learn_id=offline/mujoco/Walker2d-v2/random/iql/seed30 --seed=30

python mujoco_iql.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/iql/seed10 --seed=10
python mujoco_iql.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/iql/seed20 --seed=20
python mujoco_iql.py --env=walker2d-medium-v2 --learn_id=offline/mujoco/Walker2d-v2/medium/iql/seed30 --seed=30

python mujoco_iql.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/iql/seed10 --seed=10
python mujoco_iql.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/iql/seed20 --seed=20
python mujoco_iql.py --env=walker2d-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/expert/iql/seed30 --seed=30

python mujoco_iql.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/iql/seed10 --seed=10
python mujoco_iql.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/iql/seed20 --seed=20
python mujoco_iql.py --env=walker2d-medium-expert-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-expert/iql/seed30 --seed=30

python mujoco_iql.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/iql/seed10 --seed=10
python mujoco_iql.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/iql/seed20 --seed=20
python mujoco_iql.py --env=walker2d-medium-replay-v2 --learn_id=offline/mujoco/Walker2d-v2/medium-replay/iql/seed30 --seed=30