#!/bin/bash

# ========================= BCQ =========================
echo "---------Run BCQ in hopper---------"
python mujoco_bcq.py --env=hopper-random-v2 --learn_id=bcq/Hopper-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_bcq.py --env=hopper-medium-v2 --learn_id=bcq/Hopper-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_bcq.py --env=hopper-expert-v2 --learn_id=bcq/Hopper-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_bcq.py --env=hopper-medium-expert-v2 --learn_id=bcq/Hopper-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_bcq.py --env=hopper-medium-replay-v2 --learn_id=bcq/Hopper-v2/medium/seed10 --seed=10 --device=cuda
echo "---------Run BCQ in halfcheetah---------"
python mujoco_bcq.py --env=halfcheetah-random-v2 --learn_id=bcq/HalfCheetah-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_bcq.py --env=halfcheetah-medium-v2 --learn_id=bcq/HalfCheetah-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_bcq.py --env=halfcheetah-expert-v2 --learn_id=bcq/HalfCheetah-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_bcq.py --env=halfcheetah-medium-expert-v2 --learn_id=bcq/HalfCheetah-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_bcq.py --env=halfcheetah-medium-replay-v2 --learn_id=bcq/HalfCheetah-v2/medium/seed10 --seed=10 --device=cuda
echo "---------Run BCQ in walker2d---------"
python mujoco_bcq.py --env=walker2d-random-v2 --learn_id=bcq/Walker2d-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_bcq.py --env=walker2d-medium-v2 --learn_id=bcq/Walker2d-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_bcq.py --env=walker2d-expert-v2 --learn_id=bcq/Walker2d-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_bcq.py --env=walker2d-medium-expert-v2 --learn_id=bcq/Walker2d-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_bcq.py --env=walker2d-medium-replay-v2 --learn_id=bcq/Walker2d-v2/medium/seed10 --seed=10 --device=cuda

# ========================= BEAR =========================
echo "---------Run BEAR in hopper---------"
python mujoco_bear.py --env=hopper-random-v2 --learn_id=bear/Hopper-v2/medium/seed10 --kernel_type=laplacian --seed=10 --device=cuda
python mujoco_bear.py --env=hopper-medium-v2 --learn_id=bear/Hopper-v2/medium/seed10 --kernel_type=laplacian --seed=10 --device=cuda
python mujoco_bear.py --env=hopper-expert-v2 --learn_id=bear/Hopper-v2/medium/seed10 --kernel_type=laplacian --seed=10 --device=cuda
python mujoco_bear.py --env=hopper-medium-expert-v2 --learn_id=bear/Hopper-v2/medium/seed10 --kernel_type=laplacian --seed=10 --device=cuda
python mujoco_bear.py --env=hopper-medium-replay-v2 --learn_id=bear/Hopper-v2/medium/seed10 --kernel_type=laplacian --seed=10 --device=cuda
echo "--------bearn BEAR in halfcheetah---------"
python mujoco_bear.py --env=halfcheetah-random-v2 --learn_id=bear/HalfCheetah-v2/medium/seed10 --kernel_type=gaussian --seed=10 --device=cuda
python mujoco_bear.py --env=halfcheetah-medium-v2 --learn_id=bear/HalfCheetah-v2/medium/seed10 --kernel_type=gaussian --seed=10 --device=cuda
python mujoco_bear.py --env=halfcheetah-expert-v2 --learn_id=bear/HalfCheetah-v2/medium/seed10 --kernel_type=gaussian --seed=10 --device=cuda
python mujoco_bear.py --env=halfcheetah-medium-expert-v2 --learn_id=bear/HalfCheetah-v2/medium/seed10 --kernel_type=gaussian --seed=10 --device=cuda
python mujoco_bear.py --env=halfcheetah-medium-replay-v2 --learn_id=bear/HalfCheetah-v2/medium/seed10 --kernel_type=gaussian --seed=10 --device=cuda
echo "--------bearn BEAR in walker2d---------"
python mujoco_bear.py --env=walker2d-random-v2 --learn_id=bear/Walker2d-v2/medium/seed10 --kernel_type=laplacian --seed=10 --device=cuda
python mujoco_bear.py --env=walker2d-medium-v2 --learn_id=bear/Walker2d-v2/medium/seed10 --kernel_type=laplacian --seed=10 --device=cuda
python mujoco_bear.py --env=walker2d-expert-v2 --learn_id=bear/Walker2d-v2/medium/seed10 --kernel_type=laplacian --seed=10 --device=cuda
python mujoco_bear.py --env=walker2d-medium-expert-v2 --learn_id=bear/Walker2d-v2/medium/seed10 --kernel_type=laplacian --seed=10 --device=cuda
python mujoco_bear.py --env=walker2d-medium-replay-v2 --learn_id=bear/Walker2d-v2/medium/seed10 --kernel_type=laplacian --seed=10 --device=cuda

# ========================= CQL =========================
echo "---------Run CQL in hopper---------"
python mujoco_cql.py --env=hopper-random-v2 --learn_id=cql/Hopper-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
python mujoco_cql.py --env=hopper-medium-v2 --learn_id=cql/Hopper-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
python mujoco_cql.py --env=hopper-expert-v2 --learn_id=cql/Hopper-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
python mujoco_cql.py --env=hopper-medium-expert-v2 --learn_id=cql/Hopper-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
python mujoco_cql.py --env=hopper-medium-replay-v2 --learn_id=cql/Hopper-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
echo "--------cqln CQL in halfcheetah---------"
python mujoco_cql.py --env=halfcheetah-random-v2 --learn_id=cql/HalfCheetah-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
python mujoco_cql.py --env=halfcheetah-medium-v2 --learn_id=cql/HalfCheetah-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
python mujoco_cql.py --env=halfcheetah-expert-v2 --learn_id=cql/HalfCheetah-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
python mujoco_cql.py --env=halfcheetah-medium-expert-v2 --learn_id=cql/HalfCheetah-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
python mujoco_cql.py --env=halfcheetah-medium-replay-v2 --learn_id=cql/HalfCheetah-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
echo "--------cqln CQL in walker2d---------"
python mujoco_cql.py --env=walker2d-random-v2 --learn_id=cql/Walker2d-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
python mujoco_cql.py --env=walker2d-medium-v2 --learn_id=cql/Walker2d-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
python mujoco_cql.py --env=walker2d-expert-v2 --learn_id=cql/Walker2d-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
python mujoco_cql.py --env=walker2d-medium-expert-v2 --learn_id=cql/Walker2d-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda
python mujoco_cql.py --env=walker2d-medium-replay-v2 --learn_id=cql/Walker2d-v2/medium/seed10 --auto_alpha --with_lagrange --lagrange_thresh=-1.0 --seed=10 --device=cuda

# ========================= PLAS =========================
echo "---------Run PLAS in hopper---------"
python mujoco_plas.py --env=hopper-random-v2 --learn_id=plas/Hopper-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_plas.py --env=hopper-medium-v2 --learn_id=plas/Hopper-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_plas.py --env=hopper-expert-v2 --learn_id=plas/Hopper-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_plas.py --env=hopper-medium-expert-v2 --learn_id=plas/Hopper-v2/medium/seed10 --seed=10 --device=cuda
# The network structure should be changed in small dataset.
# python mujoco_plas.py --env=hopper-medium-replay-v2 --learn_id=plas/Hopper-v2/medium/seed10 --seed=10 --device=cuda
echo "--------plasn PLAS in halfcheetah---------"
python mujoco_plas.py --env=halfcheetah-random-v2 --learn_id=plas/HalfCheetah-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_plas.py --env=halfcheetah-medium-v2 --learn_id=plas/HalfCheetah-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_plas.py --env=halfcheetah-expert-v2 --learn_id=plas/HalfCheetah-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_plas.py --env=halfcheetah-medium-expert-v2 --learn_id=plas/HalfCheetah-v2/medium/seed10 --seed=10 --device=cuda
# python mujoco_plas.py --env=halfcheetah-medium-replay-v2 --learn_id=plas/HalfCheetah-v2/medium/seed10 --seed=10 --device=cuda
echo "--------plasn PLAS in walker2d---------"
python mujoco_plas.py --env=walker2d-random-v2 --learn_id=plas/Walker2d-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_plas.py --env=walker2d-medium-v2 --learn_id=plas/Walker2d-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_plas.py --env=walker2d-expert-v2 --learn_id=plas/Walker2d-v2/medium/seed10 --seed=10 --device=cuda
python mujoco_plas.py --env=walker2d-medium-expert-v2 --learn_id=plas/Walker2d-v2/medium/seed10 --seed=10 --device=cuda
# python mujoco_plas.py --env=walker2d-medium-replay-v2 --learn_id=plas/Walker2d-v2/medium/seed10 --seed=10 --device=cuda