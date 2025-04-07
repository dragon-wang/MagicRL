#!/bin/bash

# classic
echo "---------Run SAC in Pendulum-v1---------"
python classic_sac.py --env=Pendulum-v1 --learn_id=modelfree/sac/classic/Pendulum-v1/seed10 --seed=10
python classic_sac.py --env=Pendulum-v1 --learn_id=modelfree/sac/classic/Pendulum-v1/seed20 --seed=20
python classic_sac.py --env=Pendulum-v1 --learn_id=modelfree/sac/classic/Pendulum-v1/seed30 --seed=30

# box2d
echo "---------Run SAC in BipedalWalker-v3---------"
python box2d_sac.py --env=BipedalWalker-v3 --learn_id=modelfree/sac/box2d/BipedalWalker-v3/seed10 --seed=10
python box2d_sac.py --env=BipedalWalker-v3 --learn_id=modelfree/sac/box2d/BipedalWalker-v3/seed20 --seed=20
python box2d_sac.py --env=BipedalWalker-v3 --learn_id=modelfree/sac/box2d/BipedalWalker-v3/seed30 --seed=30

# mujoco
echo "---------Run SAC in Hooper-v4---------"
python mujoco_sac.py --env=Hopper-v4 --learn_id=modelfree/sac/mujoco/Hopper-v4/seed10 --seed=10
python mujoco_sac.py --env=Hopper-v4 --learn_id=modelfree/sac/mujoco/Hopper-v4/seed20 --seed=20
python mujoco_sac.py --env=Hopper-v4 --learn_id=modelfree/sac/mujoco/Hopper-v4/seed30 --seed=30
echo "---------Run SAC in HalfCheetah-v4---------"
python mujoco_sac.py --env=HalfCheetah-v4 --learn_id=modelfree/sac/mujoco/HalfCheetah-v4/seed10 --seed=10
python mujoco_sac.py --env=HalfCheetah-v4 --learn_id=modelfree/sac/mujoco/HalfCheetah-v4/seed20 --seed=20
python mujoco_sac.py --env=HalfCheetah-v4 --learn_id=modelfree/sac/mujoco/HalfCheetah-v4/seed30 --seed=30
echo "---------Run SAC in Walker2d-v4---------"
python mujoco_sac.py --env=Walker2d-v4 --learn_id=modelfree/sac/mujoco/Walker2d-v4/seed10 --seed=10
python mujoco_sac.py --env=Walker2d-v4 --learn_id=modelfree/sac/mujoco/Walker2d-v4/seed20 --seed=20
python mujoco_sac.py --env=Walker2d-v4 --learn_id=modelfree/sac/mujoco/Walker2d-v4/seed30 --seed=30
echo "---------Run SAC in Ant-v4---------"
python mujoco_sac.py --env=Ant-v4 --learn_id=modelfree/sac/mujoco/Ant-v4/seed10 --seed=10
python mujoco_sac.py --env=Ant-v4 --learn_id=modelfree/sac/mujoco/Ant-v4/seed20 --seed=20
python mujoco_sac.py --env=Ant-v4 --learn_id=modelfree/sac/mujoco/Ant-v4/seed30 --seed=30
echo "---------Run SAC in Humanoid-v4---------"
python mujoco_sac.py --env=Humanoid-v4 --learn_id=modelfree/sac/mujoco/Humanoid-v4/seed10 --seed=10
python mujoco_sac.py --env=Humanoid-v4 --learn_id=modelfree/sac/mujoco/Humanoid-v4/seed20 --seed=20
python mujoco_sac.py --env=Humanoid-v4 --learn_id=modelfree/sac/mujoco/Humanoid-v4/seed30 --seed=30
echo "---------Run SAC in Swimmer-v4---------"
python mujoco_sac.py --env=Swimmer-v4 --learn_id=modelfree/sac/mujoco/Swimmer-v4/seed10 --seed=10
python mujoco_sac.py --env=Swimmer-v4 --learn_id=modelfree/sac/mujoco/Swimmer-v4/seed20 --seed=20
python mujoco_sac.py --env=Swimmer-v4 --learn_id=modelfree/sac/mujoco/Swimmer-v4/seed30 --seed=30
echo "---------Run SAC in Reacher-v4---------"
python mujoco_sac.py --env=Reacher-v4 --learn_id=modelfree/sac/mujoco/Reacher-v4/seed10 --seed=10
python mujoco_sac.py --env=Reacher-v4 --learn_id=modelfree/sac/mujoco/Reacher-v4/seed20 --seed=20
python mujoco_sac.py --env=Reacher-v4 --learn_id=modelfree/sac/mujoco/Reacher-v4/seed30 --seed=30
echo "---------Run SAC in Pusher-v4---------"
python mujoco_sac.py --env=Pusher-v4 --learn_id=modelfree/sac/mujoco/Pusher-v4/seed10 --seed=10
python mujoco_sac.py --env=Pusher-v4 --learn_id=modelfree/sac/mujoco/Pusher-v4/seed20 --seed=20
python mujoco_sac.py --env=Pusher-v4 --learn_id=modelfree/sac/mujoco/Pusher-v4/seed30 --seed=30


# d4rl
echo "---------Run SAC in hopper-medium-v2---------"
python d4rl_mujoco_sac.py --env=hopper-medium-v2 --learn_id=modelfree/sac/d4rl/hop-m-v2/seed10 --seed=10
python d4rl_mujoco_sac.py --env=hopper-medium-v2 --learn_id=modelfree/sac/d4rl/hop-m-v2/seed20 --seed=20
python d4rl_mujoco_sac.py --env=hopper-medium-v2 --learn_id=modelfree/sac/d4rl/hop-m-v2/seed30 --seed=30
echo "--------Run TD3 in halfcheetah-medium-v2---------"
python d4rl_mujoco_sac.py --env=halfcheetah-medium-v2 --learn_id=modelfree/sac/d4rl/half-m-v2/seed10 --seed=10
python d4rl_mujoco_sac.py --env=halfcheetah-medium-v2 --learn_id=modelfree/sac/d4rl/half-m-v2/seed20 --seed=20
python d4rl_mujoco_sac.py --env=halfcheetah-medium-v2 --learn_id=modelfree/sac/d4rl/half-m-v2/seed30 --seed=30
echo "--------Run TD3 in walker2d-medium-v2---------"
python d4rl_mujoco_sac.py --env=walker2d-medium-v2 --learn_id=modelfree/sac/d4rl/walk-m-v2/seed10 --seed=10
python d4rl_mujoco_sac.py --env=walker2d-medium-v2 --learn_id=modelfree/sac/d4rl/walk-m-v2/seed20 --seed=20
python d4rl_mujoco_sac.py --env=walker2d-medium-v2 --learn_id=modelfree/sac/d4rl/walk-m-v2/seed30 --seed=30