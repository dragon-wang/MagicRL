#!/bin/bash

# classic
echo "---------Run TD3 in Pendulum-v1---------"
python classic_td3.py --env=Pendulum-v1 --learn_id=modelfree/td3/classic/Pendulum-v1/seed10 --seed=10
python classic_td3.py --env=Pendulum-v1 --learn_id=modelfree/td3/classic/Pendulum-v1/seed20 --seed=20
python classic_td3.py --env=Pendulum-v1 --learn_id=modelfree/td3/classic/Pendulum-v1/seed30 --seed=30

# box2d
echo "---------Run TD3 in BipedalWalker-v3---------"
python box2d_td3.py --env=BipedalWalker-v3 --learn_id=modelfree/td3/box2d/BipedalWalker-v3/seed10 --seed=10
python box2d_td3.py --env=BipedalWalker-v3 --learn_id=modelfree/td3/box2d/BipedalWalker-v3/seed20 --seed=20
python box2d_td3.py --env=BipedalWalker-v3 --learn_id=modelfree/td3/box2d/BipedalWalker-v3/seed30 --seed=30

# mujoco
echo "---------Run TD3 in Hooper-v4---------"
python mujoco_td3.py --env=Hopper-v4 --learn_id=modelfree/td3/mujoco/Hopper-v4/seed10 --seed=10
python mujoco_td3.py --env=Hopper-v4 --learn_id=modelfree/td3/mujoco/Hopper-v4/seed20 --seed=20
python mujoco_td3.py --env=Hopper-v4 --learn_id=modelfree/td3/mujoco/Hopper-v4/seed30 --seed=30
echo "---------Run TD3 in HalfCheetah-v4---------"
python mujoco_td3.py --env=HalfCheetah-v4 --learn_id=modelfree/td3/mujoco/HalfCheetah-v4/seed10 --seed=10
python mujoco_td3.py --env=HalfCheetah-v4 --learn_id=modelfree/td3/mujoco/HalfCheetah-v4/seed20 --seed=20
python mujoco_td3.py --env=HalfCheetah-v4 --learn_id=modelfree/td3/mujoco/HalfCheetah-v4/seed30 --seed=30
echo "---------Run TD3 in Walker2d-v4---------"
python mujoco_td3.py --env=Walker2d-v4 --learn_id=modelfree/td3/mujoco/Walker2d-v4/seed10 --seed=10
python mujoco_td3.py --env=Walker2d-v4 --learn_id=modelfree/td3/mujoco/Walker2d-v4/seed20 --seed=20
python mujoco_td3.py --env=Walker2d-v4 --learn_id=modelfree/td3/mujoco/Walker2d-v4/seed30 --seed=30
echo "---------Run TD3 in Ant-v4---------"
python mujoco_td3.py --env=Ant-v4 --learn_id=modelfree/td3/mujoco/Ant-v4/seed10 --seed=10
python mujoco_td3.py --env=Ant-v4 --learn_id=modelfree/td3/mujoco/Ant-v4/seed20 --seed=20
python mujoco_td3.py --env=Ant-v4 --learn_id=modelfree/td3/mujoco/Ant-v4/seed30 --seed=30
echo "---------Run TD3 in Humanoid-v4---------"
python mujoco_td3.py --env=Humanoid-v4 --learn_id=modelfree/td3/mujoco/Humanoid-v4/seed10 --seed=10
python mujoco_td3.py --env=Humanoid-v4 --learn_id=modelfree/td3/mujoco/Humanoid-v4/seed20 --seed=20
python mujoco_td3.py --env=Humanoid-v4 --learn_id=modelfree/td3/mujoco/Humanoid-v4/seed30 --seed=30
echo "---------Run TD3 in Swimmer-v4---------"
python mujoco_td3.py --env=Swimmer-v4 --learn_id=modelfree/td3/mujoco/Swimmer-v4/seed10 --seed=10
python mujoco_td3.py --env=Swimmer-v4 --learn_id=modelfree/td3/mujoco/Swimmer-v4/seed20 --seed=20
python mujoco_td3.py --env=Swimmer-v4 --learn_id=modelfree/td3/mujoco/Swimmer-v4/seed30 --seed=30
echo "---------Run TD3 in Reacher-v4---------"
python mujoco_td3.py --env=Reacher-v4 --learn_id=modelfree/td3/mujoco/Reacher-v4/seed10 --seed=10
python mujoco_td3.py --env=Reacher-v4 --learn_id=modelfree/td3/mujoco/Reacher-v4/seed20 --seed=20
python mujoco_td3.py --env=Reacher-v4 --learn_id=modelfree/td3/mujoco/Reacher-v4/seed30 --seed=30
echo "---------Run TD3 in Pusher-v4---------"
python mujoco_td3.py --env=Pusher-v4 --learn_id=modelfree/td3/mujoco/Pusher-v4/seed10 --seed=10
python mujoco_td3.py --env=Pusher-v4 --learn_id=modelfree/td3/mujoco/Pusher-v4/seed20 --seed=20
python mujoco_td3.py --env=Pusher-v4 --learn_id=modelfree/td3/mujoco/Pusher-v4/seed30 --seed=30

# d4rl
echo "---------Run TD3 in hopper-medium-v2---------"
python d4rl_mujoco_td3.py --env=hopper-medium-v2 --learn_id=modelfree/td3/d4rl/hop-m-v2/seed10 --seed=10
python d4rl_mujoco_td3.py --env=hopper-medium-v2 --learn_id=modelfree/td3/d4rl/hop-m-v2/seed20 --seed=20
python d4rl_mujoco_td3.py --env=hopper-medium-v2 --learn_id=modelfree/td3/d4rl/hop-m-v2/seed30 --seed=30
echo "--------Run TD3 in halfcheetah-medium-v2---------"
python d4rl_mujoco_td3.py --env=halfcheetah-medium-v2 --learn_id=modelfree/td3/d4rl/half-m-v2/seed10 --seed=10
python d4rl_mujoco_td3.py --env=halfcheetah-medium-v2 --learn_id=modelfree/td3/d4rl/half-m-v2/seed20 --seed=20
python d4rl_mujoco_td3.py --env=halfcheetah-medium-v2 --learn_id=modelfree/td3/d4rl/half-m-v2/seed30 --seed=30
echo "--------Run TD3 in walker2d-medium-v2---------"
python d4rl_mujoco_td3.py --env=walker2d-medium-v2 --learn_id=modelfree/td3/d4rl/walk-m-v2/seed10 --seed=10
python d4rl_mujoco_td3.py --env=walker2d-medium-v2 --learn_id=modelfree/td3/d4rl/walk-m-v2/seed20 --seed=20
python d4rl_mujoco_td3.py --env=walker2d-medium-v2 --learn_id=modelfree/td3/d4rl/walk-m-v2/seed30 --seed=30