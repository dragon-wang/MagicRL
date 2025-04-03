#!/bin/bash

# classic
echo "---------Run DDPG in Pendulum-v1---------"
python classic_ddpg.py --env=Pendulum-v1 --learn_id=modelfree/ddpg/classic/Pendulum-v1/seed10 --seed=10
python classic_ddpg.py --env=Pendulum-v1 --learn_id=modelfree/ddpg/classic/Pendulum-v1/seed20 --seed=20
python classic_ddpg.py --env=Pendulum-v1 --learn_id=modelfree/ddpg/classic/Pendulum-v1/seed30 --seed=30

# box2d
echo "---------Run DDPG in BipedalWalker-v3---------"
python box2d_ddpg.py -env=BipedalWalker-v3 --learn_id=modelfree/ddpg/box2d/BipedalWalker-v3/seed10 --seed=10
python box2d_ddpg.py -env=BipedalWalker-v3 --learn_id=modelfree/ddpg/box2d/BipedalWalker-v3/seed20 --seed=20
python box2d_ddpg.py -env=BipedalWalker-v3 --learn_id=modelfree/ddpg/box2d/BipedalWalker-v3/seed30 --seed=30

# mujoco
echo "---------Run DDPG in Hooper-v4---------"
python mujoco_ddpg.py --env=Hopper-v4 --learn_id=modelfree/ddpg/mujoco/Hopper-v4/seed10 --seed=10
python mujoco_ddpg.py --env=Hopper-v4 --learn_id=modelfree/ddpg/mujoco/Hopper-v4/seed20 --seed=20
python mujoco_ddpg.py --env=Hopper-v4 --learn_id=modelfree/ddpg/mujoco/Hopper-v4/seed30 --seed=30
echo "---------Run DDPG in HalfCheetah-v4---------"
python mujoco_ddpg.py --env=HalfCheetah-v4 --learn_id=modelfree/ddpg/mujoco/HalfCheetah-v4/seed10 --seed=10
python mujoco_ddpg.py --env=HalfCheetah-v4 --learn_id=modelfree/ddpg/mujoco/HalfCheetah-v4/seed20 --seed=20
python mujoco_ddpg.py --env=HalfCheetah-v4 --learn_id=modelfree/ddpg/mujoco/HalfCheetah-v4/seed30 --seed=30
echo "---------Run DDPG in Walker2d-v4---------"
python mujoco_ddpg.py --env=Walker2d-v4 --learn_id=modelfree/ddpg/mujoco/Walker2d-v4/seed10 --seed=10
python mujoco_ddpg.py --env=Walker2d-v4 --learn_id=modelfree/ddpg/mujoco/Walker2d-v4/seed20 --seed=20
python mujoco_ddpg.py --env=Walker2d-v4 --learn_id=modelfree/ddpg/mujoco/Walker2d-v4/seed30 --seed=30
echo "---------Run DDPG in Ant-v4---------"
python mujoco_ddpg.py --env=Ant-v4 --learn_id=modelfree/ddpg/mujoco/Ant-v4/seed10 --seed=10
python mujoco_ddpg.py --env=Ant-v4 --learn_id=modelfree/ddpg/mujoco/Ant-v4/seed20 --seed=20
python mujoco_ddpg.py --env=Ant-v4 --learn_id=modelfree/ddpg/mujoco/Ant-v4/seed30 --seed=30
echo "---------Run DDPG in Humanoid-v4---------"
python mujoco_ddpg.py --env=Humanoid-v4 --learn_id=modelfree/ddpg/mujoco/Humanoid-v4/seed10 --seed=10
python mujoco_ddpg.py --env=Humanoid-v4 --learn_id=modelfree/ddpg/mujoco/Humanoid-v4/seed20 --seed=20
python mujoco_ddpg.py --env=Humanoid-v4 --learn_id=modelfree/ddpg/mujoco/Humanoid-v4/seed30 --seed=30
echo "---------Run DDPG in Swimmer-v4---------"
python mujoco_ddpg.py --env=Swimmer-v4 --learn_id=modelfree/ddpg/mujoco/Swimmer-v4/seed10 --seed=10
python mujoco_ddpg.py --env=Swimmer-v4 --learn_id=modelfree/ddpg/mujoco/Swimmer-v4/seed20 --seed=20
python mujoco_ddpg.py --env=Swimmer-v4 --learn_id=modelfree/ddpg/mujoco/Swimmer-v4/seed30 --seed=30
echo "---------Run DDPG in Reacher-v4---------"
python mujoco_ddpg.py --env=Reacher-v4 --learn_id=modelfree/ddpg/mujoco/Reacher-v4/seed10 --seed=10
python mujoco_ddpg.py --env=Reacher-v4 --learn_id=modelfree/ddpg/mujoco/Reacher-v4/seed20 --seed=20
python mujoco_ddpg.py --env=Reacher-v4 --learn_id=modelfree/ddpg/mujoco/Reacher-v4/seed30 --seed=30
echo "---------Run DDPG in Pusher-v4---------"
python mujoco_ddpg.py --env=Pusher-v4 --learn_id=modelfree/ddpg/mujoco/Pusher-v4/seed10 --seed=10
python mujoco_ddpg.py --env=Pusher-v4 --learn_id=modelfree/ddpg/mujoco/Pusher-v4/seed20 --seed=20
python mujoco_ddpg.py --env=Pusher-v4 --learn_id=modelfree/ddpg/mujoco/Pusher-v4/seed30 --seed=30