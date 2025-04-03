#!/bin/bash

# classic
echo "--------- Run PPO in CartPole-v1 ---------"
python classic_ppo.py --env=CartPole-v1 --learn_id=modelfree/ppo/classic/CartPole-v1/seed10 --seed=10
python classic_ppo.py --env=CartPole-v1 --learn_id=modelfree/ppo/classic/CartPole-v1/seed20 --seed=20
python classic_ppo.py --env=CartPole-v1 --learn_id=modelfree/ppo/classic/CartPole-v1/seed30 --seed=30
echo "--------- Run PPO in Pendulum-v1 ---------"  # The result is bad.
python classic_ppo.py --env=Pendulum-v1 --learn_id=modelfree/ppo/classic/Pendulum-v1/seed10 --traj_length=512 --max_train_step=100000 --seed=10
python classic_ppo.py --env=Pendulum-v1 --learn_id=modelfree/ppo/classic/Pendulum-v1/seed20 --traj_length=512 --max_train_step=100000 --seed=20
python classic_ppo.py --env=Pendulum-v1 --learn_id=modelfree/ppo/classic/Pendulum-v1/seed30 --traj_length=512 --max_train_step=100000 --seed=30

# atari
echo "--------- Run PPO in PongNoFrameskip-v4 ---------"  # the result is bad.
python atari_ppo.py --env=PongNoFrameskip-v4 --learn_id=modelfree/ppo/atari/Pong-v4/seed10 --train_num=8 --eval_num=8 --max_train_step=1000000 --seed=10
python atari_ppo.py --env=PongNoFrameskip-v4 --learn_id=modelfree/ppo/atari/Pong-v4/seed20 --train_num=8 --eval_num=8 --max_train_step=1000000 --seed=20
python atari_ppo.py --env=PongNoFrameskip-v4 --learn_id=modelfree/ppo/atari/Pong-v4/seed30 --train_num=8 --eval_num=8 --max_train_step=1000000 --seed=30
echo "--------- Run PPO in BreakoutNoFrameskip-v4 ---------" 
python atari_ppo.py --env=BreakoutNoFrameskip-v4 --learn_id=modelfree/ppo/atari/Breakout-v4/seed10 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=10
python atari_ppo.py --env=BreakoutNoFrameskip-v4 --learn_id=modelfree/ppo/atari/Breakout-v4/seed20 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=20
python atari_ppo.py --env=BreakoutNoFrameskip-v4 --learn_id=modelfree/ppo/atari/Breakout-v4/seed30 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=30
echo "--------- Run PPO in EnduroNoFrameskip-v4 ---------" 
python atari_ppo.py --env=EnduroNoFrameskip-v4 --learn_id=modelfree/ppo/atari/Enduro-v4/seed10 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=10
python atari_ppo.py --env=EnduroNoFrameskip-v4 --learn_id=modelfree/ppo/atari/Enduro-v4/seed20 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=20
python atari_ppo.py --env=EnduroNoFrameskip-v4 --learn_id=modelfree/ppo/atari/Enduro-v4/seed30 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=30

# box2d
echo "--------- Run PPO in LunarLander-v2 ---------"
python box2d_ppo.py --env=LunarLander-v2 --learn_id=modelfree/ppo/box2d/LunarLander-v2/seed10 --seed=10
python box2d_ppo.py --env=LunarLander-v2 --learn_id=modelfree/ppo/box2d/LunarLander-v2/seed20 --seed=20
python box2d_ppo.py --env=LunarLander-v2 --learn_id=modelfree/ppo/box2d/LunarLander-v2/seed30 --seed=30
echo "--------- Run PPO in BipedalWalker-v3 ---------"
python box2d_ppo.py --env=BipedalWalker-v3 --learn_id=modelfree/ppo/box2d/BipedalWalker-v3/seed10 --traj_length=1024 --max_train_step=1000000 --seed=10
python box2d_ppo.py --env=BipedalWalker-v3 --learn_id=modelfree/ppo/box2d/BipedalWalker-v3/seed20 --traj_length=1024 --max_train_step=1000000 --seed=20
python box2d_ppo.py --env=BipedalWalker-v3 --learn_id=modelfree/ppo/box2d/BipedalWalker-v3/seed30 --traj_length=1024 --max_train_step=1000000 --seed=30

# mujoco
echo "---------Run PPO in Hooper-v4---------"
python mujoco_ppo.py --env=Hopper-v4 --learn_id=modelfree/ppo/mujoco/Hopper-v4/seed10 --seed=10
python mujoco_ppo.py --env=Hopper-v4 --learn_id=modelfree/ppo/mujoco/Hopper-v4/seed20 --seed=20
python mujoco_ppo.py --env=Hopper-v4 --learn_id=modelfree/ppo/mujoco/Hopper-v4/seed30 --seed=30
echo "---------Run PPO in HalfCheetah-v4---------"
python mujoco_ppo.py --env=HalfCheetah-v4 --learn_id=modelfree/ppo/mujoco/HalfCheetah-v4/seed10 --seed=10
python mujoco_ppo.py --env=HalfCheetah-v4 --learn_id=modelfree/ppo/mujoco/HalfCheetah-v4/seed20 --seed=20
python mujoco_ppo.py --env=HalfCheetah-v4 --learn_id=modelfree/ppo/mujoco/HalfCheetah-v4/seed30 --seed=30
echo "---------Run PPO in Walker2d-v4---------"
python mujoco_ppo.py --env=Walker2d-v4 --learn_id=modelfree/ppo/mujoco/Walker2d-v4/seed10 --seed=10
python mujoco_ppo.py --env=Walker2d-v4 --learn_id=modelfree/ppo/mujoco/Walker2d-v4/seed20 --seed=20
python mujoco_ppo.py --env=Walker2d-v4 --learn_id=modelfree/ppo/mujoco/Walker2d-v4/seed30 --seed=30
echo "---------Run PPO in Ant-v4---------"
python mujoco_ppo.py --env=Ant-v4 --learn_id=modelfree/ppo/mujoco/Ant-v4/seed10 --seed=10
python mujoco_ppo.py --env=Ant-v4 --learn_id=modelfree/ppo/mujoco/Ant-v4/seed20 --seed=20
python mujoco_ppo.py --env=Ant-v4 --learn_id=modelfree/ppo/mujoco/Ant-v4/seed30 --seed=30
echo "---------Run PPO in Humanoid-v4---------"
python mujoco_ppo.py --env=Humanoid-v4 --learn_id=modelfree/ppo/mujoco/Humanoid-v4/seed10 --seed=10
python mujoco_ppo.py --env=Humanoid-v4 --learn_id=modelfree/ppo/mujoco/Humanoid-v4/seed20 --seed=20
python mujoco_ppo.py --env=Humanoid-v4 --learn_id=modelfree/ppo/mujoco/Humanoid-v4/seed30 --seed=30
echo "---------Run PPO in Swimmer-v4---------"
python mujoco_ppo.py --env=Swimmer-v4 --learn_id=modelfree/ppo/mujoco/Swimmer-v4/seed10 --seed=10
python mujoco_ppo.py --env=Swimmer-v4 --learn_id=modelfree/ppo/mujoco/Swimmer-v4/seed20 --seed=20
python mujoco_ppo.py --env=Swimmer-v4 --learn_id=modelfree/ppo/mujoco/Swimmer-v4/seed30 --seed=30
echo "---------Run PPO in Reacher-v4---------"
python mujoco_ppo.py --env=Reacher-v4 --learn_id=modelfree/ppo/mujoco/Reacher-v4/seed10 --seed=10
python mujoco_ppo.py --env=Reacher-v4 --learn_id=modelfree/ppo/mujoco/Reacher-v4/seed20 --seed=20
python mujoco_ppo.py --env=Reacher-v4 --learn_id=modelfree/ppo/mujoco/Reacher-v4/seed30 --seed=30
echo "---------Run PPO in Pusher-v4---------"
python mujoco_ppo.py --env=Pusher-v4 --learn_id=modelfree/ppo/mujoco/Pusher-v4/seed10 --seed=10
python mujoco_ppo.py --env=Pusher-v4 --learn_id=modelfree/ppo/mujoco/Pusher-v4/seed20 --seed=20
python mujoco_ppo.py --env=Pusher-v4 --learn_id=modelfree/ppo/mujoco/Pusher-v4/seed30 --seed=30