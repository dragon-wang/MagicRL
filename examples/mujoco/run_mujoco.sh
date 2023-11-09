#!/bin/bash

# ========================= DDPG =========================
echo "---------Run DDPG in Hooper-v4---------"
python mujoco_ddpg.py --env=Hopper-v4 --learn_id=mujoco/ddpg/Hopper-v4
echo "---------Run DDPG in HalfCheetah-v4---------"
python mujoco_ddpg.py --env=HalfCheetah-v4 --learn_id=mujoco/ddpg/HalfCheetah-v4
echo "---------Run DDPG in Walker2d-v4---------"
python mujoco_ddpg.py --env=Walker2d-v4 --learn_id=mujoco/ddpg/Walker2d-v4
echo "---------Run DDPG in Ant-v4---------"
python mujoco_ddpg.py --env=Ant-v4 --learn_id=mujoco/ddpg/Ant-v4
echo "---------Run DDPG in Humanoid-v4---------"
python mujoco_ddpg.py --env=Humanoid-v4 --learn_id=mujoco/ddpg/Humanoid-v4
echo "---------Run DDPG in Swimmer-v4---------"
python mujoco_ddpg.py --env=Swimmer-v4 --learn_id=mujoco/ddpg/Swimmer-v4
echo "---------Run DDPG in Reacher-v4---------"
python mujoco_ddpg.py --env=Reacher-v4 --learn_id=mujoco/ddpg/Reacher-v4
echo "---------Run DDPG in Pusher-v4---------"
python mujoco_ddpg.py --env=Pusher-v4 --learn_id=mujoco/ddpg/Pusher-v4

# PPO
echo "--------- Run PPO in Hopper-v4 ---------"
python mujoco_ppo.py --env=Hopper-v4 --learn_id=mujoco/ppo/Hopper-v4
echo "--------- Run PPO in HalfCheetah-v4 ---------"
python mujoco_ppo.py --env=HalfCheetah-v4 --learn_id=mujoco/ppo/HalfCheetah-v4
echo "--------- Run PPO in Walker2d-v4 ---------"
python mujoco_ppo.py --env=Walker2d-v4 --learn_id=mujoco/ppo/Walker2d-v4

# ========================= SAC =========================
echo "---------Run SAC in Hooper-v4---------"
python mujoco_sac.py --env=Hopper-v4 --learn_id=mujoco/sac/Hopper-v4
echo "---------Run SAC in HalfCheetah-v4---------"
python mujoco_sac.py --env=HalfCheetah-v4 --learn_id=mujoco/sac/HalfCheetah-v4
echo "---------Run SAC in Walker2d-v4---------"
python mujoco_sac.py --env=Walker2d-v4 --learn_id=mujoco/sac/Walker2d-v4
echo "---------Run SAC in Ant-v4---------"
python mujoco_sac.py --env=Ant-v4 --learn_id=mujoco/sac/Ant-v4
echo "---------Run SAC in Humanoid-v4---------"
python mujoco_sac.py --env=Humanoid-v4 --learn_id=mujoco/sac/Humanoid-v4
echo "---------Run SAC in Swimmer-v4---------"
python mujoco_sac.py --env=Swimmer-v4 --learn_id=mujoco/sac/Swimmer-v4
echo "---------Run SAC in Reacher-v4---------"
python mujoco_sac.py --env=Reacher-v4 --learn_id=mujoco/sac/Reacher-v4
echo "---------Run SAC in Pusher-v4---------"
python mujoco_sac.py --env=Pusher-v4 --learn_id=mujoco/sac/Pusher-v4

# ========================= TD3 =========================
echo "---------Run TD3 in Hooper-v4---------"
python mujoco_td3.py --env=Hopper-v4 --learn_id=mujoco/td3/Hopper-v4
echo "---------Run TD3 in HalfCheetah-v4---------"
python mujoco_td3.py --env=HalfCheetah-v4 --learn_id=mujoco/td3/HalfCheetah-v4
echo "---------Run TD3 in Walker2d-v4---------"
python mujoco_td3.py --env=Walker2d-v4 --learn_id=mujoco/td3/Walker2d-v4
echo "---------Run TD3 in Ant-v4---------"
python mujoco_td3.py --env=Ant-v4 --learn_id=mujoco/td3/Ant-v4
echo "---------Run TD3 in Humanoid-v4---------"
python mujoco_td3.py --env=Humanoid-v4 --learn_id=mujoco/td3/Humanoid-v4
echo "---------Run TD3 in Swimmer-v4---------"
python mujoco_td3.py --env=Swimmer-v4 --learn_id=mujoco/td3/Swimmer-v4
echo "---------Run TD3 in Reacher-v4---------"
python mujoco_td3.py --env=Reacher-v4 --learn_id=mujoco/td3/Reacher-v4
echo "---------Run TD3 in Pusher-v4---------"
python mujoco_td3.py --env=Pusher-v4 --learn_id=mujoco/td3/Pusher-v4