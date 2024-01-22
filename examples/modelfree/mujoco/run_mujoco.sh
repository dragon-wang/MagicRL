#!/bin/bash

# ========================= DDPG =========================
echo "---------Run DDPG in Hooper-v4---------"
python mujoco_ddpg.py --env=Hopper-v4 --learn_id=modelfree/mujoco/Hopper-v4/ddpg/seed10 --seed10
python mujoco_ddpg.py --env=Hopper-v4 --learn_id=modelfree/mujoco/Hopper-v4/ddpg/seed20 --seed20
python mujoco_ddpg.py --env=Hopper-v4 --learn_id=modelfree/mujoco/Hopper-v4/ddpg/seed30 --seed30
echo "---------Run DDPG in HalfCheetah-v4---------"
python mujoco_ddpg.py --env=HalfCheetah-v4 --learn_id=modelfree/mujoco/HalfCheetah-v4/ddpg/seed10 --seed10
python mujoco_ddpg.py --env=HalfCheetah-v4 --learn_id=modelfree/mujoco/HalfCheetah-v4/ddpg/seed20 --seed20
python mujoco_ddpg.py --env=HalfCheetah-v4 --learn_id=modelfree/mujoco/HalfCheetah-v4/ddpg/seed30 --seed30
echo "---------Run DDPG in Walker2d-v4---------"
python mujoco_ddpg.py --env=Walker2d-v4 --learn_id=modelfree/mujoco/Walker2d-v4/ddpg/seed10 --seed10
python mujoco_ddpg.py --env=Walker2d-v4 --learn_id=modelfree/mujoco/Walker2d-v4/ddpg/seed20 --seed20
python mujoco_ddpg.py --env=Walker2d-v4 --learn_id=modelfree/mujoco/Walker2d-v4/ddpg/seed30 --seed30
echo "---------Run DDPG in Ant-v4---------"
python mujoco_ddpg.py --env=Ant-v4 --learn_id=modelfree/mujoco/Ant-v4/ddpg/seed10 --seed10
python mujoco_ddpg.py --env=Ant-v4 --learn_id=modelfree/mujoco/Ant-v4/ddpg/seed20 --seed20
python mujoco_ddpg.py --env=Ant-v4 --learn_id=modelfree/mujoco/Ant-v4/ddpg/seed30 --seed30
echo "---------Run DDPG in Humanoid-v4---------"
python mujoco_ddpg.py --env=Humanoid-v4 --learn_id=modelfree/mujoco/Humanoid-v4/ddpg/seed10 --seed10
python mujoco_ddpg.py --env=Humanoid-v4 --learn_id=modelfree/mujoco/Humanoid-v4/ddpg/seed20 --seed20
python mujoco_ddpg.py --env=Humanoid-v4 --learn_id=modelfree/mujoco/Humanoid-v4/ddpg/seed30 --seed30
echo "---------Run DDPG in Swimmer-v4---------"
python mujoco_ddpg.py --env=Swimmer-v4 --learn_id=modelfree/mujoco/Swimmer-v4/ddpg/seed10 --seed10
python mujoco_ddpg.py --env=Swimmer-v4 --learn_id=modelfree/mujoco/Swimmer-v4/ddpg/seed20 --seed20
python mujoco_ddpg.py --env=Swimmer-v4 --learn_id=modelfree/mujoco/Swimmer-v4/ddpg/seed30 --seed30
echo "---------Run DDPG in Reacher-v4---------"
python mujoco_ddpg.py --env=Reacher-v4 --learn_id=modelfree/mujoco/Reacher-v4/ddpg/seed10 --seed10
python mujoco_ddpg.py --env=Reacher-v4 --learn_id=modelfree/mujoco/Reacher-v4/ddpg/seed20 --seed20
python mujoco_ddpg.py --env=Reacher-v4 --learn_id=modelfree/mujoco/Reacher-v4/ddpg/seed30 --seed30
echo "---------Run DDPG in Pusher-v4---------"
python mujoco_ddpg.py --env=Pusher-v4 --learn_id=modelfree/mujoco/Pusher-v4/ddpg/seed10 --seed10
python mujoco_ddpg.py --env=Pusher-v4 --learn_id=modelfree/mujoco/Pusher-v4/ddpg/seed20 --seed20
python mujoco_ddpg.py --env=Pusher-v4 --learn_id=modelfree/mujoco/Pusher-v4/ddpg/seed30 --seed30

# PPO
echo "--------- Run PPO in Hopper-v4 ---------"
python mujoco_ppo.py --env=Hopper-v4 --learn_id=modelfree/mujoco/Hopper-v4/ppo/seed10 --seed10
python mujoco_ppo.py --env=Hopper-v4 --learn_id=modelfree/mujoco/Hopper-v4/ppo/seed20 --seed20
python mujoco_ppo.py --env=Hopper-v4 --learn_id=modelfree/mujoco/Hopper-v4/ppo/seed30 --seed30
echo "--------- Run PPO in HalfCheetah-v4 ---------"
python mujoco_ppo.py --env=HalfCheetah-v4 --learn_id=modelfree/mujoco/HalfCheetah-v4/ppo/seed10 --seed10
python mujoco_ppo.py --env=HalfCheetah-v4 --learn_id=modelfree/mujoco/HalfCheetah-v4/ppo/seed20 --seed20
python mujoco_ppo.py --env=HalfCheetah-v4 --learn_id=modelfree/mujoco/HalfCheetah-v4/ppo/seed30 --seed30
echo "--------- Run PPO in Walker2d-v4 ---------"
python mujoco_ppo.py --env=Walker2d-v4 --learn_id=modelfree/mujoco/Walker2d-v4/ppo/seed10 --seed10
python mujoco_ppo.py --env=Walker2d-v4 --learn_id=modelfree/mujoco/Walker2d-v4/ppo/seed20 --seed20
python mujoco_ppo.py --env=Walker2d-v4 --learn_id=modelfree/mujoco/Walker2d-v4/ppo/seed30 --seed30

# ========================= SAC =========================
echo "---------Run SAC in Hooper-v4---------"
python mujoco_sac.py --env=Hopper-v4 --learn_id=modelfree/mujoco/Hopper-v4/sac/seed10 --seed10
python mujoco_sac.py --env=Hopper-v4 --learn_id=modelfree/mujoco/Hopper-v4/sac/seed20 --seed20
python mujoco_sac.py --env=Hopper-v4 --learn_id=modelfree/mujoco/Hopper-v4/sac/seed30 --seed30
echo "---------Run SAC in HalfCheetah-v4---------"
python mujoco_sac.py --env=HalfCheetah-v4 --learn_id=modelfree/mujoco/HalfCheetah-v4/sac/seed10 --seed10
python mujoco_sac.py --env=HalfCheetah-v4 --learn_id=modelfree/mujoco/HalfCheetah-v4/sac/seed20 --seed20
python mujoco_sac.py --env=HalfCheetah-v4 --learn_id=modelfree/mujoco/HalfCheetah-v4/sac/seed30 --seed30
echo "---------Run SAC in Walker2d-v4---------"
python mujoco_sac.py --env=Walker2d-v4 --learn_id=modelfree/mujoco/Walker2d-v4/sac/seed10 --seed10
python mujoco_sac.py --env=Walker2d-v4 --learn_id=modelfree/mujoco/Walker2d-v4/sac/seed20 --seed20
python mujoco_sac.py --env=Walker2d-v4 --learn_id=modelfree/mujoco/Walker2d-v4/sac/seed30 --seed30
echo "---------Run SAC in Ant-v4---------"
python mujoco_sac.py --env=Ant-v4 --learn_id=modelfree/mujoco/Ant-v4/sac/seed10 --seed10
python mujoco_sac.py --env=Ant-v4 --learn_id=modelfree/mujoco/Ant-v4/sac/seed20 --seed20
python mujoco_sac.py --env=Ant-v4 --learn_id=modelfree/mujoco/Ant-v4/sac/seed30 --seed30
echo "---------Run SAC in Humanoid-v4---------"
python mujoco_sac.py --env=Humanoid-v4 --learn_id=modelfree/mujoco/Humanoid-v4/sac/seed10 --seed10
python mujoco_sac.py --env=Humanoid-v4 --learn_id=modelfree/mujoco/Humanoid-v4/sac/seed20 --seed20
python mujoco_sac.py --env=Humanoid-v4 --learn_id=modelfree/mujoco/Humanoid-v4/sac/seed30 --seed30
echo "---------Run SAC in Swimmer-v4---------"
python mujoco_sac.py --env=Swimmer-v4 --learn_id=modelfree/mujoco/Swimmer-v4/sac/seed10 --seed10
python mujoco_sac.py --env=Swimmer-v4 --learn_id=modelfree/mujoco/Swimmer-v4/sac/seed20 --seed20
python mujoco_sac.py --env=Swimmer-v4 --learn_id=modelfree/mujoco/Swimmer-v4/sac/seed30 --seed30
echo "---------Run SAC in Reacher-v4---------"
python mujoco_sac.py --env=Reacher-v4 --learn_id=modelfree/mujoco/Reacher-v4/sac/seed10 --seed10
python mujoco_sac.py --env=Reacher-v4 --learn_id=modelfree/mujoco/Reacher-v4/sac/seed20 --seed20
python mujoco_sac.py --env=Reacher-v4 --learn_id=modelfree/mujoco/Reacher-v4/sac/seed30 --seed30
echo "---------Run SAC in Pusher-v4---------"
python mujoco_sac.py --env=Pusher-v4 --learn_id=modelfree/mujoco/Pusher-v4/sac/seed10 --seed10
python mujoco_sac.py --env=Pusher-v4 --learn_id=modelfree/mujoco/Pusher-v4/sac/seed20 --seed20
python mujoco_sac.py --env=Pusher-v4 --learn_id=modelfree/mujoco/Pusher-v4/sac/seed30 --seed30

# ========================= TD3 =========================
echo "---------Run TD3 in Hooper-v4---------"
python mujoco_td3.py --env=Hopper-v4 --learn_id=modelfree/mujoco/Hopper-v4/td3/seed10 --seed10
python mujoco_td3.py --env=Hopper-v4 --learn_id=modelfree/mujoco/Hopper-v4/td3/seed20 --seed20
python mujoco_td3.py --env=Hopper-v4 --learn_id=modelfree/mujoco/Hopper-v4/td3/seed30 --seed30
echo "---------Run TD3 in HalfCheetah-v4---------"
python mujoco_td3.py --env=HalfCheetah-v4 --learn_id=modelfree/mujoco/HalfCheetah-v4/td3/seed10 --seed10
python mujoco_td3.py --env=HalfCheetah-v4 --learn_id=modelfree/mujoco/HalfCheetah-v4/td3/seed20 --seed20
python mujoco_td3.py --env=HalfCheetah-v4 --learn_id=modelfree/mujoco/HalfCheetah-v4/td3/seed30 --seed30
echo "---------Run TD3 in Walker2d-v4---------"
python mujoco_td3.py --env=Walker2d-v4 --learn_id=modelfree/mujoco/Walker2d-v4/td3/seed10 --seed10
python mujoco_td3.py --env=Walker2d-v4 --learn_id=modelfree/mujoco/Walker2d-v4/td3/seed20 --seed20
python mujoco_td3.py --env=Walker2d-v4 --learn_id=modelfree/mujoco/Walker2d-v4/td3/seed30 --seed30
echo "---------Run TD3 in Ant-v4---------"
python mujoco_td3.py --env=Ant-v4 --learn_id=modelfree/mujoco/Ant-v4/td3/seed10 --seed10
python mujoco_td3.py --env=Ant-v4 --learn_id=modelfree/mujoco/Ant-v4/td3/seed20 --seed20
python mujoco_td3.py --env=Ant-v4 --learn_id=modelfree/mujoco/Ant-v4/td3/seed30 --seed30
echo "---------Run TD3 in Humanoid-v4---------"
python mujoco_td3.py --env=Humanoid-v4 --learn_id=modelfree/mujoco/Humanoid-v4/td3/seed10 --seed10
python mujoco_td3.py --env=Humanoid-v4 --learn_id=modelfree/mujoco/Humanoid-v4/td3/seed20 --seed20
python mujoco_td3.py --env=Humanoid-v4 --learn_id=modelfree/mujoco/Humanoid-v4/td3/seed30 --seed30
echo "---------Run TD3 in Swimmer-v4---------"
python mujoco_td3.py --env=Swimmer-v4 --learn_id=modelfree/mujoco/Swimmer-v4/td3/seed10 --seed10
python mujoco_td3.py --env=Swimmer-v4 --learn_id=modelfree/mujoco/Swimmer-v4/td3/seed20 --seed20
python mujoco_td3.py --env=Swimmer-v4 --learn_id=modelfree/mujoco/Swimmer-v4/td3/seed30 --seed30
echo "---------Run TD3 in Reacher-v4---------"
python mujoco_td3.py --env=Reacher-v4 --learn_id=modelfree/mujoco/Reacher-v4/td3/seed10 --seed10
python mujoco_td3.py --env=Reacher-v4 --learn_id=modelfree/mujoco/Reacher-v4/td3/seed20 --seed20
python mujoco_td3.py --env=Reacher-v4 --learn_id=modelfree/mujoco/Reacher-v4/td3/seed30 --seed30
echo "---------Run TD3 in Pusher-v4---------"
python mujoco_td3.py --env=Pusher-v4 --learn_id=modelfree/mujoco/Pusher-v4/td3/seed10 --seed10
python mujoco_td3.py --env=Pusher-v4 --learn_id=modelfree/mujoco/Pusher-v4/td3/seed20 --seed20
python mujoco_td3.py --env=Pusher-v4 --learn_id=modelfree/mujoco/Pusher-v4/td3/seed30 --seed30