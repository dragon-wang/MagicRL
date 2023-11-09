#!/bin/bash

# ========================= DQN =========================
echo "---------Run DQN in CartPole-v1---------"
python classic_dqn.py --env=CartPole-v1 --learn_id=classic/dqn/CartPole-v1

# ========================= DDPG =========================
echo "---------Run DDPG in Pendulum-v1---------"
python classic_ddpg.py --env=Pendulum-v1 --learn_id=classic/ddpg/Pendulum-v1

# ========================= PPO =========================
echo "--------- Run PPO in CartPole-v1 ---------"
python classic_ppo.py --env=CartPole-v1 --learn_id=classic/ppo/CartPole-v1
echo "--------- Run PPO in Pendulum-v1 ---------"  # The result is bad.
python classic_ppo.py --env=Pendulum-v1 --learn_id=classic/ppo/Pendulum-v1 --traj_length=512 --max_train_step=100000


# ========================= SAC =========================
echo "---------Run SAC in Pendulum-v1---------"
python classic_sac.py --env=Pendulum-v1 --learn_id=classic/sac/Pendulum-v1

# ========================= TD3 =========================
echo "---------Run TD3 in Pendulum-v1---------"
python classic_td3.py --env=Pendulum-v1 --learn_id=classic/td3/Pendulum-v1





