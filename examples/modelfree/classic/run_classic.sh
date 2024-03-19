#!/bin/bash

# ========================= DQN =========================
echo "---------Run DQN in CartPole-v1---------"
python classic_dqn.py --env=CartPole-v1 --learn_id=modelfree/classic/CartPole-v1/dqn/seed10 --seed=10
python classic_dqn.py --env=CartPole-v1 --learn_id=modelfree/classic/CartPole-v1/dqn/seed20 --seed=20
python classic_dqn.py --env=CartPole-v1 --learn_id=modelfree/classic/CartPole-v1/dqn/seed30 --seed=30

# ========================= DDPG =========================
echo "---------Run DDPG in Pendulum-v1---------"
python classic_ddpg.py --env=Pendulum-v1 --learn_id=modelfree/classic/Pendulum-v1/ddpg/seed10 --seed=10
python classic_ddpg.py --env=Pendulum-v1 --learn_id=modelfree/classic/Pendulum-v1/ddpg/seed20 --seed=20
python classic_ddpg.py --env=Pendulum-v1 --learn_id=modelfree/classic/Pendulum-v1/ddpg/seed30 --seed=30

# ========================= PPO =========================
echo "--------- Run PPO in CartPole-v1 ---------"
python classic_ppo.py --env=CartPole-v1 --learn_id=modelfree/classic/CartPole-v1/ppo/seed10 --seed=10
python classic_ppo.py --env=CartPole-v1 --learn_id=modelfree/classic/CartPole-v1/ppo/seed20 --seed=20
python classic_ppo.py --env=CartPole-v1 --learn_id=modelfree/classic/CartPole-v1/ppo/seed30 --seed=30
echo "--------- Run PPO in Pendulum-v1 ---------"  # The result is bad.
python classic_ppo.py --env=Pendulum-v1 --learn_id=modelfree/classic/Pendulum-v1/ppo/seed10 --traj_length=512 --max_train_step=100000 --seed=10
python classic_ppo.py --env=Pendulum-v1 --learn_id=modelfree/classic/Pendulum-v1/ppo/seed20 --traj_length=512 --max_train_step=100000 --seed=20
python classic_ppo.py --env=Pendulum-v1 --learn_id=modelfree/classic/Pendulum-v1/ppo/seed30 --traj_length=512 --max_train_step=100000 --seed=30

# ========================= SAC =========================
echo "---------Run SAC in Pendulum-v1---------"
python classic_sac.py --env=Pendulum-v1 --learn_id=modelfree/classic/Pendulum-v1/sac/seed10 --seed=10
python classic_sac.py --env=Pendulum-v1 --learn_id=modelfree/classic/Pendulum-v1/sac/seed20 --seed=20
python classic_sac.py --env=Pendulum-v1 --learn_id=modelfree/classic/Pendulum-v1/sac/seed30 --seed=30

# ========================= TD3 =========================
echo "---------Run TD3 in Pendulum-v1---------"
python classic_td3.py --env=Pendulum-v1 --learn_id=modelfree/classic/Pendulum-v1/td3/seed10 --seed=10
python classic_td3.py --env=Pendulum-v1 --learn_id=modelfree/classic/Pendulum-v1/td3/seed20 --seed=20
python classic_td3.py --env=Pendulum-v1 --learn_id=modelfree/classic/Pendulum-v1/td3/seed30 --seed=30





