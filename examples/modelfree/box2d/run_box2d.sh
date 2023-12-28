#!/bin/bash

# ========================= DQN =========================
echo "---------Run DQN in LunarLander-v2---------"
python box2d_dqn.py --env=LunarLander-v2 --learn_id=box2d/dqn/LunarLander-v2

# ========================= DDPG =========================
echo "---------Run DDPG in BipedalWalker-v3---------"
python box2d_ddpg.py -env=BipedalWalker-v3 --learn_id=box2d/ddpg/BipedalWalker-v3

# ========================= PPO =========================
echo "--------- Run PPO in LunarLander-v2 ---------"
python box2d_ppo.py --env=LunarLander-v2 --learn_id=box2d/ppo/LunarLander-v2
echo "--------- Run PPO in BipedalWalker-v3 ---------"
python box2d_ppo.py --env=BipedalWalker-v3 --learn_id=box2d/ppo/BipedalWalker-v3 --traj_length=1024 --max_train_step=1000000

# ========================= SAC =========================
echo "---------Run SAC in BipedalWalker-v3---------"
python box2d_sac.py --env=BipedalWalker-v3 --learn_id=box2d/sac/BipedalWalker-v3

# ========================= TD3 =========================
echo "---------Run TD3 in BipedalWalker-v3---------"
python box2d_td3.py --env=BipedalWalker-v3 --learn_id=box2d/td3/BipedalWalker-v3