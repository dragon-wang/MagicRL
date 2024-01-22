#!/bin/bash

# ========================= DQN =========================
echo "---------Run DQN in LunarLander-v2---------"
python box2d_dqn.py --env=LunarLander-v2 --learn_id=modelfree/box2d/LunarLander-v2/dqn/seed10 --seed=10
python box2d_dqn.py --env=LunarLander-v2 --learn_id=modelfree/box2d/LunarLander-v2/dqn/seed20 --seed=20
python box2d_dqn.py --env=LunarLander-v2 --learn_id=modelfree/box2d/LunarLander-v2/dqn/seed30 --seed=30

# ========================= DDPG =========================
echo "---------Run DDPG in BipedalWalker-v3---------"
python box2d_ddpg.py -env=BipedalWalker-v3 --learn_id=modelfree/box2d/BipedalWalker-v3/ddpg/seed10 --seed=10
python box2d_ddpg.py -env=BipedalWalker-v3 --learn_id=modelfree/box2d/BipedalWalker-v3/ddpg/seed20 --seed=20
python box2d_ddpg.py -env=BipedalWalker-v3 --learn_id=modelfree/box2d/BipedalWalker-v3/ddpg/seed30 --seed=30

# ========================= PPO =========================
echo "--------- Run PPO in LunarLander-v2 ---------"
python box2d_ppo.py --env=LunarLander-v2 --learn_id=modelfree/box2d/LunarLander-v2/ppo/seed10 --seed=10
python box2d_ppo.py --env=LunarLander-v2 --learn_id=modelfree/box2d/LunarLander-v2/ppo/seed20 --seed=20
python box2d_ppo.py --env=LunarLander-v2 --learn_id=modelfree/box2d/LunarLander-v2/ppo/seed30 --seed=30
echo "--------- Run PPO in BipedalWalker-v3 ---------"
python box2d_ppo.py --env=BipedalWalker-v3 --learn_id=modelfree/box2d/BipedalWalker-v3/ppo/seed10 --traj_length=1024 --max_train_step=1000000 --seed=10
python box2d_ppo.py --env=BipedalWalker-v3 --learn_id=modelfree/box2d/BipedalWalker-v3/ppo/seed20 --traj_length=1024 --max_train_step=1000000 --seed=20
python box2d_ppo.py --env=BipedalWalker-v3 --learn_id=modelfree/box2d/BipedalWalker-v3/ppo/seed30 --traj_length=1024 --max_train_step=1000000 --seed=30

# ========================= SAC =========================
echo "---------Run SAC in BipedalWalker-v3---------"
python box2d_sac.py --env=BipedalWalker-v3 --learn_id=modelfree/box2d/BipedalWalker-v3/sac/seed10 --seed=10
python box2d_sac.py --env=BipedalWalker-v3 --learn_id=modelfree/box2d/BipedalWalker-v3/sac/seed20 --seed=20
python box2d_sac.py --env=BipedalWalker-v3 --learn_id=modelfree/box2d/BipedalWalker-v3/sac/seed30 --seed=30

# ========================= TD3 =========================
echo "---------Run TD3 in BipedalWalker-v3---------"
python box2d_td3.py --env=BipedalWalker-v3 --learn_id=modelfree/box2d/BipedalWalker-v3/td3/seed10 --seed=10
python box2d_td3.py --env=BipedalWalker-v3 --learn_id=modelfree/box2d/BipedalWalker-v3/td3/seed20 --seed=20
python box2d_td3.py --env=BipedalWalker-v3 --learn_id=modelfree/box2d/BipedalWalker-v3/td3/seed30 --seed=30