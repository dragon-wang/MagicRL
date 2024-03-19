#!/bin/bash

# ========================= DQN =========================
echo "--------- Run DQN in PongNoFrameskip-v4 ---------" 
python atari_dqn.py --env=PongNoFrameskip-v4 --learn_id=modelfree/atari/Pong-v4/dqn/seed10 --train_num=8 --eval_num=8 --max_train_step=1000000 --seed=10
python atari_dqn.py --env=PongNoFrameskip-v4 --learn_id=modelfree/atari/Pong-v4/dqn/seed20 --train_num=8 --eval_num=8 --max_train_step=1000000 --seed=20
python atari_dqn.py --env=PongNoFrameskip-v4 --learn_id=modelfree/atari/Pong-v4/dqn/seed30 --train_num=8 --eval_num=8 --max_train_step=1000000 --seed=30
echo "--------- Run DQN in BreakoutNoFrameskip-v4 ---------" 
python atari_dqn.py --env=BreakoutNoFrameskip-v4 --learn_id=modelfree/atari/Breakout-v4/dqn/seed10 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=10
python atari_dqn.py --env=BreakoutNoFrameskip-v4 --learn_id=modelfree/atari/Breakout-v4/dqn/seed20 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=20
python atari_dqn.py --env=BreakoutNoFrameskip-v4 --learn_id=modelfree/atari/Breakout-v4/dqn/seed30 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=30
echo "--------- Run DQN in EnduroNoFrameskip-v4 ---------" 
python atari_dqn.py --env=EnduroNoFrameskip-v4 --learn_id=modelfree/atari/Enduro-v4/dqn/seed10 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=10
python atari_dqn.py --env=EnduroNoFrameskip-v4 --learn_id=modelfree/atari/Enduro-v4/dqn/seed20 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=20
python atari_dqn.py --env=EnduroNoFrameskip-v4 --learn_id=modelfree/atari/Enduro-v4/dqn/seed30 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=30

# ========================= PPO =========================
echo "--------- Run PPO in PongNoFrameskip-v4 ---------"  # the result is bad.
python atari_ppo.py --env=PongNoFrameskip-v4 --learn_id=modelfree/atari/Pong-v4/ppo/seed10 --train_num=8 --eval_num=8 --max_train_step=1000000 --seed=10
python atari_ppo.py --env=PongNoFrameskip-v4 --learn_id=modelfree/atari/Pong-v4/ppo/seed20 --train_num=8 --eval_num=8 --max_train_step=1000000 --seed=20
python atari_ppo.py --env=PongNoFrameskip-v4 --learn_id=modelfree/atari/Pong-v4/ppo/seed30 --train_num=8 --eval_num=8 --max_train_step=1000000 --seed=30
echo "--------- Run PPO in BreakoutNoFrameskip-v4 ---------" 
python atari_ppo.py --env=BreakoutNoFrameskip-v4 --learn_id=modelfree/atari/Breakout-v4/ppo/seed10 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=10
python atari_ppo.py --env=BreakoutNoFrameskip-v4 --learn_id=modelfree/atari/Breakout-v4/ppo/seed20 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=20
python atari_ppo.py --env=BreakoutNoFrameskip-v4 --learn_id=modelfree/atari/Breakout-v4/ppo/seed30 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=30
echo "--------- Run PPO in EnduroNoFrameskip-v4 ---------" 
python atari_ppo.py --env=EnduroNoFrameskip-v4 --learn_id=modelfree/atari/Enduro-v4/ppo/seed10 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=10
python atari_ppo.py --env=EnduroNoFrameskip-v4 --learn_id=modelfree/atari/Enduro-v4/ppo/seed20 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=20
python atari_ppo.py --env=EnduroNoFrameskip-v4 --learn_id=modelfree/atari/Enduro-v4/ppo/seed30 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=30